"""
Feature extraction for SGSI-DTI.

Outputs:
  drug_cls_feat.pkl
  drug_token_feat.pkl
  prot_cls_feat.pkl
  prot_token_feat.pkl

Protein features are extracted with ESM-2 only.
No auxiliary feature view is exported.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "datasets"
MODEL_ROOT = PROJECT_ROOT / "models"

PROTEIN_MODEL_MAX_LENGTH = 1026
PROTEIN_TOKEN_BUDGET = PROTEIN_MODEL_MAX_LENGTH - 2
PROTEIN_LONG_CHUNK_SIZE = PROTEIN_TOKEN_BUDGET
PROTEIN_LONG_CHUNK_STRIDE = 768
PROTEIN_SHORT_BATCH_SIZE = 8


def _dataset_dir(dataset, split):
    path = DATASET_ROOT / dataset / split
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_csvs(path, split):
    fnames = ["source_train", "target_train", "target_test"] if split == "cluster" else ["train", "val", "test"]
    dfs = [pd.read_csv(path / f"{name}_with_id.csv") for name in fnames if (path / f"{name}_with_id.csv").exists()]
    if not dfs:
        raise ValueError(f"No CSV found in {path}")
    return pd.concat(dfs, ignore_index=True)


def _save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _safe_selfies(smiles):
    if pd.isna(smiles):
        return ""

    import selfies as sf

    text = str(smiles).strip()
    if not text:
        return ""
    try:
        return sf.encoder(text)
    except Exception:
        return None


_RDKIT_CHEM = None
_RDKIT_LOADED = False


def _get_rdkit_chem():
    global _RDKIT_CHEM, _RDKIT_LOADED
    if _RDKIT_LOADED:
        return _RDKIT_CHEM
    try:
        from rdkit import Chem
    except ImportError:
        Chem = None
    _RDKIT_CHEM = Chem
    _RDKIT_LOADED = True
    return _RDKIT_CHEM


def _canonicalize_smiles(smiles):
    text = "" if pd.isna(smiles) else str(smiles).strip()
    if not text:
        return ""

    chem = _get_rdkit_chem()
    if chem is None:
        return text

    mol = chem.MolFromSmiles(text)
    if mol is None:
        return None
    return chem.MolToSmiles(mol, canonical=True)


def _run_encoder(tokenizer, model, seq, device, max_length):
    inp = tokenizer(seq, return_tensors="pt", truncation=True, max_length=max_length)
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.no_grad():
        out = model(**inp)
    hidden = out.last_hidden_state[0].detach().cpu()
    attention_mask = inp.get("attention_mask")
    valid_len = int(attention_mask[0].sum().item()) if attention_mask is not None else int(hidden.size(0))
    return hidden[:valid_len]


def _run_encoder_batch(tokenizer, model, seqs, device, max_length):
    inp = tokenizer(seqs, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.no_grad():
        out = model(**inp)
    hidden = out.last_hidden_state.detach().cpu()
    attention_mask = inp.get("attention_mask")
    mask = attention_mask.cpu() if attention_mask is not None else torch.ones(hidden.size()[:2], dtype=torch.long)
    return hidden, mask


def _split_hidden_cls_and_tokens(hidden):
    if hidden.size(0) <= 2:
        cls = hidden[0].numpy().astype(np.float32)
        tokens = hidden[1:].numpy().astype(np.float32)
        if tokens.shape[0] == 0:
            tokens = hidden[:1].numpy().astype(np.float32)
        return cls, tokens
    return hidden[0].numpy().astype(np.float32), hidden[1:-1].numpy().astype(np.float32)


def _build_chunk_spans(seq_len, chunk_size, chunk_stride):
    if seq_len <= chunk_size:
        return [(0, seq_len)]

    spans = []
    start = 0
    while start < seq_len:
        end = min(start + chunk_size, seq_len)
        spans.append((start, end))
        if end >= seq_len:
            break
        start += chunk_stride

    final_start = max(seq_len - chunk_size, 0)
    final_span = (final_start, seq_len)
    if spans[-1] != final_span:
        spans.append(final_span)

    deduped = []
    seen = set()
    for span in spans:
        if span not in seen:
            deduped.append(span)
            seen.add(span)
    return deduped


def _compress_token_embeddings(tokens, token_budget):
    if tokens.shape[0] <= token_budget:
        return tokens.astype(np.float32)
    pooled = F.adaptive_avg_pool1d(
        torch.from_numpy(tokens.astype(np.float32)).transpose(0, 1).unsqueeze(0),
        token_budget,
    )
    return pooled.squeeze(0).transpose(0, 1).numpy().astype(np.float32)


def _aggregate_overlapping_tokens(seq_len, spans, chunk_tokens):
    hidden_dim = int(chunk_tokens[0].shape[1])
    token_sum = np.zeros((seq_len, hidden_dim), dtype=np.float32)
    token_count = np.zeros((seq_len, 1), dtype=np.float32)

    for (start, end), chunk_tok in zip(spans, chunk_tokens):
        expected_len = max(end - start, 0)
        usable = min(expected_len, int(chunk_tok.shape[0]))
        if usable <= 0:
            continue
        token_sum[start:start + usable] += chunk_tok[:usable]
        token_count[start:start + usable] += 1.0

    if np.any(token_count <= 0):
        uncovered = int((token_count[:, 0] <= 0).sum())
        raise ValueError(f"Token aggregation left {uncovered} uncovered positions.")
    return token_sum / token_count


def _extract_esm_protein_views(tokenizer, model, seq, device, max_length):
    seq = str(seq).strip()
    if not seq:
        raise ValueError("Empty protein sequence.")

    if len(seq) > PROTEIN_TOKEN_BUDGET:
        spans = _build_chunk_spans(len(seq), PROTEIN_LONG_CHUNK_SIZE, PROTEIN_LONG_CHUNK_STRIDE)
        chunk_reprs = []
        chunk_tokens = []
        for start, end in spans:
            chunk_hidden = _run_encoder(tokenizer, model, seq[start:end], device, max_length=max_length)
            chunk_cls, chunk_tok = _split_hidden_cls_and_tokens(chunk_hidden)
            chunk_reprs.append(chunk_cls)
            chunk_tokens.append(chunk_tok)
        merged_tokens = _aggregate_overlapping_tokens(len(seq), spans, chunk_tokens)
        merged_tokens = _compress_token_embeddings(merged_tokens, PROTEIN_TOKEN_BUDGET).astype(np.float16)
        weights = np.asarray([max(end - start, 1) for start, end in spans], dtype=np.float32)
        cls_repr = np.average(np.stack(chunk_reprs, axis=0), axis=0, weights=weights).astype(np.float32)
        return cls_repr, merged_tokens, True

    hidden = _run_encoder(tokenizer, model, seq, device, max_length=max_length)
    cls_repr, token_repr = _split_hidden_cls_and_tokens(hidden)
    return cls_repr.astype(np.float32), token_repr.astype(np.float16), False


def extract_drug_features(dataset, split):
    print("=" * 50 + "\nExtracting Drug Features (SELFormer)\n" + "=" * 50)
    path = _dataset_dir(dataset, split)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = _load_csvs(path, split)

    if "SELFIES" not in df.columns and "selfies" not in df.columns:
        if "SMILES" in df.columns:
            print("  Converting SMILES -> SELFIES...")
            df["_canonical_smiles"] = df["SMILES"].apply(_canonicalize_smiles)
            canonicalized = (
                df["_canonical_smiles"].notna()
                & df["SMILES"].notna()
                & (df["_canonical_smiles"].astype(str) != df["SMILES"].astype(str).str.strip())
            ).sum()
            if canonicalized > 0:
                print(f"  Canonicalized {int(canonicalized)} SMILES strings before SELFIES encoding.")
            df["SELFIES"] = df["_canonical_smiles"].apply(_safe_selfies)
            invalid_rows = df[df["SELFIES"].isna()][["dr_id", "SMILES"]].drop_duplicates(subset=["dr_id"])
            invalid_selfies = len(invalid_rows)
            if invalid_selfies > 0:
                print(f"  [Warning] Failed to convert {invalid_selfies} invalid SMILES to SELFIES; they will be skipped.")
        else:
            raise ValueError("CSV must contain SMILES or SELFIES.")
    smi_col = "SELFIES" if "SELFIES" in df.columns else "selfies"

    model_name = MODEL_ROOT / "drug" / "selformer"
    print(f"  Loading SELFormer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    unique = df[["dr_id", smi_col]].dropna(subset=[smi_col]).drop_duplicates(subset=["dr_id"]).sort_values("dr_id")
    ids, seqs = unique["dr_id"].tolist(), unique[smi_col].tolist()
    cls_d, tok_d = {}, {}

    for i in tqdm(range(0, len(ids), 32), desc="SELFormer"):
        b_ids, b_seqs = ids[i: i + 32], seqs[i: i + 32]
        try:
            inp = tokenizer(b_seqs, padding=True, truncation=True, max_length=512, return_tensors="pt")
            inp = {k: v.to(device) for k, v in inp.items()}
            with torch.no_grad():
                out = model(**inp)
            cls = out.last_hidden_state[:, 0, :].cpu().numpy()
            toks = out.last_hidden_state.cpu().numpy()
            mask = inp["attention_mask"].cpu().numpy()

            for j, did in enumerate(b_ids):
                cls_d[did] = cls[j]
                valid_len = int(mask[j].sum())
                tok_d[did] = toks[j][1: max(valid_len - 1, 1)].astype(np.float16)
        except Exception as e:
            print(f"[Error] batch {i}: {e}")

    for name, data in [("drug_cls_feat.pkl", cls_d), ("drug_token_feat.pkl", tok_d)]:
        with (path / name).open("wb") as f:
            pickle.dump(data, f)
        print(f"  Saved {name} ({len(data)} drugs)")

    manifest = {
        "dataset": dataset,
        "split": split,
        "feature_family": "drug",
        "encoder": str(model_name),
        "num_unique_inputs": len(ids),
        "num_saved_cls": len(cls_d),
        "num_saved_tokens": len(tok_d),
        "invalid_selfies": int(invalid_selfies) if "invalid_selfies" in locals() else 0,
    }
    _save_json(path / "drug_feature_manifest.json", manifest)

    if "invalid_rows" in locals() and invalid_selfies > 0:
        print("  Invalid SMILES skipped:")
        for row in invalid_rows.itertuples(index=False):
            print(f"    dr_id={int(row.dr_id)} smiles={row.SMILES}")
        _save_json(
            path / "drug_invalid_smiles.json",
            [{"dr_id": int(row.dr_id), "smiles": row.SMILES} for row in invalid_rows.itertuples(index=False)],
        )
    print("Drug features done!\n")


def extract_prot_features(dataset, split):
    print("=" * 50 + "\nExtracting Protein Features (ESM-2)\n" + "=" * 50)
    path = _dataset_dir(dataset, split)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = _load_csvs(path, split)

    if "Protein" not in df.columns:
        raise ValueError(
            "ESM-only protein extraction requires a 'Protein' column. "
            "This pipeline no longer falls back to 'Seq'."
        )
    seq_col = "Protein"
    print("  Using column: 'Protein'")

    unique = df[["pr_id", seq_col]].dropna().drop_duplicates(subset=["pr_id"]).sort_values("pr_id")
    ids, seqs = unique["pr_id"].tolist(), unique[seq_col].tolist()

    model_name = MODEL_ROOT / "protein" / "esm2_model"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    print(f"  Loaded: {model_name}")
    print(f"  Processing {len(ids)} unique proteins...")
    print(f"  Protein max input length: {PROTEIN_TOKEN_BUDGET} residues/tokens")

    cls_d, tok_d = {}, {}
    chunk_merged_count = 0
    failed = []

    short_items = []
    long_items = []
    for pid, raw_seq in zip(ids, seqs):
        seq = str(raw_seq).strip()
        if not seq:
            failed.append({"pr_id": int(pid), "reason": "empty_sequence"})
            continue
        if len(seq) > PROTEIN_TOKEN_BUDGET:
            long_items.append((pid, seq))
        else:
            short_items.append((pid, seq))

    for start in tqdm(range(0, len(short_items), PROTEIN_SHORT_BATCH_SIZE), desc="ESM-2 short", total=(len(short_items) + PROTEIN_SHORT_BATCH_SIZE - 1) // PROTEIN_SHORT_BATCH_SIZE):
        batch_items = short_items[start:start + PROTEIN_SHORT_BATCH_SIZE]
        batch_ids = [pid for pid, _ in batch_items]
        batch_seqs = [seq for _, seq in batch_items]
        try:
            hidden_batch, mask_batch = _run_encoder_batch(
                tokenizer=tokenizer,
                model=model,
                seqs=batch_seqs,
                device=device,
                max_length=PROTEIN_MODEL_MAX_LENGTH,
            )
            for row_idx, pid in enumerate(batch_ids):
                valid_len = int(mask_batch[row_idx].sum().item())
                hidden = hidden_batch[row_idx, :valid_len]
                cls_repr, token_repr = _split_hidden_cls_and_tokens(hidden)
                cls_d[pid] = cls_repr.astype(np.float32)
                tok_d[pid] = token_repr.astype(np.float16)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                for pid, seq in batch_items:
                    try:
                        cls_repr, token_repr, used_chunk_merge = _extract_esm_protein_views(
                            tokenizer=tokenizer,
                            model=model,
                            seq=seq,
                            device=device,
                            max_length=PROTEIN_MODEL_MAX_LENGTH,
                        )
                        cls_d[pid] = cls_repr
                        tok_d[pid] = token_repr
                        chunk_merged_count += int(used_chunk_merge)
                    except Exception as inner_e:
                        failed.append({"pr_id": int(pid), "reason": str(inner_e)})
            else:
                for pid, _ in batch_items:
                    failed.append({"pr_id": int(pid), "reason": str(e)})

    for pid, seq in tqdm(long_items, desc="ESM-2 long", total=len(long_items)):
        try:
            cls_repr, token_repr, used_chunk_merge = _extract_esm_protein_views(
                tokenizer=tokenizer,
                model=model,
                seq=seq,
                device=device,
                max_length=PROTEIN_MODEL_MAX_LENGTH,
            )
            cls_d[pid] = cls_repr
            tok_d[pid] = token_repr
            chunk_merged_count += int(used_chunk_merge)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
            failed.append({"pr_id": int(pid), "reason": str(e)})
        except Exception as e:
            failed.append({"pr_id": int(pid), "reason": str(e)})

    for name, data in [("prot_cls_feat.pkl", cls_d), ("prot_token_feat.pkl", tok_d)]:
        with (path / name).open("wb") as f:
            pickle.dump(data, f)
        print(f"  Saved {name} ({len(data)} proteins)")

    manifest = {
        "dataset": dataset,
        "split": split,
        "feature_family": "protein",
        "encoder": str(model_name),
        "source_column": seq_col,
        "num_unique_inputs": len(ids),
        "num_saved_cls": len(cls_d),
        "num_saved_tokens": len(tok_d),
        "num_short_batch_encoded": len(short_items),
        "num_long_chunk_merged": int(chunk_merged_count),
        "num_failed": len(failed),
        "protein_model_max_length": PROTEIN_MODEL_MAX_LENGTH,
        "protein_token_budget": PROTEIN_TOKEN_BUDGET,
        "long_chunk_size": PROTEIN_LONG_CHUNK_SIZE,
        "long_chunk_stride": PROTEIN_LONG_CHUNK_STRIDE,
    }
    _save_json(path / "protein_feature_manifest.json", manifest)
    if failed:
        _save_json(path / "protein_feature_failures.json", failed)
        print(f"  Failed proteins: {len(failed)}")
    if chunk_merged_count > 0:
        print(f"  Long proteins handled with internal chunk-merge: {chunk_merged_count}")
    print("ESM-2 features done!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bindingdb")
    parser.add_argument("--split", default="random")
    parser.add_argument("--task", default="all", choices=["all", "drug", "protein"])
    args = parser.parse_args()

    if args.task in ["all", "drug"]:
        extract_drug_features(args.dataset, args.split)
    if args.task in ["all", "protein"]:
        extract_prot_features(args.dataset, args.split)
