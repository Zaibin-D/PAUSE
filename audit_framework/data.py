"""Input discovery and table preparation for the PAUSE audit framework."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
except ImportError:  # pragma: no cover - native drug support becomes unavailable
    Chem = DataStructs = rdFingerprintGenerator = None


AUDIT_NAME = "pause_audit_inputs.csv"
NATIVE_SUPPORT_NAME = "native_source_support.pkl"
TARGET_SEQUENCE_SUPPORT_NAME = "target_sequence_support.pkl"
DIRECT_TARGET_SUPPORT_NAME = "direct_target_support.pkl"
JOINT_SUPPORT_NAME = "joint_source_support.pkl"
DIRECT_TARGET_HITS_NAME = "direct_target_to_source.tsv"
SUPPORT_NEIGHBOURS = 5
MMSEQS_MIN_IDENTITY = 30.0
MMSEQS_MIN_QUERY_COVERAGE = 0.50
MMSEQS_MAX_EVALUE = 1.0e-3

DEFAULT_TEST_ROOTS = (
    "audit_framework/cache/test_audits/pace",
    "audit_framework/cache/test_audits/tapb",
    "audit_framework/cache/test_audits/drugban",
)
DEFAULT_DATASETS = ("biosnap", "bindingdb", "human")


def model_name(root: str | Path) -> str:
    name = Path(root).name.lower()
    if "tapb" in name:
        return "TAPB"
    if "drugban" in name:
        return "DrugBAN"
    if "pace" in name:
        return "PACE"
    if "can_only" in name:
        return "CAN-only"
    return Path(root).name


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    )


def _sigmoid(values: pd.Series) -> np.ndarray:
    x = np.asarray(values, dtype=float).clip(-40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def _canonical_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        number = float(text)
    except (TypeError, ValueError):
        return text
    if np.isfinite(number) and number.is_integer():
        return str(int(number))
    return text


def _id_keys(values: pd.Series) -> pd.Series:
    return values.map(_canonical_id)


def _pair_keys(left: pd.Series, right: pd.Series) -> pd.Series:
    return pd.Series(
        list(zip(_id_keys(left), _id_keys(right))),
        index=left.index,
        dtype=object,
    )


def _entity_cluster_map(
    tables: list[pd.DataFrame],
    id_column: str,
    cluster_column: str,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for table in tables:
        if id_column not in table or cluster_column not in table:
            continue
        ids = _id_keys(table[id_column])
        clusters = _id_keys(table[cluster_column])
        for entity_id, cluster_id in zip(ids, clusters):
            if entity_id and cluster_id:
                mapping.setdefault(entity_id, cluster_id)
    return mapping


def _read_support_columns(path: Path) -> pd.DataFrame:
    columns = pd.read_csv(path, nrows=0).columns
    selected = [
        column
        for column in (
            "dr_id",
            "pr_id",
            "drug_cluster",
            "target_cluster",
        )
        if column in columns
    ]
    return pd.read_csv(path, usecols=selected)


@lru_cache(maxsize=32)
def _source_support_payload(
    dataset_root: str,
    dataset: str,
    split: str,
) -> dict[str, object] | None:
    split_dir = Path(dataset_root) / dataset / split
    train_name = "source_train_with_id.csv" if split == "cluster" else "train_with_id.csv"
    train_path = split_dir / train_name
    if not train_path.exists():
        return None

    train = _read_support_columns(train_path)
    if not {"dr_id", "pr_id"}.issubset(train.columns):
        return None
    related_tables = [train]
    for path in sorted(split_dir.glob("*_with_id.csv")):
        if path == train_path:
            continue
        try:
            related_tables.append(_read_support_columns(path))
        except (OSError, pd.errors.ParserError):
            continue

    drug_keys = _id_keys(train["dr_id"])
    target_keys = _id_keys(train["pr_id"])
    pair_keys = _pair_keys(train["dr_id"], train["pr_id"])
    drug_clusters = _entity_cluster_map(
        related_tables,
        "dr_id",
        "drug_cluster",
    )
    target_clusters = _entity_cluster_map(
        related_tables,
        "pr_id",
        "target_cluster",
    )
    train_drug_clusters = drug_keys.map(drug_clusters).fillna("")
    train_target_clusters = target_keys.map(target_clusters).fillna("")
    valid_cluster_pair = train_drug_clusters.ne("") & train_target_clusters.ne("")
    cluster_pairs = pd.Series(
        list(zip(train_drug_clusters, train_target_clusters)),
        index=train.index,
        dtype=object,
    )
    return {
        "drug_count": drug_keys.value_counts().to_dict(),
        "target_count": target_keys.value_counts().to_dict(),
        "pair_count": pair_keys.value_counts().to_dict(),
        "drug_cluster_map": drug_clusters,
        "target_cluster_map": target_clusters,
        "drug_cluster_count": train_drug_clusters.loc[
            train_drug_clusters.ne("")
        ].value_counts().to_dict(),
        "target_cluster_count": train_target_clusters.loc[
            train_target_clusters.ne("")
        ].value_counts().to_dict(),
        "cluster_pair_count": cluster_pairs.loc[
            valid_cluster_pair
        ].value_counts().to_dict(),
        "source_train_path": str(train_path.resolve()),
        "source_train_rows": int(len(train)),
        "source_columns": tuple(train.columns),
    }


def _nearest_source_distances(
    feature_payload: dict[object, object],
    source_ids: set[str],
) -> tuple[dict[str, float], dict[str, float]]:
    rows = []
    keys = []
    for entity_id, value in feature_payload.items():
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size and np.isfinite(array).all():
            keys.append(_canonical_id(entity_id))
            rows.append(array)
    if not rows:
        return {}, {}
    matrix = np.vstack(rows).astype(np.float32)
    source_positions = [
        index for index, key in enumerate(keys) if key in source_ids
    ]
    if len(source_positions) < 2:
        return {}, {}

    source = matrix[source_positions]
    scaler = StandardScaler()
    source_scaled = scaler.fit_transform(source)
    all_scaled = scaler.transform(matrix)
    component_n = min(32, source_scaled.shape[0] - 1, source_scaled.shape[1])
    if component_n >= 2 and source_scaled.shape[1] > component_n:
        reducer = PCA(
            n_components=component_n,
            svd_solver="randomized",
            random_state=2026,
        )
        source_projected = reducer.fit_transform(source_scaled)
        all_projected = reducer.transform(all_scaled)
    else:
        source_projected = source_scaled
        all_projected = all_scaled
        component_n = source_projected.shape[1]

    neighbour_n = min(5, len(source_projected))
    neighbours = NearestNeighbors(
        n_neighbors=neighbour_n,
        metric="euclidean",
        algorithm="auto",
    )
    neighbours.fit(source_projected)
    distances, _ = neighbours.kneighbors(all_projected)
    scale = max(float(np.sqrt(max(component_n, 1))), 1.0)
    nearest = distances[:, 0] / scale
    density = distances.mean(axis=1) / scale
    return (
        {key: float(value) for key, value in zip(keys, nearest)},
        {key: float(value) for key, value in zip(keys, density)},
    )


def _morgan_source_distances(
    entity_path: Path,
    source_ids: set[str],
) -> tuple[dict[str, float], dict[str, float]]:
    if (
        Chem is None
        or DataStructs is None
        or rdFingerprintGenerator is None
        or not entity_path.exists()
    ):
        return {}, {}
    table = pd.read_csv(entity_path)
    if "dr_id" not in table:
        return {}, {}
    smiles_column = next(
        (
            column
            for column in (
                "canonical_smiles",
                "selected_component_smiles",
                "smiles",
            )
            if column in table
        ),
        None,
    )
    if smiles_column is None:
        return {}, {}

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
    )
    keys: list[str] = []
    fingerprints = []
    for entity_id, smiles in table[["dr_id", smiles_column]].itertuples(
        index=False,
        name=None,
    ):
        if pd.isna(smiles):
            continue
        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is None:
            continue
        keys.append(_canonical_id(entity_id))
        fingerprints.append(generator.GetFingerprint(molecule))
    source_fingerprints = [
        fingerprint
        for key, fingerprint in zip(keys, fingerprints)
        if key in source_ids
    ]
    if not source_fingerprints:
        return {}, {}

    neighbour_n = min(5, len(source_fingerprints))
    nearest: dict[str, float] = {}
    density: dict[str, float] = {}
    for key, fingerprint in zip(keys, fingerprints):
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(
                fingerprint,
                source_fingerprints,
            ),
            dtype=float,
        )
        if not similarities.size:
            continue
        top = np.partition(
            similarities,
            max(len(similarities) - neighbour_n, 0),
        )[-neighbour_n:]
        nearest[key] = float(1.0 - similarities.max())
        density[key] = float(1.0 - top.mean())
    return nearest, density


def _morgan_source_neighbours(
    entity_path: Path,
    source_ids: set[str],
    *,
    neighbours: int = SUPPORT_NEIGHBOURS,
) -> dict[str, tuple[tuple[str, float], ...]]:
    """Return fixed top-k source-drug neighbours without outcome labels."""

    if (
        Chem is None
        or DataStructs is None
        or rdFingerprintGenerator is None
        or not entity_path.exists()
    ):
        return {}
    table = pd.read_csv(entity_path)
    if "dr_id" not in table:
        return {}
    smiles_column = next(
        (
            column
            for column in (
                "canonical_smiles",
                "selected_component_smiles",
                "smiles",
            )
            if column in table
        ),
        None,
    )
    if smiles_column is None:
        return {}

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
    )
    keys: list[str] = []
    fingerprints = []
    for entity_id, smiles in table[["dr_id", smiles_column]].itertuples(
        index=False,
        name=None,
    ):
        if pd.isna(smiles):
            continue
        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is None:
            continue
        keys.append(_canonical_id(entity_id))
        fingerprints.append(generator.GetFingerprint(molecule))

    source = [
        (key, fingerprint)
        for key, fingerprint in zip(keys, fingerprints)
        if key in source_ids
    ]
    if not source:
        return {}
    source_keys = [key for key, _ in source]
    source_fingerprints = [fingerprint for _, fingerprint in source]
    neighbour_n = min(int(neighbours), len(source))
    output: dict[str, tuple[tuple[str, float], ...]] = {}
    for key, fingerprint in zip(keys, fingerprints):
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(
                fingerprint,
                source_fingerprints,
            ),
            dtype=float,
        )
        order = np.argsort(-similarities, kind="stable")[:neighbour_n]
        output[key] = tuple(
            (source_keys[int(index)], float(similarities[int(index)]))
            for index in order
        )
    return output


def _cosine_source_distances(
    feature_path: Path,
    source_ids: set[str],
    *,
    feature_dim: int | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    if not feature_path.exists():
        return {}, {}
    try:
        with feature_path.open("rb") as handle:
            feature_payload = pickle.load(handle)
    except (OSError, pickle.UnpicklingError):
        return {}, {}

    keys = []
    rows = []
    for entity_id, value in feature_payload.items():
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if feature_dim is not None:
            if array.size < int(feature_dim):
                continue
            array = array[: int(feature_dim)]
        if array.size and np.isfinite(array).all():
            keys.append(_canonical_id(entity_id))
            rows.append(array)
    if not rows:
        return {}, {}
    matrix = np.vstack(rows).astype(np.float32)
    source_positions = [
        index for index, key in enumerate(keys) if key in source_ids
    ]
    if not source_positions:
        return {}, {}

    neighbour_n = min(5, len(source_positions))
    neighbours = NearestNeighbors(
        n_neighbors=neighbour_n,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    neighbours.fit(matrix[source_positions])
    distances, _ = neighbours.kneighbors(matrix)
    return (
        {
            key: float(value)
            for key, value in zip(keys, distances[:, 0])
        },
        {
            key: float(value)
            for key, value in zip(keys, distances.mean(axis=1))
        },
    )


@lru_cache(maxsize=32)
def _prior_distance_payload(
    dataset_root: str,
    dataset: str,
    split: str,
) -> dict[str, dict[str, float]] | None:
    source = _source_support_payload(dataset_root, dataset, split)
    if source is None:
        return None
    split_dir = Path(dataset_root) / dataset / split
    pime_dir = split_dir / "pime"
    if not pime_dir.exists():
        fallback = Path(dataset_root) / dataset / "cluster" / "pime"
        pime_dir = fallback if fallback.exists() else pime_dir
    drug_path = pime_dir / "drug_prior_feat.pkl"
    target_path = pime_dir / "target_prior_feat.pkl"
    if not drug_path.exists() or not target_path.exists():
        return None
    try:
        with drug_path.open("rb") as handle:
            drug_features = pickle.load(handle)
        with target_path.open("rb") as handle:
            target_features = pickle.load(handle)
    except (OSError, pickle.UnpicklingError):
        return None

    drug_nearest, drug_density = _nearest_source_distances(
        drug_features,
        set(source["drug_count"]),
    )
    target_nearest, target_density = _nearest_source_distances(
        target_features,
        set(source["target_count"]),
    )
    return {
        "drug_nearest": drug_nearest,
        "drug_density": drug_density,
        "target_nearest": target_nearest,
        "target_density": target_density,
    }


@lru_cache(maxsize=32)
def compute_native_distance_payload(
    dataset_root: str,
    dataset: str,
    split: str,
) -> dict[str, dict[str, float]] | None:
    source = _source_support_payload(dataset_root, dataset, split)
    if source is None:
        return None
    split_dir = Path(dataset_root) / dataset / split
    pime_dir = split_dir / "pime"
    drug_nearest, drug_density = _morgan_source_distances(
        pime_dir / "drug_entity.csv",
        set(source["drug_count"]),
    )
    target_path = split_dir / "prot_cls_feat.pkl"
    target_feature_dim = None
    if not target_path.exists():
        target_path = pime_dir / "target_prior_feat.pkl"
        target_feature_dim = 1280
    target_nearest, target_density = _cosine_source_distances(
        target_path,
        set(source["target_count"]),
        feature_dim=target_feature_dim,
    )
    if not any(
        (drug_nearest, drug_density, target_nearest, target_density)
    ):
        return None
    return {
        "drug_nearest": drug_nearest,
        "drug_density": drug_density,
        "target_nearest": target_nearest,
        "target_density": target_density,
    }


def _mmseqs_query_id(value: object) -> str:
    text = str(value).strip()
    if text.startswith("pr_id="):
        text = text.split("=", 1)[1]
    return _canonical_id(text)


def _read_direct_target_hits(path: Path) -> pd.DataFrame:
    columns = (
        "query",
        "target",
        "identity",
        "alignment_length",
        "query_start",
        "query_end",
        "target_start",
        "target_end",
        "evalue",
        "bits",
        "query_length",
        "target_length",
    )
    hits = pd.read_csv(
        path,
        sep="\t",
        names=columns,
        usecols=list(columns),
    )
    for column in (
        "identity",
        "alignment_length",
        "evalue",
        "query_length",
        "target_length",
    ):
        hits[column] = pd.to_numeric(hits[column], errors="coerce")
    hits["query_id"] = hits["query"].map(_mmseqs_query_id)
    hits["target_id"] = hits["target"].map(_mmseqs_query_id)
    hits["query_coverage"] = (
        hits["alignment_length"] / hits["query_length"]
    )
    hits["target_coverage"] = (
        hits["alignment_length"] / hits["target_length"]
    )
    hits["support_score"] = (
        hits["identity"].clip(0.0, 100.0) / 100.0
    ) * hits[["query_coverage", "target_coverage"]].min(
        axis=1
    ).clip(0.0, 1.0)
    return hits.replace([np.inf, -np.inf], np.nan)


@lru_cache(maxsize=32)
def compute_direct_target_support_payload(
    dataset_root: str,
    dataset: str,
    split: str,
) -> dict[str, object] | None:
    """Build direct all-target to source-target MMseqs2 support."""

    source = _source_support_payload(dataset_root, dataset, split)
    if source is None:
        return None
    pime_dir = Path(dataset_root) / dataset / split / "pime"
    hits_path = pime_dir / "mmseqs" / DIRECT_TARGET_HITS_NAME
    entity_path = pime_dir / "target_entity.csv"
    if not hits_path.exists() or not entity_path.exists():
        return None
    try:
        hits = _read_direct_target_hits(hits_path)
        entities = pd.read_csv(entity_path, usecols=["pr_id"])
    except (OSError, pd.errors.ParserError, ValueError):
        return None

    source_ids = set(source["target_count"])
    hits = hits.loc[
        hits["target_id"].isin(source_ids)
        & hits["query_id"].ne("")
        & hits["target_id"].ne("")
        & hits["support_score"].notna()
    ].copy()
    hits = hits.sort_values(
        ["query_id", "support_score", "bits"],
        ascending=[True, False, False],
        kind="stable",
    )
    hits = hits.drop_duplicates(["query_id", "target_id"], keep="first")

    nearest: dict[str, float] = {}
    density: dict[str, float] = {}
    neighbours: dict[str, tuple[tuple[str, float], ...]] = {}
    grouped = {
        query_id: group.head(SUPPORT_NEIGHBOURS)
        for query_id, group in hits.groupby("query_id", sort=False)
    }
    for entity_id in entities["pr_id"].map(_canonical_id):
        group = grouped.get(entity_id)
        if group is None:
            selected: tuple[tuple[str, float], ...] = ()
            scores: list[float] = []
        else:
            selected = tuple(
                (str(target_id), float(np.clip(score, 0.0, 1.0)))
                for target_id, score in group[
                    ["target_id", "support_score"]
                ].itertuples(index=False, name=None)
            )
            scores = [score for _, score in selected]
        padded = scores + [0.0] * (SUPPORT_NEIGHBOURS - len(scores))
        nearest[entity_id] = float(1.0 - max(scores, default=0.0))
        density[entity_id] = float(1.0 - np.mean(padded))
        neighbours[entity_id] = selected
    return {
        "target_nearest": nearest,
        "target_density": density,
        "target_neighbours": neighbours,
        "neighbours": SUPPORT_NEIGHBOURS,
        "score_definition": "identity_fraction_x_min_query_target_coverage",
    }


@lru_cache(maxsize=32)
def _direct_target_support_payload(
    dataset_root: str,
    dataset: str,
    split: str,
) -> dict[str, object] | None:
    support_path = (
        Path(dataset_root)
        / dataset
        / split
        / "pime"
        / DIRECT_TARGET_SUPPORT_NAME
    )
    if support_path.exists():
        try:
            with support_path.open("rb") as handle:
                payload = pickle.load(handle)
        except (OSError, pickle.UnpicklingError, EOFError):
            payload = None
        required = {
            "target_nearest",
            "target_density",
            "target_neighbours",
        }
        if isinstance(payload, dict) and required.issubset(payload):
            return payload
    return None


@lru_cache(maxsize=32)
def compute_joint_support_payload(
    dataset_root: str,
    dataset: str,
    split: str,
) -> dict[str, object] | None:
    """Build model-independent neighbours for source-train joint support."""

    source = _source_support_payload(dataset_root, dataset, split)
    direct_target = _direct_target_support_payload(
        dataset_root,
        dataset,
        split,
    )
    if source is None or direct_target is None:
        return None
    pime_dir = Path(dataset_root) / dataset / split / "pime"
    drug_neighbours = _morgan_source_neighbours(
        pime_dir / "drug_entity.csv",
        set(source["drug_count"]),
    )
    if not drug_neighbours:
        return None
    return {
        "drug_neighbours": drug_neighbours,
        "target_neighbours": direct_target["target_neighbours"],
        "source_pairs": set(source["pair_count"]),
        "neighbours": SUPPORT_NEIGHBOURS,
        "drug_metric": "morgan_radius2_2048_tanimoto",
        "target_metric": direct_target.get("score_definition", ""),
        "exact_query_pair_excluded": True,
    }


@lru_cache(maxsize=32)
def _joint_support_payload(
    dataset_root: str,
    dataset: str,
    split: str,
) -> dict[str, object] | None:
    support_path = (
        Path(dataset_root)
        / dataset
        / split
        / "pime"
        / JOINT_SUPPORT_NAME
    )
    if support_path.exists():
        try:
            with support_path.open("rb") as handle:
                payload = pickle.load(handle)
        except (OSError, pickle.UnpicklingError, EOFError):
            payload = None
        required = {
            "drug_neighbours",
            "target_neighbours",
            "source_pairs",
        }
        if isinstance(payload, dict) and required.issubset(payload):
            return payload
    return None


def joint_nearest_train_distance(
    drug_id: object,
    target_id: object,
    payload: dict[str, object],
) -> float:
    """Distance to the closest observed similar pair, excluding exact pair."""

    query_drug = _canonical_id(drug_id)
    query_target = _canonical_id(target_id)
    drug_neighbours = payload.get("drug_neighbours", {}).get(query_drug, ())
    target_neighbours = payload.get("target_neighbours", {}).get(
        query_target,
        (),
    )
    source_pairs = payload.get("source_pairs", set())
    best = 0.0
    for source_drug, drug_similarity in drug_neighbours:
        for source_target, target_similarity in target_neighbours:
            source_pair = (
                _canonical_id(source_drug),
                _canonical_id(source_target),
            )
            if source_pair == (query_drug, query_target):
                continue
            if source_pair in source_pairs:
                best = max(
                    best,
                    float(drug_similarity) * float(target_similarity),
                )
    return float(1.0 - np.clip(best, 0.0, 1.0))


@lru_cache(maxsize=32)
def compute_target_sequence_support_payload(
    dataset_root: str,
    dataset: str,
    split: str,
) -> dict[str, dict[str, float]] | None:
    """Build fixed target-family support from source-train MMseqs2 hits."""

    source = _source_support_payload(dataset_root, dataset, split)
    if source is None:
        return None
    pime_dir = Path(dataset_root) / dataset / split / "pime"
    mmseqs_paths = sorted((pime_dir / "mmseqs").glob("*_vs_sprot.tsv"))
    entity_path = pime_dir / "target_entity.csv"
    if not mmseqs_paths or not entity_path.exists():
        return None

    columns = (
        "query",
        "hit",
        "identity",
        "alignment_length",
        "query_start",
        "query_end",
        "target_start",
        "target_end",
        "evalue",
        "bits",
        "query_length",
        "target_length",
    )
    try:
        hits = pd.read_csv(
            mmseqs_paths[0],
            sep="\t",
            names=columns,
            usecols=list(columns),
        )
        entities = pd.read_csv(entity_path, usecols=["pr_id"])
    except (OSError, pd.errors.ParserError, ValueError):
        return None
    for column in (
        "identity",
        "alignment_length",
        "evalue",
        "query_length",
        "target_length",
    ):
        hits[column] = pd.to_numeric(hits[column], errors="coerce")
    hits["query_id"] = hits["query"].map(_mmseqs_query_id)
    hits["query_coverage"] = (
        hits["alignment_length"] / hits["query_length"]
    )
    hits["target_coverage"] = (
        hits["alignment_length"] / hits["target_length"]
    )
    qualifying = hits.loc[
        hits["identity"].ge(MMSEQS_MIN_IDENTITY)
        & hits["query_coverage"].ge(MMSEQS_MIN_QUERY_COVERAGE)
        & hits["evalue"].le(MMSEQS_MAX_EVALUE)
    ].copy()
    source_ids = set(source["target_count"])
    reference_hits = set(
        qualifying.loc[
            qualifying["query_id"].isin(source_ids),
            "hit",
        ].astype(str)
    )
    qualifying = qualifying.loc[
        qualifying["hit"].astype(str).isin(reference_hits)
    ].copy()
    qualifying["support_score"] = (
        qualifying["identity"].clip(0.0, 100.0) / 100.0
    ) * qualifying[["query_coverage", "target_coverage"]].min(
        axis=1
    ).clip(0.0, 1.0)
    best = qualifying.groupby("query_id")["support_score"].max()
    family_count = qualifying.groupby("query_id")["hit"].nunique()

    entity_ids = entities["pr_id"].map(_canonical_id)
    distance: dict[str, float] = {}
    unseen: dict[str, float] = {}
    sparsity: dict[str, float] = {}
    for entity_id in entity_ids:
        score = float(best.get(entity_id, 0.0))
        count = int(family_count.get(entity_id, 0))
        distance[entity_id] = float(1.0 - np.clip(score, 0.0, 1.0))
        unseen[entity_id] = float(count == 0)
        sparsity[entity_id] = float(1.0 / (1.0 + count))
    return {
        "target_distance": distance,
        "target_family_unseen": unseen,
        "target_family_sparsity": sparsity,
    }


@lru_cache(maxsize=32)
def _target_sequence_support_payload(
    dataset_root: str,
    dataset: str,
    split: str,
) -> dict[str, dict[str, float]] | None:
    support_path = (
        Path(dataset_root)
        / dataset
        / split
        / "pime"
        / TARGET_SEQUENCE_SUPPORT_NAME
    )
    if support_path.exists():
        try:
            with support_path.open("rb") as handle:
                payload = pickle.load(handle)
        except (OSError, pickle.UnpicklingError):
            payload = None
        required = {
            "target_distance",
            "target_family_unseen",
            "target_family_sparsity",
        }
        if isinstance(payload, dict) and required.issubset(payload):
            return payload
    return compute_target_sequence_support_payload(
        dataset_root,
        dataset,
        split,
    )


@lru_cache(maxsize=32)
def _native_distance_payload(
    dataset_root: str,
    dataset: str,
    split: str,
) -> dict[str, dict[str, float]] | None:
    support_path = (
        Path(dataset_root)
        / dataset
        / split
        / "pime"
        / NATIVE_SUPPORT_NAME
    )
    if support_path.exists():
        try:
            with support_path.open("rb") as handle:
                payload = pickle.load(handle)
        except (OSError, pickle.UnpicklingError):
            payload = None
        required = {
            "drug_nearest",
            "drug_density",
            "target_nearest",
            "target_density",
        }
        if isinstance(payload, dict) and required.issubset(payload):
            return payload
    return compute_native_distance_payload(dataset_root, dataset, split)


def source_support_provenance(
    *,
    dataset: str,
    split: str = "cluster",
    dataset_root: str | Path = "datasets",
) -> dict[str, object]:
    """Describe the fixed, label-free references used by source-domain E."""

    resolved_root = str(Path(dataset_root).resolve())
    payload = _source_support_payload(resolved_root, str(dataset), str(split))
    if payload is None:
        return {
            "dataset": dataset,
            "split": split,
            "source_reference_status": "unavailable",
        }
    native = _native_distance_payload(resolved_root, str(dataset), str(split))
    target_sequence = _target_sequence_support_payload(
        resolved_root,
        str(dataset),
        str(split),
    )
    direct_target = _direct_target_support_payload(
        resolved_root,
        str(dataset),
        str(split),
    )
    joint_support = _joint_support_payload(
        resolved_root,
        str(dataset),
        str(split),
    )
    return {
        "dataset": dataset,
        "split": split,
        "source_reference_status": "available",
        "source_train_path": payload["source_train_path"],
        "source_train_rows": payload["source_train_rows"],
        "source_columns_loaded": "|".join(payload["source_columns"]),
        "outcome_columns_loaded": False,
        "drug_native_metric": "morgan_radius2_2048_tanimoto",
        "target_native_metric": "prot_cls_esm_cosine",
        "native_support_asset": str(
            Path(resolved_root)
            / dataset
            / split
            / "pime"
            / NATIVE_SUPPORT_NAME
        ),
        "native_support_status": (
            "available" if native is not None else "unavailable"
        ),
        "target_sequence_metric": (
            "mmseqs2_identity_x_min_coverage_to_source_family"
        ),
        "target_sequence_thresholds": (
            f"identity>={MMSEQS_MIN_IDENTITY:g}|"
            f"query_coverage>={MMSEQS_MIN_QUERY_COVERAGE:g}|"
            f"evalue<={MMSEQS_MAX_EVALUE:g}"
        ),
        "target_sequence_support_asset": str(
            Path(resolved_root)
            / dataset
            / split
            / "pime"
            / TARGET_SEQUENCE_SUPPORT_NAME
        ),
        "target_sequence_support_status": (
            "available" if target_sequence is not None else "unavailable"
        ),
        "direct_target_metric": (
            "mmseqs2_identity_x_min_coverage_to_source_targets_top5"
        ),
        "direct_target_support_asset": str(
            Path(resolved_root)
            / dataset
            / split
            / "pime"
            / DIRECT_TARGET_SUPPORT_NAME
        ),
        "direct_target_support_status": (
            "available" if direct_target is not None else "unavailable"
        ),
        "joint_support_metric": (
            "max_observed_source_pair_drug_tanimoto_x_target_sequence"
        ),
        "joint_support_exact_pair_excluded": True,
        "joint_support_asset": str(
            Path(resolved_root)
            / dataset
            / split
            / "pime"
            / JOINT_SUPPORT_NAME
        ),
        "joint_support_status": (
            "available" if joint_support is not None else "unavailable"
        ),
    }


def attach_source_support(
    table: pd.DataFrame,
    *,
    dataset: str,
    dataset_root: str | Path = "datasets",
) -> pd.DataFrame:
    """Attach label-free support counts from the frozen predictor's train split."""

    out = table.copy()
    if not {"dr_id", "pr_id"}.issubset(out.columns):
        return out
    split = (
        str(out["split"].dropna().iloc[0]).lower()
        if "split" in out and out["split"].notna().any()
        else "cluster"
    )
    payload = _source_support_payload(
        str(Path(dataset_root).resolve()),
        str(dataset),
        split,
    )
    if payload is None:
        return out

    drug_keys = _id_keys(out["dr_id"])
    target_keys = _id_keys(out["pr_id"])
    pair_keys = _pair_keys(out["dr_id"], out["pr_id"])
    drug_clusters = drug_keys.map(payload["drug_cluster_map"]).fillna("")
    target_clusters = target_keys.map(payload["target_cluster_map"]).fillna("")
    cluster_pairs = pd.Series(
        list(zip(drug_clusters, target_clusters)),
        index=out.index,
        dtype=object,
    )
    mappings = (
        ("source_drug_count", drug_keys, payload["drug_count"]),
        ("source_target_count", target_keys, payload["target_count"]),
        ("source_pair_count", pair_keys, payload["pair_count"]),
        (
            "source_drug_cluster_count",
            drug_clusters,
            payload["drug_cluster_count"],
        ),
        (
            "source_target_cluster_count",
            target_clusters,
            payload["target_cluster_count"],
        ),
        (
            "source_cluster_pair_count",
            cluster_pairs,
            payload["cluster_pair_count"],
        ),
    )
    for column, keys, counts in mappings:
        out[column] = keys.map(counts).fillna(0.0).astype(float)
    distances = _prior_distance_payload(
        str(Path(dataset_root).resolve()),
        str(dataset),
        split,
    )
    if distances is not None:
        out["drug_nearest_train_distance"] = drug_keys.map(
            distances["drug_nearest"]
        )
        out["target_nearest_train_distance"] = target_keys.map(
            distances["target_nearest"]
        )
        out["drug_train_knn_distance"] = drug_keys.map(
            distances["drug_density"]
        )
        out["target_train_knn_distance"] = target_keys.map(
            distances["target_density"]
        )
        out["domain_shift_score"] = pd.concat(
            [
                out["drug_nearest_train_distance"],
                out["target_nearest_train_distance"],
            ],
            axis=1,
        ).max(axis=1, skipna=False)
        out["domain_density_distance"] = pd.concat(
            [
                out["drug_train_knn_distance"],
                out["target_train_knn_distance"],
            ],
            axis=1,
        ).max(axis=1, skipna=False)
    native_distances = _native_distance_payload(
        str(Path(dataset_root).resolve()),
        str(dataset),
        split,
    )
    if native_distances is not None:
        out["drug_morgan_nearest_train_distance"] = drug_keys.map(
            native_distances["drug_nearest"]
        )
        out["drug_morgan_knn_distance"] = drug_keys.map(
            native_distances["drug_density"]
        )
        out["target_esm_nearest_train_distance"] = target_keys.map(
            native_distances["target_nearest"]
        )
        out["target_esm_knn_distance"] = target_keys.map(
            native_distances["target_density"]
        )
        nearest = out[
            [
                "drug_morgan_nearest_train_distance",
                "target_esm_nearest_train_distance",
            ]
        ]
        density = out[
            [
                "drug_morgan_knn_distance",
                "target_esm_knn_distance",
            ]
        ]
        out["native_domain_shift_score"] = nearest.max(
            axis=1,
            skipna=False,
        )
        out["native_domain_density_distance"] = density.max(
            axis=1,
            skipna=False,
        )
        out["native_support_imbalance"] = (
            nearest.iloc[:, 0] - nearest.iloc[:, 1]
        ).abs()
        out["native_density_imbalance"] = (
            density.iloc[:, 0] - density.iloc[:, 1]
        ).abs()
    target_sequence = _target_sequence_support_payload(
        str(Path(dataset_root).resolve()),
        str(dataset),
        split,
    )
    if target_sequence is not None:
        out["target_mmseqs_nearest_train_distance"] = target_keys.map(
            target_sequence["target_distance"]
        )
        out["target_mmseqs_family_unseen"] = target_keys.map(
            target_sequence["target_family_unseen"]
        )
        out["target_mmseqs_family_sparsity"] = target_keys.map(
            target_sequence["target_family_sparsity"]
        )
    direct_target = _direct_target_support_payload(
        str(Path(dataset_root).resolve()),
        str(dataset),
        split,
    )
    if direct_target is not None:
        out["target_direct_nearest_train_distance"] = target_keys.map(
            direct_target["target_nearest"]
        )
        out["target_direct_knn_distance"] = target_keys.map(
            direct_target["target_density"]
        )
    joint_support = _joint_support_payload(
        str(Path(dataset_root).resolve()),
        str(dataset),
        split,
    )
    if joint_support is not None:
        out["joint_nearest_train_distance"] = [
            joint_nearest_train_distance(drug_id, target_id, joint_support)
            for drug_id, target_id in zip(drug_keys, target_keys)
        ]
    return out


def prepare_table(
    path: Path,
    model: str,
    dataset: str,
    run_id: str,
    *,
    dataset_root: str | Path = "datasets",
) -> pd.DataFrame:
    out = pd.read_csv(path)
    out["model"] = model
    out["dataset"] = dataset
    out["run_id"] = run_id
    out["label"] = numeric(out["label"])
    if "p_base" not in out:
        out["p_base"] = _sigmoid(numeric(out["s_base"]))
    out["p_base"] = numeric(out["p_base"])
    if "base_pred" not in out:
        out["base_pred"] = (out["p_base"] >= 0.5).astype(float)
    out["base_pred"] = numeric(out["base_pred"])
    if "base_correct" not in out:
        out["base_correct"] = (out["base_pred"] == out["label"]).astype(float)
    out["base_correct"] = numeric(out["base_correct"])
    out["base_wrong"] = 1.0 - out["base_correct"]
    if "base_confidence" not in out:
        out["base_confidence"] = 2.0 * (out["p_base"] - 0.5).abs()
    out["base_confidence"] = numeric(out["base_confidence"]).clip(0.0, 1.0)
    return attach_source_support(
        out,
        dataset=dataset,
        dataset_root=dataset_root,
    )


def round_numeric(frame: pd.DataFrame, precision: int) -> pd.DataFrame:
    out = frame.copy()
    numeric_columns = out.select_dtypes(include=[np.number]).columns
    out[numeric_columns] = out[numeric_columns].round(int(precision))
    return out


def write_csv(frame: pd.DataFrame, path: Path, precision: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    round_numeric(frame, precision).to_csv(path, index=False)
    print(f"[write] {path}")
