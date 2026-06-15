from dataclasses import dataclass


@dataclass
class StandardizedMol:
    mol: object
    canonical_smiles: str
    status: str
    num_components: int
    selected_component_smiles: str
    standardization_note: str


def load_chem():
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError(
            "RDKit is required for PIME drug evidence. Activate the project "
            "environment from environment.yml or install rdkit before running this builder."
        ) from exc
    return Chem


def _component_score(mol):
    heavy_atoms = mol.GetNumHeavyAtoms()
    carbon_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
    non_metal_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in {3, 4, 11, 12, 13, 19, 20, 30})
    return carbon_atoms > 0, non_metal_atoms, heavy_atoms


def standardize_smiles(smiles):
    """Parse SMILES and select the main covalent component for drug evidence.

    Salts/counterions should not become mechanism-channel fragments.  We select
    the largest organic-like component and record the decision in metadata.
    """

    Chem = load_chem()
    text = "" if smiles is None else str(smiles).strip()
    if not text:
        return StandardizedMol(None, "", "missing", 0, "", "empty_smiles")

    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return StandardizedMol(None, "", "invalid", 0, "", "rdkit_parse_failed")

    try:
        components = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True))
    except Exception:
        components = [mol]

    if not components:
        return StandardizedMol(None, "", "invalid", 0, "", "no_components_after_parse")

    selected = max(components, key=_component_score)
    selected_smiles = Chem.MolToSmiles(selected, canonical=True)
    note = "single_component"
    if len(components) > 1:
        note = f"multi_component_selected_main_component;num_components={len(components)}"
    return StandardizedMol(
        mol=selected,
        canonical_smiles=selected_smiles,
        status="ok",
        num_components=len(components),
        selected_component_smiles=selected_smiles,
        standardization_note=note,
    )
