from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reproducibility"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def write_environment() -> None:
    names = {
        "anndata", "h5py", "matplotlib", "numpy", "pandas", "scanpy",
        "scikit-learn", "scipy", "seaborn", "torch", "tqdm"
    }
    rows = []
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name")
        if name and name.lower() in names:
            rows.append((name.lower(), dist.version))
    rows.sort()
    py = __import__("sys").version.replace("\n", " ")
    out = ["# Observed workspace runtime", f"python=={py}"]
    out.extend(f"{name}=={version}" for name, version in rows)
    (OUT / "environment_lock.txt").write_text("\n".join(out) + "\n")

def write_panel() -> None:
    meta = json.loads((ROOT / "data/processed/metadata.json").read_text())
    panel = {
        "panel_name": "Norman control-relative response panel",
        "output_gene_count": len(meta["genes"]),
        "output_genes": meta["genes"],
        "pathway_count": meta["n_pathways"],
        "pathway_release": meta["pathway_release"],
        "pathway_names": meta["pathway_names"],
        "pathway_vocab_count": len(meta["pathway_vocab"]),
        "processed_cell_counts": meta["n_cells"],
        "panel_selection": meta["panel_selection"],
    }
    (OUT / "panel_manifest.json").write_text(json.dumps(panel, indent=2) + "\n")

def write_hashes() -> None:
    rels = []
    for root in ("src", "paper/figure_prompts"):
        rels.extend(p for p in (ROOT / root).rglob("*") if p.is_file())
    rels += [
        ROOT / "paper/main.tex", ROOT / "paper/references.bib",
        ROOT / "data/processed/metadata.json",
        ROOT / "data/processed/input_checksums.json",
        ROOT / "results/intervention_priority_simulation.json",
        ROOT / "results/neat_diagnostics.json",
        ROOT / "results/repaired_split_local_controls.json",
        ROOT / "results/repaired_2x2_factorial.json",
        ROOT / "results/repaired_ablation_frozen_control.json",
        ROOT / "results/repaired_stratified_analysis.json",
        ROOT / "results/spearman_exact_rerank_audit.json",
        ROOT / "paper/figures_batch/Fig2_cell_batch_repaired_performance.pdf",
        ROOT / "paper/figures_batch/Fig3_cell_batch_strata.pdf",
        ROOT / "paper/figures_batch/Fig4_cell_batch_ablation_repaired.pdf",
        ROOT / "paper/figures_batch/Fig5_identity_cluster_uncertainty_repaired.pdf",
        ROOT / "paper/figures_batch/Fig6_intervention_priority.pdf",
        ROOT / "figures/ai_generated/Fig1_model_mechanism_round4.png",
    ]
    unique = sorted({p for p in rels if p.exists()})
    lines = []
    for p in unique:
        lines.append(f"{sha256(p)}  {p.relative_to(ROOT)}")
    (OUT / "source_manifest.sha256").write_text("\n".join(lines) + "\n")

if __name__ == "__main__":
    write_environment()
    write_panel()
    write_hashes()
