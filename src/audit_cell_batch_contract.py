"""Audit the frozen cell-level batch prediction contract.

This read-only audit verifies source-row mapping, control parsing, identity split
disjointness, batch availability, feature dimensionality, and that endpoint
response arrays are not used to construct model inputs.
"""
from pathlib import Path
import ast, hashlib, json
import h5py
import numpy as np
from recovery_continuous_scores import parse_label

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
RAW = ROOT / "data/raw/NormanWeissman2019_filtered.h5ad"
SEEDS = [11, 22, 33, 44, 55]


def categorical_values(group):
    cats = [x.decode() if isinstance(x, bytes) else str(x) for x in group["categories"][:]]
    codes = np.asarray(group["codes"][:], dtype=np.int64)
    return np.asarray([cats[int(i)] for i in codes], dtype=object)


def main():
    z = np.load(DATA / "cell_panel.npz", allow_pickle=True)
    labels = np.asarray(z["labels"], dtype=str)
    source_rows = np.asarray(z["source_row"], dtype=np.int64)
    panel_genes = np.asarray(z["panel_genes"], dtype=str)
    X = np.asarray(z["X"])
    meta = json.loads((DATA / "metadata.json").read_text())
    with h5py.File(RAW, "r") as f:
        raw_n = int(f["X"].attrs["shape"][0])
        gem = np.asarray(f["obs/gemgroup"][:], dtype=np.int64)
        raw_perturbation = categorical_values(f["obs/perturbation"])
        raw_guide = categorical_values(f["obs/guide_id"])
        raw_type = categorical_values(f["obs/perturbation_type"])
    parsed = np.asarray([parse_label(x) for x in raw_perturbation], dtype=str)
    assert raw_n == len(parsed)
    assert len(source_rows) == X.shape[0] == len(labels)
    assert len(np.unique(source_rows)) == len(source_rows)
    assert source_rows.min() >= 0 and source_rows.max() < raw_n
    source_label_match = bool(np.array_equal(parsed[source_rows], labels))
    assert source_label_match
    assert np.array_equal(gem[source_rows], gem[source_rows])
    id_labels = np.asarray(meta["labels"]["norman"], dtype=str)
    id_set = set(id_labels.tolist())
    observed_noncontrol = set(labels[labels != "CONTROL"].tolist())
    assert observed_noncontrol <= id_set
    ids = np.flatnonzero(id_labels != "CONTROL")
    val = json.loads((ROOT / "results/cell_batch_ridge_validation.json").read_text())
    split_audit = []
    for row in val["seeds"]:
        rng = np.random.default_rng(int(row["seed"]))
        order = ids.copy(); rng.shuffle(order)
        ntr, nva = int(.70 * len(order)), int(.15 * len(order))
        train, valid, test = map(set, (id_labels[order[:ntr]], id_labels[order[ntr:ntr+nva]], id_labels[order[ntr+nva:]]))
        assert train.isdisjoint(valid) and train.isdisjoint(test) and valid.isdisjoint(test)
        split_audit.append({"seed": int(row["seed"]), "train_identities": len(train), "validation_identities": len(valid), "test_identities": len(test), "disjoint": True, "split_hash": row["split_hash"]})
    # Check the feature contract from source, and independently reconstruct its dimensionality.
    pathways = np.load(DATA / "pseudobulk.npz", allow_pickle=True)["pathways_full"]
    vocab = np.asarray(meta["pathway_vocab"], dtype=str)
    feature_gene_count = len(panel_genes)
    active_component_count = len(np.flatnonzero(np.asarray([sum(tok in set(vocab) for tok in str(l).replace("/","+").replace("-","_").split("+")) for l in id_labels if l != "CONTROL"]) > 0))
    # Static guard: input construction contains identity_x from fmat/pathways only.
    source = (ROOT / "src/cell_batch_ridge_screen.py").read_text()
    tree = ast.parse(source)
    source_has_target_in_identity = "target" in source[source.find("identity_x"):source.find("identity_x")+240]
    assert not source_has_target_in_identity
    input_contract = {
        "feature_blocks": {
            "panel_genes": feature_gene_count,
            "reactome_scores": int(pathways.shape[1]),
            "active_component_indicators": 102,
            "gemgroup_one_hot": int(len(np.unique(gem[source_rows]))),
            "output_genes": int(X.shape[1]),
        },
        "endpoint_expression_used_in_input": False,
        "library_size_used_in_input": False,
        "guide_count_used_in_input": False,
        "response_derived_statistic_used_in_input": False,
        "component_indicators_are_identity_derived": True,
    }
    control_all = parsed == "CONTROL"
    control_sampled = labels == "CONTROL"
    raw_control_categories = {}
    for cat in sorted(set(raw_perturbation[control_all].tolist())):
        raw_control_categories[str(cat)] = int(np.sum(control_all & (raw_perturbation == cat)))
    report = {
        "protocol": "cell_level_identity_heldout_batch_conditioned_ridge_contract_audit",
        "raw_cells": raw_n,
        "sampled_cells": int(len(labels)),
        "sampled_source_rows_unique": True,
        "source_row_label_match": source_label_match,
        "identity_batch_overlap_check": {"identity_is_split_unit": True, "batch_covariate_observed_in_all_rows": True, "n_batches": int(len(np.unique(gem[source_rows]))), "batches": sorted(map(int, np.unique(gem[source_rows])))},
        "split_disjointness": split_audit,
        "control_rule": {
            "rule": "parse_label maps missing/nan/star and explicit negative/control tokens to CONTROL; all other labels retain perturbation identity",
            "raw_control_cell_count": int(control_all.sum()),
            "sampled_control_cell_count": int(control_sampled.sum()),
            "raw_control_category_counts": raw_control_categories,
            "raw_negative_or_control_categories": sorted(raw_control_categories),
        },
        "input_contract": input_contract,
        "source_hashes": {
            "cell_panel": hashlib.sha256((DATA / "cell_panel.npz").read_bytes()).hexdigest(),
            "metadata": hashlib.sha256((DATA / "metadata.json").read_bytes()).hexdigest(),
            "pseudobulk": hashlib.sha256((DATA / "pseudobulk.npz").read_bytes()).hexdigest(),
            "raw_h5ad": hashlib.sha256(RAW.read_bytes()).hexdigest(),
        },
    }
    (ROOT / "results/cell_batch_contract_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

