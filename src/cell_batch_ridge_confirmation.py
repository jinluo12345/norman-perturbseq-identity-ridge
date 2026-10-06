"""Locked test confirmation for the pre-registered gemgroup Ridge screen."""
from pathlib import Path
import hashlib, json
import h5py
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

from cell_batch_ridge_screen import ridge_predictions

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
RAW = ROOT / "data/raw/NormanWeissman2019_filtered.h5ad"
SEEDS = [11, 22, 33, 44, 55]


def metric(y, p):
    a, b = np.asarray(y).ravel(), np.asarray(p).ravel()
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "pearson": float(pearsonr(a, b).statistic),
        "spearman": float(spearmanr(a, b).statistic),
        "n_cells": int(len(y)),
    }


def main():
    val = json.loads((ROOT / "results/cell_batch_ridge_validation.json").read_text())
    assert val["all_seed_strict_winner"]
    z = np.load(DATA / "cell_panel.npz", allow_pickle=True)
    cells = z["X"].astype(np.float64)
    cell_labels = np.asarray(z["labels"], dtype=str)
    source_rows = z["source_row"].astype(int)
    with h5py.File(RAW, "r") as f:
        gem = np.asarray(f["obs"]["gemgroup"][:], dtype=int)[source_rows]
    meta = json.loads((DATA / "metadata.json").read_text())
    id_labels = np.asarray(meta["labels"]["norman"], dtype=str)
    pathways = np.load(DATA / "pseudobulk.npz", allow_pickle=True)["pathways_full"].astype(np.float64)
    vocab = np.asarray(meta["pathway_vocab"]); genes = np.asarray(meta["genes"]); gi = {g: i for i, g in enumerate(vocab)}
    fmat = np.zeros((len(id_labels), len(vocab)), dtype=np.float64)
    for i, label in enumerate(id_labels):
        if label != "CONTROL":
            for token in str(label).replace("/", "+").replace("-", "_").split("+"):
                if token in gi: fmat[i, gi[token]] = 1.0
    active = np.flatnonzero(fmat.sum(0) > 0)
    identity_x = np.c_[fmat[:, [gi[g] for g in genes]], fmat @ pathways, fmat[:, active]]
    id_to_i = {label: i for i, label in enumerate(id_labels)}
    cell_id = np.asarray([id_to_i[label] for label in cell_labels], dtype=int)
    target = cells - cells[cell_labels == "CONTROL"].mean(0)
    batches = np.unique(gem); batch_map = {int(b): j for j, b in enumerate(batches)}
    batch_onehot = np.zeros((len(gem), len(batches)), dtype=np.float64)
    for i, b in enumerate(gem): batch_onehot[i, batch_map[int(b)]] = 1.0
    ids = np.flatnonzero(id_labels != "CONTROL")
    rows = []
    for expected in val["seeds"]:
        seed = int(expected["seed"])
        rng = np.random.default_rng(seed); order = ids.copy(); rng.shuffle(order)
        split_hash = hashlib.sha256(np.asarray(order, dtype=np.int64).tobytes()).hexdigest()
        assert split_hash == expected["split_hash"], (seed, split_hash, expected["split_hash"])
        ntr, nva = int(0.70 * len(order)), int(0.15 * len(order))
        train_labels = set(id_labels[order[:ntr]].tolist()) | {"CONTROL"}
        val_labels = set(id_labels[order[ntr:ntr + nva]].tolist())
        test_labels = set(id_labels[order[ntr + nva:]].tolist())
        test_cells = np.isin(cell_labels, list(test_labels))
        train_ids = np.flatnonzero(np.isin(id_labels, list(train_labels)))
        base_rows = [target[cell_id == ii].mean(0) for ii in train_ids]
        base_y = np.asarray(base_rows)
        xm = identity_x[train_ids].mean(0); xs = identity_x[train_ids].std(0) + 1e-3
        xtrain = (identity_x[train_ids] - xm) / xs
        base_q = (identity_x - xm) / xs
        base_alpha = float(expected["baseline_alpha"])
        base_pred = ridge_predictions(xtrain, base_y, base_q, [base_alpha])[base_alpha][cell_id]
        gr_x, gr_y = [], []
        for ii in train_ids:
            for b in batches:
                mask = (cell_id == ii) & (gem == b)
                if mask.sum() >= 2:
                    gr_x.append(np.r_[identity_x[ii], batch_onehot[np.flatnonzero(mask)[0]]])
                    gr_y.append(target[mask].mean(0))
        gr_x, gr_y = np.asarray(gr_x), np.asarray(gr_y)
        gmu, gsd = gr_x.mean(0), gr_x.std(0) + 1e-3
        gr_xs = (gr_x - gmu) / gsd
        query_xs = (np.c_[identity_x[cell_id], batch_onehot] - gmu) / gsd
        cand_alpha = float(expected["batch_candidate_alpha"])
        cand_pred = ridge_predictions(gr_xs, gr_y, query_xs, [cand_alpha])[cand_alpha]
        bm, cm = metric(target[test_cells], base_pred[test_cells]), metric(target[test_cells], cand_pred[test_cells])
        delta = {"rmse": cm["rmse"] - bm["rmse"], "pearson": cm["pearson"] - bm["pearson"], "spearman": cm["spearman"] - bm["spearman"]}
        rows.append({"seed": seed, "split_hash": split_hash, "baseline_alpha": base_alpha, "candidate_alpha": cand_alpha,
                     "baseline_test": bm, "candidate_test": cm, "test_delta_candidate_minus_baseline": delta,
                     "test_identities": sorted(test_labels), "n_test_identities": len(test_labels)})
        print(seed, bm, cm, delta, flush=True)
    out = {
        "protocol": "locked_confirmation_cell_identity_by_gemgroup_ridge",
        "selection": "alphas and identity splits frozen from validation-only screen; no test metric used for choices",
        "baseline": val["baseline"], "candidate": val["candidate"], "seeds": rows,
        "all_test_strict_winner": bool(all(r["test_delta_candidate_minus_baseline"]["rmse"] < 0 and r["test_delta_candidate_minus_baseline"]["pearson"] > 0 and r["test_delta_candidate_minus_baseline"]["spearman"] > 0 for r in rows)),
        "input_hashes": {"cell_panel": hashlib.sha256((DATA / "cell_panel.npz").read_bytes()).hexdigest(), "metadata": hashlib.sha256((DATA / "metadata.json").read_bytes()).hexdigest(), "pseudobulk": hashlib.sha256((DATA / "pseudobulk.npz").read_bytes()).hexdigest(), "raw_h5ad": hashlib.sha256(RAW.read_bytes()).hexdigest()},
    }
    (ROOT / "results/cell_batch_ridge_confirmation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"all_test_strict_winner": out["all_test_strict_winner"]}, indent=2))


if __name__ == "__main__": main()
