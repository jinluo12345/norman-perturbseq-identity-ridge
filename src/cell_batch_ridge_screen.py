"""Validation-only cell-level Ridge with observed gemgroup metadata.

The cell panel is fixed before fitting.  gemgroup is read only from the raw
observation metadata and is a technical batch covariate; no endpoint
expression statistic enters the input.  Models are fit to identity-by-batch
means to prevent cell-count imbalance from dominating the estimator.
"""
from pathlib import Path
import hashlib, json
import h5py
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
RAW = ROOT / "data/raw/NormanWeissman2019_filtered.h5ad"
OUT = ROOT / "results"
SEEDS = [11, 22, 33, 44, 55]
ALPHAS = [0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]


def metric(y, p):
    a, b = np.asarray(y).ravel(), np.asarray(p).ravel()
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "pearson": float(pearsonr(a, b).statistic),
        "spearman": float(spearmanr(a, b).statistic),
        "n_cells": int(len(y)),
    }


def ridge_predictions(xtr, ytr, xq, alphas):
    """Multi-output Ridge grid via one SVD per training design."""
    ymean = ytr.mean(0)
    xmean = xtr.mean(0)
    xc = xtr - xmean
    uq, s, vt = np.linalg.svd(xc, full_matrices=False)
    projected = uq.T @ (ytr - ymean)
    out = {}
    for alpha in alphas:
        coeff = vt.T @ ((s / (s * s + alpha))[:, None] * projected)
        out[alpha] = (xq - xmean) @ coeff + ymean
    return out


def main():
    z = np.load(DATA / "cell_panel.npz", allow_pickle=True)
    cells = z["X"].astype(np.float64)
    cell_labels = np.asarray(z["labels"], dtype=str)
    source_rows = z["source_row"].astype(int)
    with h5py.File(RAW, "r") as f:
        gem = np.asarray(f["obs"]["gemgroup"][:], dtype=int)[source_rows]
    meta = json.loads((DATA / "metadata.json").read_text())
    id_labels = np.asarray(meta["labels"]["norman"], dtype=str)
    pathways = np.load(DATA / "pseudobulk.npz", allow_pickle=True)["pathways_full"].astype(np.float64)
    vocab = np.asarray(meta["pathway_vocab"])
    genes = np.asarray(meta["genes"])
    gi = {g: i for i, g in enumerate(vocab)}
    fmat = np.zeros((len(id_labels), len(vocab)), dtype=np.float64)
    for i, label in enumerate(id_labels):
        if label != "CONTROL":
            for token in str(label).replace("/", "+").replace("-", "_").split("+"):
                if token in gi:
                    fmat[i, gi[token]] = 1.0
    active = np.flatnonzero(fmat.sum(0) > 0)
    identity_x = np.c_[fmat[:, [gi[g] for g in genes]], fmat @ pathways, fmat[:, active]]
    id_to_i = {label: i for i, label in enumerate(id_labels)}
    cell_id = np.asarray([id_to_i[label] for label in cell_labels], dtype=int)
    control_mask = cell_labels == "CONTROL"
    control_mean = cells[control_mask].mean(0)
    target = cells - control_mean
    batches = np.unique(gem)
    batch_map = {int(b): j for j, b in enumerate(batches)}
    batch_onehot = np.zeros((len(gem), len(batches)), dtype=np.float64)
    for i, b in enumerate(gem):
        batch_onehot[i, batch_map[int(b)]] = 1.0
    ids = np.flatnonzero(id_labels != "CONTROL")
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        order = ids.copy(); rng.shuffle(order)
        ntr, nva = int(0.70 * len(order)), int(0.15 * len(order))
        train_labels = set(id_labels[order[:ntr]].tolist()) | {"CONTROL"}
        val_labels = set(id_labels[order[ntr:ntr + nva]].tolist())
        test_labels = set(id_labels[order[ntr + nva:]].tolist())
        train_cells = np.isin(cell_labels, list(train_labels))
        val_cells = np.isin(cell_labels, list(val_labels))
        test_cells = np.isin(cell_labels, list(test_labels))
        train_ids = np.flatnonzero(np.isin(id_labels, list(train_labels)))
        # Baseline: train on identity means with no batch covariate.
        base_rows = []
        for ii in train_ids:
            mask = cell_id == ii
            base_rows.append(target[mask].mean(0))
        base_y = np.asarray(base_rows)
        xm = identity_x[train_ids].mean(0); xs = identity_x[train_ids].std(0) + 1e-3
        xtrain = (identity_x[train_ids] - xm) / xs
        base_candidates = []
        base_pred_grid = ridge_predictions(xtrain, base_y, (identity_x - xm) / xs, ALPHAS)
        for alpha in ALPHAS:
            pred_id = base_pred_grid[alpha]
            pred = pred_id[cell_id]
            base_candidates.append((alpha, None, metric(target[val_cells], pred[val_cells]), pred))
        ba, _, bm, base_pred = min(base_candidates, key=lambda q: q[2]["rmse"])
        # Candidate: identity-by-gemgroup means, with the same identity vector
        # and a one-hot gemgroup technical covariate.
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
        query_x = np.c_[identity_x[cell_id], batch_onehot]
        query_xs = (query_x - gmu) / gsd
        candidates = []
        cand_pred_grid = ridge_predictions(gr_xs, gr_y, query_xs, ALPHAS)
        for alpha in ALPHAS:
            pred = cand_pred_grid[alpha]
            candidates.append((alpha, None, metric(target[val_cells], pred[val_cells]), pred))
        ca, _, cm, cand_pred = min(candidates, key=lambda q: q[2]["rmse"])
        strict = cm["rmse"] < bm["rmse"] and cm["pearson"] > bm["pearson"] and cm["spearman"] > bm["spearman"]
        rows.append({
            "seed": seed,
            "split_hash": hashlib.sha256(np.asarray(order, dtype=np.int64).tobytes()).hexdigest(),
            "n_train_group_rows": int(len(gr_x)),
            "baseline_alpha": ba,
            "baseline_validation": bm,
            "batch_candidate_alpha": ca,
            "batch_candidate_validation": cm,
            "strict_all_metric_validation_win": bool(strict),
            "validation_identities": sorted(val_labels),
            "test_identities": sorted(test_labels),
        })
        print(seed, "base", bm, "batch", cm, "strict", strict, flush=True)
    out = {
        "protocol": "validation_only_cell_identity_by_gemgroup_ridge",
        "candidate": "identity-by-gemgroup mean Ridge with observed assay-batch gemgroup one-hot metadata",
        "baseline": "identity-mean Ridge with identical perturbation/pathway/component features and no batch covariate",
        "selection": "alpha selected by validation cell-level RMSE; no locked test targets scored",
        "n_cells": int(len(cells)), "n_batches": int(len(batches)), "batches": batches.tolist(),
        "seeds": rows,
        "all_seed_strict_winner": bool(all(r["strict_all_metric_validation_win"] for r in rows)),
        "input_hashes": {
            "cell_panel": hashlib.sha256((DATA / "cell_panel.npz").read_bytes()).hexdigest(),
            "metadata": hashlib.sha256((DATA / "metadata.json").read_bytes()).hexdigest(),
            "raw_h5ad": hashlib.sha256(RAW.read_bytes()).hexdigest(),
        },
    }
    (OUT / "cell_batch_ridge_validation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"all_seed_strict_winner": out["all_seed_strict_winner"]}, indent=2))


if __name__ == "__main__":
    main()
