"""Identity-cluster paired bootstrap for the frozen batch Ridge confirmation."""
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
B = 500


def metric(y, p):
    a, b = np.asarray(y).ravel(), np.asarray(p).ravel()
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "pearson": float(pearsonr(a, b).statistic),
        "spearman": float(spearmanr(a, b).statistic),
    }


def main():
    val = json.loads((ROOT / "results/cell_batch_ridge_validation.json").read_text())
    conf = json.loads((ROOT / "results/cell_batch_ridge_confirmation.json").read_text())
    z = np.load(DATA / "cell_panel.npz", allow_pickle=True)
    cells = z["X"].astype(np.float64); cell_labels = np.asarray(z["labels"], dtype=str); source_rows = z["source_row"].astype(int)
    with h5py.File(RAW, "r") as f: gem = np.asarray(f["obs"]["gemgroup"][:], dtype=int)[source_rows]
    meta = json.loads((DATA / "metadata.json").read_text()); id_labels = np.asarray(meta["labels"]["norman"], dtype=str)
    pp = np.load(DATA / "pseudobulk.npz", allow_pickle=True); pathways = pp["pathways_full"].astype(np.float64)
    vocab = np.asarray(meta["pathway_vocab"]); genes = np.asarray(meta["genes"]); gi = {g: i for i, g in enumerate(vocab)}
    fmat = np.zeros((len(id_labels), len(vocab)), dtype=np.float64)
    for i, label in enumerate(id_labels):
        if label != "CONTROL":
            for token in str(label).replace("/", "+").replace("-", "_").split("+"):
                if token in gi: fmat[i, gi[token]] = 1.0
    active = np.flatnonzero(fmat.sum(0) > 0)
    identity_x = np.c_[fmat[:, [gi[g] for g in genes]], fmat @ pathways, fmat[:, active]]
    id_to_i = {label: i for i, label in enumerate(id_labels)}; cell_id = np.asarray([id_to_i[label] for label in cell_labels], dtype=int)
    target = cells - cells[cell_labels == "CONTROL"].mean(0)
    batches = np.unique(gem); batch_map = {int(b): j for j, b in enumerate(batches)}
    batch_onehot = np.zeros((len(gem), len(batches)), dtype=np.float64)
    for i, b in enumerate(gem): batch_onehot[i, batch_map[int(b)]] = 1.0
    ids = np.flatnonzero(id_labels != "CONTROL")
    rows = []
    for expected, locked in zip(val["seeds"], conf["seeds"]):
        seed = int(expected["seed"]); rng = np.random.default_rng(seed); order = ids.copy(); rng.shuffle(order)
        assert hashlib.sha256(np.asarray(order, dtype=np.int64).tobytes()).hexdigest() == expected["split_hash"] == locked["split_hash"]
        ntr, nva = int(0.70 * len(order)), int(0.15 * len(order))
        train_labels = set(id_labels[order[:ntr]].tolist()) | {"CONTROL"}; test_labels = sorted(set(id_labels[order[ntr + nva:]].tolist()))
        train_ids = np.flatnonzero(np.isin(id_labels, list(train_labels)))
        test_mask = np.isin(cell_labels, test_labels)
        base_y = np.asarray([target[cell_id == ii].mean(0) for ii in train_ids])
        xm, xs = identity_x[train_ids].mean(0), identity_x[train_ids].std(0) + 1e-3
        base_pred_id = ridge_predictions((identity_x[train_ids] - xm) / xs, base_y, (identity_x - xm) / xs, [float(expected["baseline_alpha"])])[float(expected["baseline_alpha"])]
        base_pred = base_pred_id[cell_id]
        gr_x, gr_y = [], []
        for ii in train_ids:
            for b in batches:
                mask = (cell_id == ii) & (gem == b)
                if mask.sum() >= 2:
                    gr_x.append(np.r_[identity_x[ii], batch_onehot[np.flatnonzero(mask)[0]]]); gr_y.append(target[mask].mean(0))
        gr_x, gr_y = np.asarray(gr_x), np.asarray(gr_y); gmu, gsd = gr_x.mean(0), gr_x.std(0) + 1e-3
        qx = (np.c_[identity_x[cell_id], batch_onehot] - gmu) / gsd
        ca = float(expected["batch_candidate_alpha"]); cand_pred = ridge_predictions((gr_x - gmu) / gsd, gr_y, qx, [ca])[ca]
        # Use complete test identities as the resampling unit.  This avoids
        # treating thousands of cells from one perturbation as independent.
        y_groups = [target[(cell_labels == l)] for l in test_labels]
        a_groups = [base_pred[(cell_labels == l)] for l in test_labels]
        c_groups = [cand_pred[(cell_labels == l)] for l in test_labels]
        point_b, point_c = metric(np.vstack(y_groups), np.vstack(a_groups)), metric(np.vstack(y_groups), np.vstack(c_groups))
        # RMSE and Pearson are computed from cluster sufficient statistics;
        # this avoids repeatedly sorting thousands of cells for Spearman.
        def stats(yy, pp):
            yv, pv = yy.ravel(), pp.ravel()
            return np.array([len(yv), np.sum((yv - pv) ** 2), np.sum(yv), np.sum(pv), np.sum(yv * yv), np.sum(pv * pv), np.sum(yv * pv)], float)
        ys, ass, css = np.asarray([stats(y_groups[k], a_groups[k]) for k in range(len(test_labels))]), np.asarray([stats(y_groups[k], a_groups[k]) for k in range(len(test_labels))]), np.asarray([stats(y_groups[k], c_groups[k]) for k in range(len(test_labels))])
        deltas = np.empty((B, 3), dtype=float)
        for j in range(B):
            ix = rng.integers(0, len(test_labels), size=len(test_labels))
            sb, sc = ass[ix].sum(0), css[ix].sum(0)
            def fast(s):
                n, sse, sy, sp, sy2, sp2, syp = s
                den = np.sqrt(max((sy2 - sy * sy / n) * (sp2 - sp * sp / n), 1e-30))
                return np.sqrt(sse / n), (syp - sy * sp / n) / den
            mb, mc = fast(sb), fast(sc)
            deltas[j] = [mc[0] - mb[0], mc[1] - mb[1], np.nan]
        rows.append({
            "seed": seed, "n_test_identities": len(test_labels), "n_test_cells": int(test_mask.sum()),
            "point_baseline": point_b, "point_candidate": point_c,
            "paired_delta_candidate_minus_baseline": {"rmse": float(point_c["rmse"] - point_b["rmse"]), "pearson": float(point_c["pearson"] - point_b["pearson"]), "spearman": float(point_c["spearman"] - point_b["spearman"])},
            "identity_cluster_bootstrap": {
                "B": B, "unit": "held-out perturbation identity cluster", "seed": seed + 100000,
                "delta_mean_rmse_pearson": np.nanmean(deltas[:, :2], axis=0).tolist(), "delta_ci95_rmse_pearson": np.nanquantile(deltas[:, :2], [0.025, 0.975], axis=0).tolist(),
                "fraction_delta_rmse_below_zero": float(np.mean(deltas[:, 0] < 0)),
                "fraction_delta_pearson_above_zero": float(np.mean(deltas[:, 1] > 0)),
                "spearman_bootstrap": "not computed; point estimate is reported and cluster bootstrap covers RMSE/Pearson",
            },
        })
        print(seed, rows[-1]["paired_delta_candidate_minus_baseline"], rows[-1]["identity_cluster_bootstrap"]["delta_ci95_rmse_pearson"], flush=True)
    out = {"protocol": "identity_cluster_paired_bootstrap_frozen_cell_batch_confirmation", "B": B, "seeds": rows, "input_hashes": conf["input_hashes"]}
    (ROOT / "results/cell_batch_ridge_bootstrap.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__": main()
