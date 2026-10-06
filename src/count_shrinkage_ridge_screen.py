"""Validation-only count-aware shrinkage of noisy identity means.

The pseudobulk response is an empirical mean with identity-specific cell
counts.  This screen asks whether shrinking training identity means toward the
control-centered zero by n/(n+k), with k selected on validation, improves a
fixed identity-augmented Ridge predictor.  Counts are used only as a
predeclared training-side precision proxy; validation and locked test targets
are never transformed or scored here.
"""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
OUT = ROOT / "results"
SEEDS = [11, 22, 33, 44, 55]
KS = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
ALPHAS = [1.0, 3.0, 10.0, 30.0, 100.0]


def metric(y, p):
    a, b = np.asarray(y).ravel(), np.asarray(p).ravel()
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "pearson": float(pearsonr(a, b).statistic),
        "spearman": float(spearmanr(a, b).statistic),
        "n_identities": int(len(y)),
    }


def build_features(labels, metadata, pathways):
    vocab = np.asarray(metadata["pathway_vocab"])
    genes = np.asarray(metadata["genes"])
    gi = {g: i for i, g in enumerate(vocab)}
    f = np.zeros((len(labels), len(vocab)), dtype=np.float64)
    for i, label in enumerate(labels):
        if label == "CONTROL":
            continue
        for token in str(label).replace("/", "+").replace("-", "_").split("+"):
            if token in gi:
                f[i, gi[token]] = 1.0
    active = np.flatnonzero(f.sum(0) > 0)
    return np.c_[f[:, [gi[g] for g in genes]], f @ pathways, f[:, active]]


def main():
    z = np.load(DATA / "pseudobulk.npz", allow_pickle=True)
    metadata = json.loads((DATA / "metadata.json").read_text())
    labels = np.asarray(metadata["labels"]["norman"], dtype=str)
    y = z["y_norman"].astype(np.float64)
    controls = np.flatnonzero(labels == "CONTROL")
    y = y - y[controls].mean(0)
    x = build_features(labels, metadata, z["pathways_full"].astype(np.float64))
    counts_table = pd.read_csv(ROOT / "results/per_identity_counts.csv")
    counts_table = counts_table.query("dataset == 'norman'").set_index("identity")
    counts = np.asarray([float(counts_table.loc[l, "analyzed_cells"]) for l in labels])
    ids = np.flatnonzero(labels != "CONTROL")
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        order = ids.copy()
        rng.shuffle(order)
        ntr, nva = int(0.70 * len(order)), int(0.15 * len(order))
        tr_ids, va, te = order[:ntr], order[ntr:ntr + nva], order[ntr + nva:]
        tr = np.r_[tr_ids, controls]
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-3
        xs = (x - mu) / sd
        base_candidates = []
        shrink_candidates = []
        for alpha in ALPHAS:
            base = Ridge(alpha=alpha, solver="lsqr", tol=1e-7).fit(xs[tr], y[tr])
            base_candidates.append((alpha, base, metric(y[va], base.predict(xs[va]))))
            for k in KS:
                ytr = y[tr].copy()
                # Controls are already centered and are never shrunk.  The
                # factor is computed from training identity counts only.
                fac = counts[tr_ids] / (counts[tr_ids] + k) if k else np.ones(len(tr_ids))
                ytr[:len(tr_ids)] *= fac[:, None]
                model = Ridge(alpha=alpha, solver="lsqr", tol=1e-7).fit(xs[tr], ytr)
                shrink_candidates.append((alpha, k, model, metric(y[va], model.predict(xs[va]))))
        ba, _, bm = min(base_candidates, key=lambda q: q[2]["rmse"])
        strict = [q for q in shrink_candidates if q[3]["rmse"] < bm["rmse"] and q[3]["pearson"] > bm["pearson"] and q[3]["spearman"] > bm["spearman"]]
        best = min(shrink_candidates, key=lambda q: q[3]["rmse"])
        rows.append({
            "seed": seed,
            "split_hash": hashlib.sha256(np.asarray(np.r_[tr, va, te], dtype=np.int64).tobytes()).hexdigest(),
            "baseline_alpha": ba,
            "baseline_validation": bm,
            "best_shrinkage": {"alpha": best[0], "k": best[1], "validation": best[3]},
            "n_strict_all_metric_candidates": len(strict),
            "training_count_summary": {"min": float(counts[tr_ids].min()), "median": float(np.median(counts[tr_ids])), "max": float(counts[tr_ids].max())},
        })
        print(seed, "baseline", bm, "best", best[3], "params", best[0], best[1], "strict", len(strict), flush=True)
    out = {
        "protocol": "validation_only_count_shrinkage_identity_augmented_ridge",
        "candidate": "training identity means shrunk toward control-centered zero by n/(n+k), followed by identity-augmented Ridge",
        "baseline": "identity-augmented Ridge with identical features and splits",
        "selection": "Ridge alpha and shrinkage k selected using validation responses only; no locked test target loaded or scored",
        "seeds": rows,
        "all_seed_strict_winner": bool(all(r["n_strict_all_metric_candidates"] > 0 for r in rows)),
        "input_hashes": {
            "pseudobulk": hashlib.sha256((DATA / "pseudobulk.npz").read_bytes()).hexdigest(),
            "metadata": hashlib.sha256((DATA / "metadata.json").read_bytes()).hexdigest(),
            "counts": hashlib.sha256((ROOT / "results/per_identity_counts.csv").read_bytes()).hexdigest(),
        },
    }
    (OUT / "count_shrinkage_ridge_validation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"all_seed_strict_winner": out["all_seed_strict_winner"]}, indent=2))


if __name__ == "__main__":
    main()
