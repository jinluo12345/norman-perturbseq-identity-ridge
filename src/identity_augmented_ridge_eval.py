"""Audit a stronger identity-conditioned linear baseline.

The baseline uses the full set of perturbation-component indicators present in
Norman labels, alongside the fixed 512-gene panel and pathway scores.  Feature
statistics are fit on training identities plus controls only.
"""
from pathlib import Path
import hashlib, json
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
OUT = ROOT / "results"


def metric(y, p):
    return {"rmse": float(np.sqrt(mean_squared_error(y, p))),
            "pearson": float(pearsonr(y.ravel(), p.ravel()).statistic),
            "spearman": float(spearmanr(y.ravel(), p.ravel()).statistic),
            "n_identities": int(len(y))}


def run():
    z = np.load(DATA / "pseudobulk.npz", allow_pickle=True)
    m = json.loads((DATA / "metadata.json").read_text())
    labels = np.asarray(m["labels"]["norman"], str)
    y = z["y_norman"].astype(np.float32)
    ctrl = np.flatnonzero(labels == "CONTROL")
    y = y - y[ctrl].mean(0)
    P = z["pathways_full"].astype(np.float32)
    vocab = np.asarray(m["pathway_vocab"])
    genes = np.asarray(m["genes"])
    gi = {x: i for i, x in enumerate(vocab)}
    F = np.zeros((len(labels), len(vocab)), np.float32)
    for i, label in enumerate(labels):
        if label != "CONTROL":
            for token in str(label).replace("/", "+").replace("-", "_").split("+"):
                if token in gi:
                    F[i, gi[token]] = 1
    active = np.flatnonzero(F.sum(0) > 0)
    F_gene_panel = F[:, [gi[x] for x in genes]]
    X = np.c_[F_gene_panel, F @ P, F[:, active]]
    ids = np.flatnonzero(labels != "CONTROL")
    rng = np.random.default_rng(11); rng.shuffle(ids)
    ntr, nva = int(.7 * len(ids)), int(.15 * len(ids))
    tr = np.r_[ids[:ntr], ctrl]; va = ids[ntr:ntr+nva]; te = ids[ntr+nva:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-3
    Xs = (X - mu) / sd
    rows = []
    for alpha in [0.1, 1, 3, 10, 30, 100, 300]:
        model = Ridge(alpha=alpha, solver="cholesky").fit(Xs[tr], y[tr])
        rows.append({"alpha": alpha, "validation": metric(y[va], model.predict(Xs[va]))})
    best = min(rows, key=lambda r: r["validation"]["rmse"])["alpha"]
    model = Ridge(alpha=best, solver="cholesky").fit(Xs[tr], y[tr])
    out = {
        "protocol": "identity_augmented_ridge_locked_split11",
        "feature_definition": "512-gene panel membership + Reactome pathway scores + all active perturbation-component indicators",
        "n_active_identity_features": int(len(active)), "n_features": int(X.shape[1]),
        "split_seed": 11,
        "split_hash": hashlib.sha256(np.asarray(np.r_[tr, va, te], dtype=np.int64).tobytes()).hexdigest(),
        "alpha_grid": rows, "selected_alpha_by_validation_rmse": best,
        "validation": metric(y[va], model.predict(Xs[va])),
        "test": metric(y[te], model.predict(Xs[te])),
        "input_hashes": {
            "pseudobulk": hashlib.sha256((DATA / "pseudobulk.npz").read_bytes()).hexdigest(),
            "metadata": hashlib.sha256((DATA / "metadata.json").read_bytes()).hexdigest(),
        },
    }
    (OUT / "identity_augmented_ridge_eval.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    run()
