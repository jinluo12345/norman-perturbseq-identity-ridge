"""Repeated identity-held-out validation for a locked PLS latent decoder.

Only the validation partition is scored here.  Test identities remain locked
until a separate confirmation script is frozen.
"""
from pathlib import Path
import hashlib, json
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
OUT = ROOT / "results"
SPLITS = [11, 22, 33, 44, 55]
N_COMPONENTS = 40


def metric(y, p):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "pearson": float(pearsonr(y.ravel(), p.ravel()).statistic),
        "spearman": float(spearmanr(y.ravel(), p.ravel()).statistic),
        "n_identities": int(len(y)),
    }


def build():
    z = np.load(DATA / "pseudobulk.npz", allow_pickle=True)
    m = json.loads((DATA / "metadata.json").read_text())
    labs = np.asarray(m["labels"]["norman"], str)
    y = z["y_norman"].astype(np.float32)
    ctrl = np.flatnonzero(labs == "CONTROL")
    y = y - y[ctrl].mean(0)
    P = z["pathways_full"].astype(np.float32)
    vocab = np.asarray(m["pathway_vocab"])
    genes = np.asarray(m["genes"])
    gi = {x: i for i, x in enumerate(vocab)}
    F = np.zeros((len(labs), len(vocab)), np.float32)
    for i, label in enumerate(labs):
        if label != "CONTROL":
            for token in str(label).replace("/", "+").replace("-", "_").split("+"):
                if token in gi:
                    F[i, gi[token]] = 1
    X = np.c_[F[:, [gi[x] for x in genes]], F @ P]
    return labs, y, X, ctrl


def run():
    labs, y, X, ctrl = build()
    ids = np.flatnonzero(labs != "CONTROL")
    rows = []
    for split_seed in SPLITS:
        rng = np.random.default_rng(split_seed)
        order = ids.copy(); rng.shuffle(order)
        ntr, nva = int(.7 * len(order)), int(.15 * len(order))
        tr = np.r_[order[:ntr], ctrl]
        va = order[ntr:ntr + nva]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-3
        Xs = (X - mu) / sd
        ridge = Ridge(alpha=10, solver="cholesky").fit(Xs[tr], y[tr])
        pls = PLSRegression(n_components=N_COMPONENTS, scale=False, max_iter=1000).fit(Xs[tr], y[tr])
        rows.append({
            "split_seed": split_seed,
            "split_hash": hashlib.sha256(np.asarray(np.r_[tr, va], dtype=np.int64).tobytes()).hexdigest(),
            "ridge_validation": metric(y[va], ridge.predict(Xs[va])),
            "pls_validation": metric(y[va], pls.predict(Xs[va])),
            "n_train_identities": int(len(tr) - len(ctrl)),
            "n_validation_identities": int(len(va)),
        })
    result = {
        "protocol": "repeated_identity_validation",
        "model": "PLSRegression",
        "n_components": N_COMPONENTS,
        "baseline": "joint standardized gene+pathway Ridge alpha=10",
        "splits": rows,
        "all_metrics_improved_each_split": all(
            r["pls_validation"]["rmse"] < r["ridge_validation"]["rmse"]
            and r["pls_validation"]["pearson"] > r["ridge_validation"]["pearson"]
            and r["pls_validation"]["spearman"] > r["ridge_validation"]["spearman"]
            for r in rows
        ),
    }
    (OUT / "pls_repeated_validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
