"""Validation-only graph-regularized multi-task decoder.

The candidate keeps the identity-augmented input contract fixed (panel-gene
membership, Reactome scores and active perturbation components), but couples
the 512 output genes through a fixed Reactome gene-overlap Laplacian.  For a
coefficient matrix B this solves

  ||Y-XB||_F^2 + alpha ||B||_F^2 + gamma tr(B L B^T),

where L is built only from public pathway membership (no held-out responses).
The output graph is diagonalized once; each eigenmode is fit with its own
ridge strength alpha + gamma*lambda.  Alpha and gamma are selected on each
validation split by RMSE, and test metrics are read only after locking.
"""
from pathlib import Path
import hashlib, itertools, json
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
OUT = ROOT / "results"
SEEDS = [11, 22, 33, 44, 55]
ALPHAS = [1.0, 3.0, 10.0, 30.0, 100.0]
GAMMAS = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]


def tok(label):
    return [x for x in str(label).replace("/", "+").replace("-", "_").split("+")
            if x and x not in {"ONLY", "MOD"}]


def metric(y, p):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "pearson": float(pearsonr(y.ravel(), p.ravel()).statistic),
        "spearman": float(spearmanr(y.ravel(), p.ravel()).statistic),
        "n_identities": int(len(y)),
    }


def make_features(labels, vocab, genes, P):
    gi = {x: i for i, x in enumerate(vocab)}
    F = np.zeros((len(labels), len(vocab)), np.float64)
    for i, label in enumerate(labels):
        if label == "CONTROL":
            continue
        for t in tok(label):
            if t in gi:
                F[i, gi[t]] = 1.0
    active = np.flatnonzero(F.sum(0) > 0)
    A = F[:, [gi[x] for x in genes]]
    # Restrict the pathway multiplication to components observed in Norman;
    # multiplying the 33,694-row vocabulary by all 256 pathways is unnecessary.
    B = F[:, active] @ P[active]
    return np.c_[A, B, F[:, active]], active


def graph_laplacian(genes, pathway_gene_lists):
    """Normalized overlap Laplacian for the output gene panel."""
    gix = {g: i for i, g in enumerate(genes)}
    A = np.zeros((len(genes), len(genes)), np.float64)
    for members in pathway_gene_lists:
        inds = [gix[g] for g in members if g in gix]
        if len(inds) < 2:
            continue
        # One vote per shared pathway, normalized by set size to avoid hubs.
        w = 1.0 / np.sqrt(float(len(inds)))
        ii = np.asarray(inds, dtype=int)
        A[np.ix_(ii, ii)] += w
    np.fill_diagonal(A, 0.0)
    # Cosine-like normalization prevents genes with many pathway annotations
    # from dominating the graph penalty.
    deg = A.sum(1)
    inv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    An = inv[:, None] * A * inv[None, :]
    L = np.diag((An.sum(1))) - An
    L = (L + L.T) * 0.5
    lam, Q = np.linalg.eigh(L)
    return lam, Q, {"n_edges": int(np.count_nonzero(np.triu(A, 1))),
                    "mean_degree": float(deg.mean()),
                    "n_isolated": int(np.sum(deg == 0))}


def graph_factor(Xtr, Ytr, Q):
    """Factor X'X once; all (alpha,gamma) candidates are diagonal updates."""
    ymean = Ytr.mean(0)
    Yc = Ytr - ymean
    XtX = Xtr.T @ Xtr
    sx, V = np.linalg.eigh((XtX + XtX.T) * 0.5)
    G = V.T @ (Xtr.T @ (Yc @ Q))
    return sx, V, G, ymean


def graph_predict(Xq, factor, lam, Q, alpha, gamma):
    sx, V, G, ymean = factor
    C = V @ (G / (sx[:, None] + alpha + gamma * lam[None, :]))
    return (Xq @ C) @ Q.T + ymean


def split(ids, seed):
    rng = np.random.default_rng(seed)
    o = np.asarray(ids, dtype=int).copy(); rng.shuffle(o)
    ntr, nva = int(0.7 * len(o)), int(0.15 * len(o))
    return o[:ntr], o[ntr:ntr+nva], o[ntr+nva:]


def run():
    z = np.load(DATA / "pseudobulk.npz", allow_pickle=True)
    m = json.loads((DATA / "metadata.json").read_text())
    labels = np.asarray(m["labels"]["norman"], str)
    y = z["y_norman"].astype(np.float64)
    ctrl = np.flatnonzero(labels == "CONTROL")
    y = y - y[ctrl].mean(0)
    X, active = make_features(labels, np.asarray(m["pathway_vocab"]),
                              np.asarray(m["genes"]), z["pathways_full"].astype(np.float64))
    lam, Q, graph_stats = graph_laplacian(np.asarray(m["genes"]), m["pathway_gene_lists"])
    ids = np.flatnonzero(labels != "CONTROL")
    rows = []
    for seed in SEEDS:
        tr0, va, te = split(ids, seed)
        tr = np.r_[tr0, ctrl]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-3
        Xs = (X - mu) / sd
        gf = graph_factor(Xs[tr], y[tr], Q)
        # Existing strong baseline uses the same feature contract and alpha grid.
        base_grid = []
        for alpha in ALPHAS:
            bm = Ridge(alpha=alpha, solver="cholesky").fit(Xs[tr], y[tr])
            base_grid.append((metric(y[va], bm.predict(Xs[va])), alpha))
        base_val, base_alpha = min(base_grid, key=lambda q: q[0]["rmse"])
        cand = []
        for alpha, gamma in itertools.product(ALPHAS, GAMMAS):
            pv = graph_predict(Xs[va], gf, lam, Q, alpha, gamma)
            cand.append((metric(y[va], pv), alpha, gamma))
        best = min(cand, key=lambda q: q[0]["rmse"])
        # Lock selected alpha/gamma, then score held-out identities once.
        bp = Ridge(alpha=base_alpha, solver="cholesky").fit(Xs[tr], y[tr]).predict(Xs[te])
        cp = graph_predict(Xs[te], gf, lam, Q, best[1], best[2])
        btest, ctest = metric(y[te], bp), metric(y[te], cp)
        rows.append({
            "split_seed": seed,
            "split_hash": hashlib.sha256(np.asarray(np.r_[tr, va, te], dtype=np.int64).tobytes()).hexdigest(),
            "baseline_alpha": float(base_alpha), "candidate_alpha": float(best[1]), "candidate_gamma": float(best[2]),
            "baseline_validation": base_val, "candidate_validation": best[0],
            "baseline_test": btest, "candidate_test": ctest,
            "test_delta_candidate_minus_baseline": {k: ctest[k] - btest[k] for k in ("rmse", "pearson", "spearman")},
            "n_train": int(len(tr)), "n_validation": int(len(va)), "n_test": int(len(te)),
        })
        print(seed, rows[-1], flush=True)
    out = {
        "protocol": "validation_selected_reactome_gene_graph_multitask_ridge",
        "candidate": "output-coupled graph-regularized multi-task ridge with Reactome gene-overlap Laplacian",
        "baseline": "identity-augmented Ridge using identical 870-feature input",
        "selection": "alpha and gamma selected by validation RMSE only; test scored after lock",
        "split_seeds": SEEDS, "alpha_grid": ALPHAS, "gamma_grid": GAMMAS,
        "n_features": int(X.shape[1]), "n_active_identity_features": int(len(active)),
        "graph": graph_stats,
        "spectral_range": [float(lam.min()), float(lam.max())],
        "splits": rows,
        "all_validation_metrics_improved": all(r["candidate_validation"]["rmse"] < r["baseline_validation"]["rmse"] and r["candidate_validation"]["pearson"] > r["baseline_validation"]["pearson"] and r["candidate_validation"]["spearman"] > r["baseline_validation"]["spearman"] for r in rows),
        "all_test_metrics_improved": all(r["candidate_test"]["rmse"] < r["baseline_test"]["rmse"] and r["candidate_test"]["pearson"] > r["baseline_test"]["pearson"] and r["candidate_test"]["spearman"] > r["baseline_test"]["spearman"] for r in rows),
        "input_hashes": {"pseudobulk": hashlib.sha256((DATA / "pseudobulk.npz").read_bytes()).hexdigest(), "metadata": hashlib.sha256((DATA / "metadata.json").read_bytes()).hexdigest()},
    }
    (OUT / "gene_graph_multitask_validation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("all_validation_metrics_improved", "all_test_metrics_improved", "graph", "spectral_range")}, indent=2))


if __name__ == "__main__":
    run()
