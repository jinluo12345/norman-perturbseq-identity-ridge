"""Validation-only screen for structured low-rank identity decoders.

The strongest available baseline uses standardized [gene-panel membership,
Reactome pathway scores, active perturbation-component indicators] followed by
joint Ridge.  This script tests a genuinely different hypothesis: the mapping
from identity features to the 512-gene response is low-rank.  A Ridge fit is
used only to obtain a regularized coefficient estimate; the active-component
block is then projected onto a rank-r subspace (or the full coefficient matrix
for the global variant).  All alpha/rank selection uses the validation
partition; no test profiles are loaded or scored.
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
SEEDS = [11, 22, 33, 44, 55]
ALPHAS = [3, 10, 30, 100]
RANKS = [4, 8, 16, 32, 64]

def metric(y, p):
    a, b = np.asarray(y).ravel(), np.asarray(p).ravel()
    return {"rmse": float(np.sqrt(mean_squared_error(y, p))),
            "pearson": float(pearsonr(a, b).statistic),
            "spearman": float(spearmanr(a, b).statistic),
            "n_identities": int(len(y))}

def build():
    z = np.load(DATA / "pseudobulk.npz", allow_pickle=True)
    m = json.loads((DATA / "metadata.json").read_text())
    labs = np.asarray(m["labels"]["norman"], str)
    y = z["y_norman"].astype(np.float64)
    ctrl = np.flatnonzero(labs == "CONTROL")
    y = y - y[ctrl].mean(0)
    P = z["pathways_full"].astype(np.float64)
    vocab = np.asarray(m["pathway_vocab"]); genes = np.asarray(m["genes"])
    gi = {x: i for i, x in enumerate(vocab)}
    F = np.zeros((len(labs), len(vocab)), np.float64)
    for i, lab in enumerate(labs):
        if lab == "CONTROL":
            continue
        for tok in str(lab).replace("/", "+").replace("-", "_").split("+"):
            if tok in gi: F[i, gi[tok]] = 1.0
    active = np.flatnonzero(F.sum(0) > 0)
    A = F[:, [gi[g] for g in genes]]
    B = F @ P
    C = F[:, active]
    X = np.c_[A, B, C]
    return labs, y, X, ctrl, active

def split(labs, seed):
    ids = np.flatnonzero(labs != "CONTROL")
    rng = np.random.default_rng(seed); order = ids.copy(); rng.shuffle(order)
    ntr, nva = int(.70 * len(order)), int(.15 * len(order))
    return np.r_[order[:ntr], np.flatnonzero(labs == "CONTROL")], order[ntr:ntr+nva], order[ntr+nva:]

def truncated(W, rank):
    # Stable economy SVD and exact rank projection; rank=0 is zero block.
    if rank >= min(W.shape): return W.copy()
    if rank == 0: return np.zeros_like(W)
    u, s, vt = np.linalg.svd(W, full_matrices=False)
    return (u[:, :rank] * s[:rank]) @ vt[:rank]

def main():
    labs, y, X, ctrl, active = build()
    rows = []
    for seed in SEEDS:
        tr, va, te = split(labs, seed)
        mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-3; Xs = (X - mu) / sd
        # Candidate/baseline fits are all trained on identical rows.
        base_rows = []
        candidate_rows = []
        # Blocks: A 512, B 256, C 102. Only C is rank constrained in the
        # compositional candidate; all-block truncation is a secondary screen.
        n_a, n_b = A_N, B_N
        for alpha in ALPHAS:
            fit = Ridge(alpha=alpha, solver="cholesky").fit(Xs[tr], y[tr])
            pva = fit.predict(Xs[va]); base_rows.append({"alpha": alpha, "validation": metric(y[va], pva)})
            W = fit.coef_.copy()  # [512 outputs, features]
            # Compute each identity-block SVD once per alpha.
            Wi = W[:, n_a+n_b:].T
            ui, si, vti = np.linalg.svd(Wi, full_matrices=False)
            # Identity-block SVD, preserving the fitted gene/pathway decoder.
            for rank in RANKS:
                Wi_r = (ui[:, :rank] * si[:rank]) @ vti[:rank]
                Wr = W.copy(); Wr[:, n_a+n_b:] = Wi_r.T
                p = Xs[va] @ Wr.T + fit.intercept_
                candidate_rows.append({"kind":"identity_block_lowrank", "alpha":alpha,
                                       "rank":rank, "validation":metric(y[va],p)})
        best_base = min(base_rows, key=lambda q:q["validation"]["rmse"])
        # Primary gate is all three metrics versus validation-selected baseline.
        cand_best = min(candidate_rows, key=lambda q:q["validation"]["rmse"])
        base = next(q for q in base_rows if q["alpha"] == best_base["alpha"])
        # Best candidate under an all-metric Pareto requirement, if any.
        winners = [q for q in candidate_rows
                   if q["validation"]["rmse"] < base["validation"]["rmse"]
                   and q["validation"]["pearson"] > base["validation"]["pearson"]
                   and q["validation"]["spearman"] > base["validation"]["spearman"]]
        rows.append({"split_seed": seed,
                     "split_hash": hashlib.sha256(np.asarray(np.r_[tr,va,te], dtype=np.int64).tobytes()).hexdigest(),
                     "baseline_grid": base_rows,
                     "baseline_selected": base,
                     "candidate_grid": candidate_rows,
                     "best_candidate_by_rmse": cand_best,
                     "all_metric_winners": winners,
                     "n_train_identities": int(len(tr)-len(ctrl)),
                     "n_validation_identities": int(len(va)),
                     "n_test_identities_locked": int(len(te))})
        print(seed, 'base', best_base['alpha'], base['validation'], 'best', cand_best['kind'], cand_best['alpha'], cand_best['rank'], cand_best['validation'], 'winners',len(winners))
    out = {"protocol":"validation_only_structured_lowrank_identity_screen",
           "feature_definition":"512 gene membership + 256 Reactome scores + 102 active component indicators",
           "candidate":"Ridge coefficient projection with rank-constrained identity block or full mapping",
           "selection":"alpha and rank selected on validation RMSE; no test rows scored",
           "seeds":rows,
           "all_metric_winner_each_split": all(bool(r['all_metric_winners']) for r in rows),
           "input_hashes": {"pseudobulk":hashlib.sha256((DATA/'pseudobulk.npz').read_bytes()).hexdigest(),
                            "metadata":hashlib.sha256((DATA/'metadata.json').read_bytes()).hexdigest()}}
    (OUT/'lowrank_identity_validation_screen.json').write_text(json.dumps(out,indent=2))
    print(json.dumps({"all_metric_winner_each_split":out['all_metric_winner_each_split'],"seeds":[{"seed":r['split_seed'],"base":r['baseline_selected']['validation'],"best":r['best_candidate_by_rmse']['validation'],"kind":r['best_candidate_by_rmse']['kind'],"wins":len(r['all_metric_winners'])} for r in rows]},indent=2))

if __name__ == '__main__':
    A_N, B_N = 512, 256
    main()
