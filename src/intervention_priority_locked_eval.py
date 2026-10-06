"""Locked two-component priority evaluation with frozen ranking and bootstrap rules."""
from pathlib import Path
import hashlib, json
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
OUT = ROOT / "results"
SEEDS = [11, 22, 33, 44, 55]
ALPHA_GRID = [1, 3, 10, 30, 100]
TOP_K = [5, 10, 20]
BOOT = 2000


def toks(label):
    return [] if label == "CONTROL" else str(label).split("+")


def ndcg_at_k(pred, obs, k):
    k = min(k, len(pred))
    order = np.argsort(-pred)
    ideal = np.sort(obs)[::-1][:k]
    gain = np.maximum(obs[order[:k]], 0.0)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gain * disc))
    idcg = float(np.sum(np.maximum(ideal, 0.0) * disc))
    return dcg / idcg if idcg > 0 else 0.0


def bootstrap_metrics(pred, obs, k, draws):
    """Paired identity-cluster sensitivity using one common draw matrix.

    The same resampled locked-combination identities are reused for every k
    and endpoint within a split so that differences are paired and auditable.
    """
    n = len(pred)
    vals = []
    for ix in draws:
        p, o = pred[ix], obs[ix]
        order_p = np.argsort(-p)[:min(k, n)]
        order_o = np.argsort(-o)[:min(k, n)]
        hit = len(set(order_p.tolist()) & set(order_o.tolist()))
        kk = min(k, n)
        vals.append({
            "precision": hit / kk,
            "recall": hit / kk,
            "ndcg": ndcg_at_k(p, o, kk),
            "energy_spearman": float(spearmanr(p, o).statistic) if len(np.unique(p)) > 1 and len(np.unique(o)) > 1 else 0.0,
        })
    out = {}
    for key in vals[0]:
        a = np.asarray([v[key] for v in vals], float)
        out[key] = {"point": float(np.mean(a)), "ci95": [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]}
    return out


def build():
    z = np.load(DATA / "pseudobulk.npz", allow_pickle=True)
    m = json.loads((DATA / "metadata.json").read_text())
    labels = np.asarray(m["labels"]["norman"], str)
    y = z["y_norman"].astype(float)
    ctrl = np.flatnonzero(labels == "CONTROL")
    y = y - y[ctrl].mean(0)
    vocab = np.asarray(m["pathway_vocab"])
    genes = np.asarray(m["genes"])
    pathways = z["pathways_full"].astype(float)
    vi = {g: i for i, g in enumerate(vocab)}
    components = sorted({t for lab in labels for t in toks(lab) if t in vi})
    ci = {g: i for i, g in enumerate(components)}
    F = np.zeros((len(labels), len(components)), float)
    for i, lab in enumerate(labels):
        for t in toks(lab):
            if t in ci: F[i, ci[t]] = 1.0
    G = np.zeros((len(labels), len(genes)), float)
    for j, g in enumerate(genes):
        if g in ci: G[:, j] = F[:, ci[g]]
    P = F @ pathways[[vi[g] for g in components]] if components else np.zeros((len(labels), pathways.shape[1]))
    X = np.c_[G, P, F]
    panel_idx = np.asarray([vi[g] for g in genes], dtype=int)
    return labels, y, X, pathways, panel_idx


def project(y, panel_idx, pathways):
    W = pathways[panel_idx]
    return y @ W / (W.sum(0) + 1e-8)


def main():
    labels, y, X, pathways, panel_idx = build()
    singleton = {lab: i for i, lab in enumerate(labels) if lab != "CONTROL" and "+" not in lab}
    combo = np.asarray([i for i, lab in enumerate(labels) if "+" in lab and all(t in singleton for t in toks(lab))], dtype=int)
    controls = np.flatnonzero(labels == "CONTROL")
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        order = combo.copy(); rng.shuffle(order)
        ntr, nva = int(0.6 * len(order)), int(0.2 * len(order))
        train_combo, val_combo, test_combo = order[:ntr], order[ntr:ntr+nva], order[ntr+nva:]
        train = np.r_[controls, np.flatnonzero(np.asarray([lab != "CONTROL" and "+" not in lab for lab in labels])), train_combo]
        mu, sd = X[train].mean(0), X[train].std(0) + 1e-3
        Xs = (X - mu) / sd
        fit_rows = []
        for alpha in ALPHA_GRID:
            fit = Ridge(alpha=alpha, solver="cholesky").fit(Xs[train], y[train])
            p = fit.predict(Xs[val_combo])
            fit_rows.append((float(np.sqrt(np.mean((p-y[val_combo])**2))), alpha, fit))
        _, alpha, fit = min(fit_rows, key=lambda q: q[0])
        pred = fit.predict(Xs[test_combo]); obs = y[test_combo]
        draw_seed = int(seed * 1000 + 791)
        draw_rng = np.random.default_rng(draw_seed)
        common_draws = draw_rng.integers(0, len(test_combo), size=(BOOT, len(test_combo)), dtype=np.int32)
        draw_hash = hashlib.sha256(common_draws.tobytes()).hexdigest()
        pred_e = np.sqrt(np.mean(pred*pred, axis=1)); obs_e = np.sqrt(np.mean(obs*obs, axis=1))
        top = {}
        for k in TOP_K:
            kk = min(k, len(test_combo)); pi = np.argsort(-pred_e)[:kk]; oi = np.argsort(-obs_e)[:kk]; hit = len(set(pi.tolist()) & set(oi.tolist()))
            top[str(k)] = {
                "k": kk, "denominator": int(len(test_combo)), "precision": float(hit/kk), "recall": float(hit/kk), "overlap": float(hit/kk),
                "ndcg": ndcg_at_k(pred_e, obs_e, kk), "assay_fraction": float(kk/len(test_combo)),
                "null_overlap": float(kk/len(test_combo)), "enrichment": float((hit/kk)/(kk/len(test_combo))),
                "selected_labels": [str(labels[test_combo[i]]) for i in pi],
                "bootstrap": bootstrap_metrics(pred_e, obs_e, kk, common_draws),
            }
        pred_p = project(pred, panel_idx, pathways); obs_p = project(obs, panel_idx, pathways)
        rows.append({
            "seed": seed, "alpha_selected_by_validation_rmse": int(alpha), "n_validation_combinations": int(len(val_combo)), "n_locked_test_combinations": int(len(test_combo)),
            "bootstrap_draws": {"seed": draw_seed, "B": BOOT, "unit": "locked combination identity", "shape": [BOOT, int(len(test_combo))], "hash_sha256": draw_hash, "pairing": "common matrix reused across k and endpoints"},
            "locked_energy_spearman": float(spearmanr(pred_e, obs_e).statistic),
            "locked_pathway_spearman": float(spearmanr(pred_p.ravel(), obs_p.ravel()).statistic),
            "locked_pathway_cosine_mean": float(np.mean(np.sum(pred_p*obs_p,axis=1)/(np.linalg.norm(pred_p,axis=1)*np.linalg.norm(obs_p,axis=1)+1e-12))),
            "top_k": top, "locked_labels": [str(x) for x in labels[test_combo]],
        })
    stability = {}
    for k in TOP_K:
        sets = [set(r["top_k"][str(k)]["selected_labels"]) for r in rows]
        j = [len(a&b)/len(a|b) for i,a in enumerate(sets) for b in sets[i+1:]]
        stability[str(k)] = {"mean_pairwise_jaccard": float(np.mean(j)), "pairwise_jaccard": [float(x) for x in j]}
    out = {
        "protocol": "locked_validation_selected_retrospective_priority_evaluation",
        "scientific_question": "Can the pre-specified response-energy rank enrich high-response held-out Norman combinations?",
        "hit_definition": "top-k locked observed response energy, used only as a retrospective endpoint",
        "ranking_definition": "top-k predicted 512-gene response energy; higher first",
        "selection": "alpha chosen by validation profile RMSE; locked combinations are scored once and never used for tuning",
        "top_k": TOP_K, "bootstrap": {"B": BOOT, "unit": "locked combination identity", "interval": "percentile sensitivity interval"},
        "seeds": rows, "stability": stability,
        "input_hashes": {"script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "pseudobulk": hashlib.sha256((DATA/'pseudobulk.npz').read_bytes()).hexdigest(), "metadata": hashlib.sha256((DATA/'metadata.json').read_bytes()).hexdigest()},
    }
    (OUT/'intervention_priority_locked_evaluation.json').write_text(json.dumps(out, indent=2))
    print(json.dumps({"path": str(OUT/'intervention_priority_locked_evaluation.json'), "stability": stability, "locked_seeds": len(rows)}, indent=2))

if __name__ == "__main__": main()
