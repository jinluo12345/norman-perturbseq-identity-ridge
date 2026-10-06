"""Retrospective intervention-priority simulation on held-out Norman combinations.

The simulation is validation-only: each split holds out combinations while all
single perturbations and a subset of combinations remain available for fitting.
The fitted identity/pathway/component ridge is used to rank candidate
combinations by predicted control-relative response energy.  Observed response
energy defines the retrospective hit set.  No confirmation identity is used.
"""
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


def pathway_projection(y, panel_idx, pathways):
    """Project 512-gene profiles into the fixed 256 Reactome modules."""
    W = pathways[panel_idx]
    den = W.sum(0) + 1e-8
    return y @ W / den


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
            if t in ci:
                F[i, ci[t]] = 1.0
    G = np.zeros((len(labels), len(genes)), float)
    for j, g in enumerate(genes):
        if g in ci:
            G[:, j] = F[:, ci[g]]
    P = F @ pathways[[vi[g] for g in components]] if components else np.zeros((len(labels), pathways.shape[1]))
    X = np.c_[G, P, F]
    panel_idx = np.asarray([vi[g] for g in genes], dtype=int)
    return labels, y, X, pathways, panel_idx


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
        train_combo, val_combo, test_combo = order[:ntr], order[ntr:ntr + nva], order[ntr + nva:]
        train = np.r_[controls, np.flatnonzero(np.asarray([lab != "CONTROL" and "+" not in lab for lab in labels])), train_combo]
        mu, sd = X[train].mean(0), X[train].std(0) + 1e-3
        Xs = (X - mu) / sd
        # Alpha is selected using validation profile RMSE, with no ranking metric
        # or held-out combination outcome entering the selection.
        fits = []
        for alpha in ALPHA_GRID:
            fit = Ridge(alpha=alpha, solver="cholesky").fit(Xs[train], y[train])
            pred = fit.predict(Xs[val_combo])
            rmse = float(np.sqrt(np.mean((pred - y[val_combo]) ** 2)))
            fits.append((rmse, alpha, fit))
        _, alpha, fit = min(fits, key=lambda q: q[0])
        pred = fit.predict(Xs[val_combo])
        obs = y[val_combo]
        pred_energy = np.sqrt(np.mean(pred * pred, axis=1))
        obs_energy = np.sqrt(np.mean(obs * obs, axis=1))
        pred_path = pathway_projection(pred, panel_idx, pathways)
        obs_path = pathway_projection(obs, panel_idx, pathways)
        p_spear = float(spearmanr(pred_path.ravel(), obs_path.ravel()).statistic)
        p_cos = float(np.mean(np.sum(pred_path * obs_path, axis=1) / (np.linalg.norm(pred_path, axis=1) * np.linalg.norm(obs_path, axis=1) + 1e-12)))
        rank_spear = float(spearmanr(pred_energy, obs_energy).statistic)
        top = {}
        for k in TOP_K:
            k = min(k, len(val_combo))
            ps, os = set(np.argsort(-pred_energy)[:k]), set(np.argsort(-obs_energy)[:k])
            top[str(k)] = {
                "predicted_hit_count": int(len(ps & os)),
                "hit_rate_precision": float(len(ps & os) / k),
                "topk_overlap": float(len(ps & os) / k),
                "observed_recall": float(len(ps & os) / k),
                "ndcg": ndcg_at_k(pred_energy, obs_energy, k),
                "relative_assay_fraction": float(k / len(val_combo)),
                "selected_labels": [str(labels[val_combo[i]]) for i in np.argsort(-pred_energy)[:k]],
            }
        rows.append({
            "seed": seed, "alpha_selected_by_validation_rmse": int(alpha),
            "n_validation_combinations": int(len(val_combo)), "n_locked_test_combinations": int(len(test_combo)),
            "rank_spearman_response_energy": rank_spear,
            "pathway_profile_spearman": p_spear, "pathway_profile_cosine_mean": p_cos,
            "top_k": top,
            "validation_labels": [str(x) for x in labels[val_combo]],
        })
    stability = {}
    for k in TOP_K:
        # Use selected labels already recorded, which are the auditable ranking output.
        sets = [set(r["top_k"][str(min(k, r["n_validation_combinations"]))]["selected_labels"]) for r in rows]
        jacc = [len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1:]]
        stability[str(k)] = {"mean_pairwise_jaccard": float(np.mean(jacc)), "pairwise_jaccard": [float(x) for x in jacc]}
    out = {
        "protocol": "validation_only_retrospective_intervention_priority_simulation",
        "scientific_question": "Can the locked identity/pathway/component predictor prioritize combinations for follow-up assays using predicted response energy?",
        "candidate_definition": "rank by RMS control-relative 512-gene response energy; observed energy defines retrospective hit set",
        "pathway_consistency": "fixed 256 Reactome module projection of the 512-gene response; Spearman and mean cosine",
        "cost_definition": "relative assay fraction k / number of validation combinations; no monetary cost is imputed",
        "alpha_selection": "validation profile RMSE only; no locked test combination is scored or used for selection",
        "seeds": rows, "stability": stability,
        "input_hashes": {"pseudobulk": hashlib.sha256((DATA / "pseudobulk.npz").read_bytes()).hexdigest(), "metadata": hashlib.sha256((DATA / "metadata.json").read_bytes()).hexdigest()},
    }
    (OUT / "intervention_priority_simulation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"n_seeds": len(rows), "stability": stability, "mean_pathway_spearman": float(np.mean([r["pathway_profile_spearman"] for r in rows]))}, indent=2))


if __name__ == "__main__":
    main()
