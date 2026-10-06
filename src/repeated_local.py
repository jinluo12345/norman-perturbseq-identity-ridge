"""Split-local Norman refits for the Pathway-NEAT audit.

This job deliberately recomputes the response panel inside every identity split
from the sampled Norman cells.  No held-out identity contributes to prevalence
based panel selection.  The resulting JSON contains split manifests, panels,
learning curves, optimization-seed metrics and paired NEAT-minus-ridge deltas.
It is intended for an offline qzcli GPU run.
"""
from pathlib import Path
import json, random, sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
import anndata as ad

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
RAW = ROOT / "data/raw"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))
from preprocess import clean_gene, obs_target  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def target_tokens(label):
    if label in {"CONTROL", "NAN", "*"}:
        return []
    return [x for x in str(label).replace("/", "+").replace("-", "_").split("+")
            if x and x not in {"ONLY", "MOD"}]


def features(labels, target_vocab, P):
    gidx = {g: i for i, g in enumerate(target_vocab)}
    Fg = np.zeros((len(labels), len(target_vocab)), np.float32)
    for i, lab in enumerate(labels):
        for tok in target_tokens(lab):
            if tok in gidx:
                Fg[i, gidx[tok]] = 1
    return Fg, Fg @ P


class PathwayNEAT(nn.Module):
    def __init__(self, n_path, n_out, rank=16, use_path=True, use_gene=True, use_direct=True):
        super().__init__()
        self.use_path, self.use_gene, self.use_direct = use_path, use_gene, use_direct
        self.intercept = nn.Parameter(torch.zeros(n_out))
        self.path_direct = nn.Parameter(torch.zeros(n_path, n_out)) if use_direct else None
        self.path_u = nn.Parameter(torch.randn(n_path, rank) * 0.03)
        self.path_v = nn.Parameter(torch.randn(rank, n_out) * 0.03)
        self.path_gate = nn.Parameter(torch.zeros(n_path))
        self.gene = nn.Linear(n_out, n_out, bias=False) if use_gene else None

    def forward(self, Fg, Fp):
        terms = []
        if self.use_path:
            terms.append((Fp * torch.nn.functional.softplus(self.path_gate)) @ self.path_u @ self.path_v)
            if self.use_direct:
                terms.append(Fp @ self.path_direct)
        if self.use_gene:
            terms.append(self.gene(Fg))
        y = self.intercept
        if terms:
            y = y + sum(terms)
        return y, terms


def metrics(y, p):
    y = np.asarray(y); p = np.asarray(p)
    fy, fp = y.ravel(), p.ravel()
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "pearson": float(pearsonr(fy, fp).statistic) if np.std(fp) > 1e-8 else 0.0,
        "spearman": float(spearmanr(fy, fp).statistic) if np.std(fp) > 1e-8 else 0.0,
        "n_profiles": int(len(y)),
    }


def split_identity(labels, seed):
    ids = np.array([i for i, lab in enumerate(labels) if lab != "CONTROL"], dtype=int)
    rng = np.random.default_rng(seed); rng.shuffle(ids)
    ntr, nva = int(0.70 * len(ids)), int(0.15 * len(ids))
    tr0, va, te = ids[:ntr], ids[ntr:ntr + nva], ids[ntr + nva:]
    controls = np.array([i for i, lab in enumerate(labels) if lab == "CONTROL"], dtype=int)
    return np.concatenate([tr0, controls]), va, te


def normalize_and_aggregate(a, cells, col_idx, labels, unique_labels):
    """Read selected raw counts, log-normalize and produce label pseudobulks."""
    X = a[cells, col_idx].X
    if sparse.issparse(X):
        X = X.tocsr().astype(np.float32)
        lib = np.asarray(X.sum(axis=1)).ravel(); lib[lib <= 0] = 1
        X = X.multiply((1e4 / lib)[:, None]).log1p().tocsr()
    else:
        X = np.asarray(X, dtype=np.float32)
        lib = X.sum(axis=1); lib[lib <= 0] = 1
        X = np.log1p(X * (1e4 / lib)[:, None])
    out = []
    for lab in unique_labels:
        ix = np.flatnonzero(labels == lab)
        if sparse.issparse(X):
            out.append(np.asarray(X[ix].mean(axis=0)).ravel())
        else:
            out.append(X[ix].mean(axis=0))
    return np.asarray(out, dtype=np.float32)


def fit_neat(Fg, Fp, y, tr, va, te, P, opt_seed, device):
    """Train one optimization seed with the original NEAT settings."""
    set_seed(opt_seed)
    fpmean, fpstd = Fp[tr].mean(0), Fp[tr].std(0) + 1e-3
    Fps = (Fp - fpmean) / fpstd
    ym, ysdev = y[tr].mean(0), y[tr].std(0) + 1e-3
    ys = (y - ym) / ysdev
    model = PathwayNEAT(P.shape[1], y.shape[1], rank=16, use_path=True, use_gene=True, use_direct=True).to(device)
    init = Ridge(alpha=0.1).fit(Fps[tr], ys[tr])
    model.path_direct.data.copy_(torch.tensor(init.coef_.T, dtype=torch.float32, device=device))
    Xg = torch.tensor(Fg, dtype=torch.float32, device=device)
    Xp = torch.tensor(Fps, dtype=torch.float32, device=device)
    Y = torch.tensor(ys, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-3)
    best, best_state, stall = float("inf"), None, 0
    hist = []
    for ep in range(1000):
        model.train(); opt.zero_grad()
        pred, _ = model(Xg[tr], Xp[tr]); loss = ((pred - Y[tr]) ** 2).mean()
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            tl = ((model(Xg[tr], Xp[tr])[0] - Y[tr]) ** 2).mean().item()
            vl = ((model(Xg[va], Xp[va])[0] - Y[va]) ** 2).mean().item()
        hist.append({"epoch": ep + 1, "train_mse_std": float(tl), "val_mse_std": float(vl)})
        if vl < best:
            best, stall = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stall += 1
        if stall >= 70:
            break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_test = model(Xg[te], Xp[te])[0].detach().cpu().numpy() * ysdev + ym
    return {
        "optimization_seed": int(opt_seed),
        "epochs_run": int(len(hist)),
        "best_epoch": int(np.argmin([h["val_mse_std"] for h in hist]) + 1),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "metrics": metrics(y[te], pred_test),
        "learning_curve": hist,
        "pred_test": pred_test.tolist(),
    }


def get_common_genes(norm_genes):
    """Use only symbols shared by the available public archives for comparability."""
    common = set(norm_genes)
    for name in ["adamson_single", "adamson_combo", "dixit"]:
        path = {"adamson_single": RAW / "AdamsonWeissman2016_GSM2406675_10X001.h5ad",
                "adamson_combo": RAW / "AdamsonWeissman2016_GSM2406677_10X005.h5ad",
                "dixit": RAW / "DixitRegev2016_K562_TFs_13_days.h5ad"}[name]
        if path.exists():
            ext = ad.read_h5ad(path, backed="r")
            common.intersection_update(clean_gene(x) for x in ext.var_names)
            ext.file.close()
    return common


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for repeated split refit")
    device = "cuda"
    z = np.load(DATA / "pseudobulk.npz", allow_pickle=True)
    meta = json.loads((DATA / "metadata.json").read_text())
    tv = np.asarray(meta["pathway_vocab"])
    Pfull = z["pathways_full"].astype(np.float32)
    labels = np.asarray(meta["labels"]["norman"])
    raw_path = RAW / "NormanWeissman2019_filtered.h5ad"
    a = ad.read_h5ad(raw_path, backed="r")
    norm_genes_raw = np.asarray([clean_gene(x) for x in a.var_names])
    # Reproduce the deterministic identity-stratified 60k sampled cell set used
    # by preprocessing; this keeps response aggregation comparable to the main run.
    target = obs_target(a.obs, "norman").to_numpy()
    rng = np.random.default_rng(17); keep = []
    for lab in sorted(set(target)):
        ix = np.flatnonzero(target == lab); n = max(8, int(60000 * len(ix) / len(target)))
        keep.extend(rng.choice(ix, min(len(ix), n), replace=False).tolist())
    keep = np.array(sorted(set(keep)), dtype=int)
    sampled_labels = target[keep]
    unique_labels = np.asarray(meta["labels"]["norman"])
    common = get_common_genes(norm_genes_raw)
    # First-occurrence mapping mirrors preprocess.py and pathway_vocab.
    tv_idx = {g: i for i, g in enumerate(tv)}
    raw_idx = {}
    for i, g in enumerate(norm_genes_raw):
        if g in tv_idx and g not in raw_idx:
            raw_idx[g] = i
    split_seeds = [11, 22, 33, 44, 55]
    opt_seeds = [11, 22, 33]
    all_runs = []
    for split_seed in split_seeds:
        tr, va, te = split_identity(unique_labels.tolist(), split_seed)
        tr_labs, va_labs, te_labs = unique_labels[tr], unique_labels[va], unique_labels[te]
        train_identity_set = set(tr_labs) - {"CONTROL"}
        train_cells = keep[np.isin(sampled_labels, list(train_identity_set | {"CONTROL"}))]
        # Panel selection reads raw counts only from training identities.
        Xtr = a[train_cells, :].X
        prev = np.asarray(Xtr.getnnz(axis=0)).ravel() if sparse.issparse(Xtr) else np.count_nonzero(Xtr, axis=0)
        order = np.argsort(prev, kind="stable")[::-1]
        panel = [norm_genes_raw[i] for i in order if norm_genes_raw[i] in common and norm_genes_raw[i] in tv_idx][:512]
        panel = np.asarray(panel, dtype=str)
        if len(panel) != 512:
            raise RuntimeError(f"split {split_seed} yielded {len(panel)} panel genes")
        col_idx = np.asarray([raw_idx[g] for g in panel], dtype=int)
        # Aggregate all sampled profiles after panel is frozen.  Test cells enter
        # only as the held-out response, never panel prevalence or model fitting.
        y_all = normalize_and_aggregate(a, keep, col_idx, sampled_labels, unique_labels)
        c = np.flatnonzero(unique_labels == "CONTROL")[0]
        y_all = y_all - y_all[c]
        panel_tv_idx = np.asarray([tv_idx[g] for g in panel], dtype=int)
        # Pathway incidence is defined on the full target vocabulary.  The
        # split-local panel changes only the output/residual gene columns; the
        # perturbation-to-pathway input remains the full 33,694-gene map.
        Fg_full, Fp = features(unique_labels.tolist(), tv, Pfull)
        Fg = Fg_full[:, panel_tv_idx]
        fpmean, fpstd = Fp[tr].mean(0), Fp[tr].std(0) + 1e-3
        Fps = (Fp - fpmean) / fpstd
        # Linear baselines share the exact split-local panel and standardized Fp.
        ridge_p = Ridge(alpha=0.1).fit(Fps[tr], y_all[tr])
        ridge_g = Ridge(alpha=10.0).fit(Fg[tr], y_all[tr])
        rp = ridge_p.predict(Fps[te]); rg = ridge_g.predict(Fg[te])
        neat_runs = [fit_neat(Fg, Fp, y_all, tr, va, te, Pfull, s, device) for s in opt_seeds]
        neat_mean = {k: float(np.mean([r["metrics"][k] for r in neat_runs])) for k in ["rmse", "pearson", "spearman"]}
        neat_sd = {k: float(np.std([r["metrics"][k] for r in neat_runs], ddof=1)) for k in ["rmse", "pearson", "spearman"]}
        all_runs.append({
            "split_seed": int(split_seed),
            "split": {"train_indices": tr.tolist(), "validation_indices": va.tolist(), "test_indices": te.tolist(),
                      "train_labels": tr_labs.tolist(), "validation_labels": va_labs.tolist(), "test_labels": te_labs.tolist()},
            "panel_selection": {"method": "training-identity raw-count prevalence", "n_training_cells": int(len(train_cells)),
                                "n_candidate_genes": int(len(common)), "n_panel_genes": int(len(panel)),
                                "panel_genes": panel.tolist(), "panel_prevalence_top": [int(prev[raw_idx[g]]) for g in panel[:20]],
                                "selection_identity_labels": sorted(train_identity_set | {"CONTROL"}),
                                "selection_identity_overlap_with_test": sorted((train_identity_set | {"CONTROL"}) & set(te_labs.tolist())),
                                "test_identities_used_for_selection": False},
            "models": {"pathway_ridge": {**metrics(y_all[te], rp), "alpha": 0.1},
                       "gene_ridge": {**metrics(y_all[te], rg), "alpha": 10.0},
                       "pathway_neat": {"mean": neat_mean, "sd": neat_sd, "runs": neat_runs}},
            "paired_deltas": [{"optimization_seed": int(r["optimization_seed"]),
                               "rmse_neat_minus_ridge": float(r["metrics"]["rmse"] - metrics(y_all[te], rp)["rmse"]),
                               "pearson_neat_minus_ridge": float(r["metrics"]["pearson"] - metrics(y_all[te], rp)["pearson"]),
                               "spearman_neat_minus_ridge": float(r["metrics"]["spearman"] - metrics(y_all[te], rp)["spearman"])} for r in neat_runs],
            "n_profiles": {"train_including_control": int(len(tr)), "validation": int(len(va)), "test_noncontrol": int(len(te))},
        })
        print(json.dumps({"split_seed": split_seed, "panel_head": panel[:5].tolist(),
                          "ridge": all_runs[-1]["models"]["pathway_ridge"],
                          "neat_mean": neat_mean}), flush=True)
    a.file.close()
    summary = {"device": device, "split_seeds": split_seeds, "optimization_seeds": opt_seeds,
               "panel_size": 512, "runs": all_runs}
    (OUT / "repeated_local_refit.json").write_text(json.dumps(summary))
    rows = []
    for r in all_runs:
        for d in r["paired_deltas"]:
            rows.append({"split_seed": r["split_seed"], **d,
                         "ridge_rmse": r["models"]["pathway_ridge"]["rmse"],
                         "ridge_pearson": r["models"]["pathway_ridge"]["pearson"]})
    pd.DataFrame(rows).to_csv(OUT / "repeated_local_deltas.csv", index=False)
    pd.DataFrame([{"split_seed": r["split_seed"],
                    "ridge_rmse": r["models"]["pathway_ridge"]["rmse"],
                    "ridge_pearson": r["models"]["pathway_ridge"]["pearson"],
                    "neat_rmse_mean": r["models"]["pathway_neat"]["mean"]["rmse"],
                    "neat_pearson_mean": r["models"]["pathway_neat"]["mean"]["pearson"],
                    "neat_rmse_sd": r["models"]["pathway_neat"]["sd"]["rmse"],
                    "neat_pearson_sd": r["models"]["pathway_neat"]["sd"]["pearson"],
                    "n_test": r["n_profiles"]["test_noncontrol"]} for r in all_runs]).to_csv(OUT / "repeated_local_summary.csv", index=False)


if __name__ == "__main__":
    main()
