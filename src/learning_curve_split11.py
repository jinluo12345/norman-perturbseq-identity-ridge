"""Nested identity learning curve for Pathway-NEAT versus validation-tuned pathway Ridge.

The confirmation split is frozen at split seed 11.  Fractions are nested subsets
of the original non-control training identities; validation and confirmation
identities never enter fitting, normalization, or panel selection.  The panel
and pathway map are the same fixed 512-gene inputs used by the primary run.
"""
from pathlib import Path
import json, random
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

FRACTIONS = [0.25, 0.50, 0.75, 1.00]
OPT_SEEDS = [11, 22, 33]
ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


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
                Fg[i, gidx[tok]] = 1.0
    return Fg, Fg @ P


def metrics(y, p):
    y = np.asarray(y); p = np.asarray(p)
    fy, fp = y.ravel(), p.ravel()
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "pearson": float(pearsonr(fy, fp).statistic) if np.std(fp) > 1e-8 else 0.0,
        "spearman": float(spearmanr(fy, fp).statistic) if np.std(fp) > 1e-8 else 0.0,
        "n_profiles": int(len(y)),
    }


class PathwayNEAT(torch.nn.Module):
    def __init__(self, n_path, n_out, rank=16):
        super().__init__()
        self.intercept = torch.nn.Parameter(torch.zeros(n_out))
        self.path_direct = torch.nn.Parameter(torch.zeros(n_path, n_out))
        self.path_u = torch.nn.Parameter(torch.randn(n_path, rank) * 0.03)
        self.path_v = torch.nn.Parameter(torch.randn(rank, n_out) * 0.03)
        self.path_gate = torch.nn.Parameter(torch.zeros(n_path))
        self.gene = torch.nn.Linear(n_out, n_out, bias=False)

    def forward(self, Fg, Fp):
        path = (Fp * torch.nn.functional.softplus(self.path_gate)) @ self.path_u @ self.path_v
        return self.intercept + path + Fp @ self.path_direct + self.gene(Fg)


def fit_neat(Fg, Fp, y, tr, va, te, opt_seed, device):
    set_seed(opt_seed)
    fpmean, fpstd = Fp[tr].mean(0), Fp[tr].std(0) + 1e-3
    Fps = (Fp - fpmean) / fpstd
    ym, ysdev = y[tr].mean(0), y[tr].std(0) + 1e-3
    ys = (y - ym) / ysdev
    model = PathwayNEAT(Fp.shape[1], y.shape[1], rank=16).to(device)
    # Match the primary implementation's warm start from the pathway Ridge.
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
        pred = model(Xg[tr], Xp[tr]); loss = ((pred - Y[tr]) ** 2).mean()
        loss.backward(); opt.step(); model.eval()
        with torch.no_grad():
            train_mse = ((model(Xg[tr], Xp[tr]) - Y[tr]) ** 2).mean().item()
            val_mse = ((model(Xg[va], Xp[va]) - Y[va]) ** 2).mean().item()
        hist.append({"epoch": ep + 1, "train_mse_std": float(train_mse), "val_mse_std": float(val_mse)})
        if val_mse < best:
            best, stall = val_mse, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stall += 1
        if stall >= 70:
            break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xg, Xp).detach().cpu().numpy() * ysdev + ym
    return {
        "optimization_seed": int(opt_seed),
        "epochs_run": int(len(hist)),
        "best_epoch": int(np.argmin([h["val_mse_std"] for h in hist]) + 1),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "train_metrics": metrics(y[tr], pred[tr]),
        "validation_metrics": metrics(y[va], pred[va]),
        "test_metrics": metrics(y[te], pred[te]),
        "learning_curve": hist,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the learning-curve training")
    device = "cuda"
    z = np.load(DATA / "pseudobulk.npz", allow_pickle=True)
    meta = json.loads((DATA / "metadata.json").read_text())
    labels = np.asarray(meta["labels"]["norman"])
    tv = np.asarray(meta["pathway_vocab"])
    P = z["pathways_full"].astype(np.float32)
    y = z["y_norman"].astype(np.float32)
    controls = np.flatnonzero(labels == "CONTROL")
    y = y - y[controls].mean(0)
    Fg_full, Fp = features(labels.tolist(), tv, P)
    out_idx = np.asarray([int(np.where(tv == g)[0][0]) for g in np.asarray(meta["genes"])])
    Fg = Fg_full[:, out_idx]

    # Exactly reproduce the confirmation split used in train_models.py.
    ids = np.asarray([i for i, lab in enumerate(labels) if lab != "CONTROL"], dtype=int)
    rng = np.random.default_rng(11); rng.shuffle(ids)
    ntr, nva = int(0.70 * len(ids)), int(0.15 * len(ids))
    original_train, va, te = ids[:ntr], ids[ntr:ntr + nva], ids[ntr + nva:]
    tr_control = np.concatenate([original_train, controls])

    rows = []
    results = {
        "device": device, "split_seed": 11, "fractions": FRACTIONS,
        "optimization_seeds": OPT_SEEDS, "ridge_alpha_grid": ALPHAS,
        "panel_size": int(Fg.shape[1]), "n_pathways": int(Fp.shape[1]),
        "split": {"original_train_indices": original_train.tolist(),
                  "validation_indices": va.tolist(), "test_indices": te.tolist(),
                  "original_train_labels": labels[original_train].tolist(),
                  "validation_labels": labels[va].tolist(), "test_labels": labels[te].tolist(),
                  "test_labels_untouched": True},
        "runs": []
    }
    for fraction in FRACTIONS:
        k = max(1, int(np.ceil(fraction * len(original_train))))
        tr_ids = original_train[:k]
        tr = np.concatenate([tr_ids, controls])
        # Fit standardization only on this nested fraction (including control).
        fpmean, fpstd = Fp[tr].mean(0), Fp[tr].std(0) + 1e-3
        Fps = (Fp - fpmean) / fpstd
        # Validation-only alpha selection.  The selected model is fit on tr;
        # validation labels select alpha but are never used in test fitting.
        alpha_scores = []
        for alpha in ALPHAS:
            rg = Ridge(alpha=alpha).fit(Fps[tr], y[tr])
            alpha_scores.append({"alpha": alpha, **metrics(y[va], rg.predict(Fps[va]))})
        selected = min(alpha_scores, key=lambda r: (r["rmse"], r["alpha"]))
        ridge = Ridge(alpha=selected["alpha"]).fit(Fps[tr], y[tr])
        ridge_train = metrics(y[tr], ridge.predict(Fps[tr]))
        ridge_val = metrics(y[va], ridge.predict(Fps[va]))
        ridge_test = metrics(y[te], ridge.predict(Fps[te]))
        neat_runs = [fit_neat(Fg, Fp, y, tr, va, te, s, device) for s in OPT_SEEDS]
        run = {
            "fraction": float(fraction), "n_training_identities": int(len(tr_ids)),
            "n_training_rows_including_control": int(len(tr)),
            "train_indices": tr.tolist(), "train_labels": labels[tr].tolist(),
            "ridge": {"selected_alpha": float(selected["alpha"]),
                      "alpha_validation_grid": alpha_scores,
                      "train_metrics": ridge_train, "validation_metrics": ridge_val,
                      "test_metrics": ridge_test},
            "neat_runs": neat_runs,
            "neat_test_mean": {k: float(np.mean([r["test_metrics"][k] for r in neat_runs]))
                               for k in ["rmse", "pearson", "spearman"]},
            "neat_test_sd": {k: float(np.std([r["test_metrics"][k] for r in neat_runs], ddof=1))
                             for k in ["rmse", "pearson", "spearman"]},
        }
        run["paired_delta_neat_minus_ridge"] = {
            k: float(run["neat_test_mean"][k] - ridge_test[k])
            for k in ["rmse", "pearson", "spearman"]
        }
        results["runs"].append(run)
        for nr in neat_runs:
            rows.append({"fraction": fraction, "n_training_identities": len(tr_ids),
                         "model": "Pathway-NEAT", "optimization_seed": nr["optimization_seed"],
                         "parameter_count": nr["parameter_count"], "epochs_run": nr["epochs_run"],
                         "best_epoch": nr["best_epoch"], "selected_ridge_alpha": selected["alpha"],
                         **{f"train_{k}": nr["train_metrics"][k] for k in ["rmse", "pearson", "spearman"]},
                         **{f"validation_{k}": nr["validation_metrics"][k] for k in ["rmse", "pearson", "spearman"]},
                         **{f"test_{k}": nr["test_metrics"][k] for k in ["rmse", "pearson", "spearman"]}})
        rows.append({"fraction": fraction, "n_training_identities": len(tr_ids),
                     "model": "Pathway Ridge", "optimization_seed": "deterministic",
                     "parameter_count": int(Fp.shape[1] * y.shape[1] + y.shape[1]),
                     "epochs_run": 0, "best_epoch": 0, "selected_ridge_alpha": selected["alpha"],
                     **{f"train_{k}": ridge_train[k] for k in ["rmse", "pearson", "spearman"]},
                     **{f"validation_{k}": ridge_val[k] for k in ["rmse", "pearson", "spearman"]},
                     **{f"test_{k}": ridge_test[k] for k in ["rmse", "pearson", "spearman"]}})
        print(json.dumps({"fraction": fraction, "n_train": len(tr_ids),
                          "ridge_test": ridge_test, "neat_test_mean": run["neat_test_mean"],
                          "delta": run["paired_delta_neat_minus_ridge"]}), flush=True)
    (OUT / "learning_curve_split11.json").write_text(json.dumps(results))
    pd.DataFrame(rows).to_csv(OUT / "learning_curve_split11.csv", index=False)


if __name__ == "__main__":
    main()
