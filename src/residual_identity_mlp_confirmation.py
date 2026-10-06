"""Locked confirmation for the residual correction selected on validation.

The architecture, split, optimizer, and lambda are frozen before this script is
run.  No test metric is used for model or hyperparameter selection.
"""
from pathlib import Path
import argparse, hashlib, json, random
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
OUT = ROOT / "results"
LOCKED_LAMBDA = 0.25
SPLIT_SEED = 11


def metric(y, p):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "pearson": float(pearsonr(y.ravel(), p.ravel()).statistic),
        "spearman": float(spearmanr(y.ravel(), p.ravel()).statistic),
        "n_identities": int(len(y)),
    }


class Net(nn.Module):
    def __init__(self, d, o):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, o),
        )

    def forward(self, x):
        return self.net(x)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_data():
    z = np.load(DATA / "pseudobulk.npz", allow_pickle=True)
    m = json.loads((DATA / "metadata.json").read_text())
    labs = np.asarray(m["labels"]["norman"], str)
    y = z["y_norman"].astype(np.float32)
    ctrl = np.flatnonzero(labs == "CONTROL")
    y = y - y[ctrl].mean(0)
    pathways = z["pathways_full"].astype(np.float32)
    vocab = np.asarray(m["pathway_vocab"])
    genes = np.asarray(m["genes"])
    gi = {x: i for i, x in enumerate(vocab)}
    F = np.zeros((len(labs), len(vocab)), np.float32)
    for i, label in enumerate(labs):
        if label != "CONTROL":
            for token in str(label).replace("/", "+").replace("-", "_").split("+"):
                if token in gi:
                    F[i, gi[token]] = 1
    X = np.c_[F[:, [gi[x] for x in genes]], F @ pathways]
    ids = np.flatnonzero(labs != "CONTROL")
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(ids)
    ntr, nva = int(0.7 * len(ids)), int(0.15 * len(ids))
    tr = np.r_[ids[:ntr], ctrl]
    va = ids[ntr:ntr + nva]
    te = ids[ntr + nva:]
    return z, m, labs, y, X, tr, va, te


def run(seed, epochs=500, device_name="cuda"):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    z, metadata, labels, y, X, tr, va, te = load_data()
    mu = X[tr].mean(0)
    sd = X[tr].std(0) + 1e-3
    Xs = (X - mu) / sd
    ridge = Ridge(alpha=10, solver="cholesky").fit(Xs[tr], y[tr])
    base = ridge.predict(Xs)
    residual = y - base
    rs = residual[tr].std(0) + 1e-3
    tx = torch.tensor(Xs[tr], dtype=torch.float32, device=device)
    ty = torch.tensor(residual[tr] / rs, dtype=torch.float32, device=device)
    vx = torch.tensor(Xs[va], dtype=torch.float32, device=device)
    vy = torch.tensor(residual[va] / rs, dtype=torch.float32, device=device)
    net = Net(tx.shape[1], y.shape[1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    best, state, stall = float("inf"), None, 0
    curve = []
    for epoch in range(epochs):
        net.train(); opt.zero_grad()
        loss = ((net(tx) - ty) ** 2).mean(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            val_loss = float(((net(vx) - vy) ** 2).mean().cpu())
        curve.append({"epoch": epoch + 1, "train_mse": float(loss.detach().cpu()), "val_resid_mse": val_loss})
        if val_loss < best - 1e-7:
            best, stall = val_loss, 0
            state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            stall += 1
        if stall >= 60:
            break
    if state is not None:
        net.load_state_dict(state)
    net.eval()
    with torch.no_grad():
        correction = net(torch.tensor(Xs, dtype=torch.float32, device=device)).cpu().numpy() * rs
    pred = base + LOCKED_LAMBDA * correction
    split_hash = hashlib.sha256(np.asarray(np.r_[tr, va, te], dtype=np.int64).tobytes()).hexdigest()
    out = {
        "protocol": "locked_confirmation",
        "seed": seed,
        "device": str(device),
        "split_seed": SPLIT_SEED,
        "split_hash": split_hash,
        "locked_lambda": LOCKED_LAMBDA,
        "architecture": "Linear(768,64)-Tanh-Linear(64,64)-Tanh-Linear(64,512)",
        "optimizer": "AdamW(lr=1e-3,weight_decay=1e-2), validation early stop patience=60, max_epochs=500",
        "n_train_identities": int(len(tr) - len(np.flatnonzero(labels == "CONTROL"))),
        "n_validation_identities": int(len(va)),
        "n_test_identities": int(len(te)),
        "ridge_validation": metric(y[va], base[va]),
        "ridge_test": metric(y[te], base[te]),
        "residual_mlp_validation": metric(y[va], pred[va]),
        "residual_mlp_test": metric(y[te], pred[te]),
        "epochs_ran": len(curve),
        "best_validation_residual_mse": best,
        "input_hashes": {
            "script": sha256(Path(__file__)),
            "pseudobulk": sha256(DATA / "pseudobulk.npz"),
            "metadata": sha256(DATA / "metadata.json"),
        },
        "curve": curve,
    }
    path = OUT / f"residual_identity_mlp_seed{seed}_confirmation.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k not in {"curve", "input_hashes"}}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run(seed=args.seed, epochs=args.epochs, device_name=args.device)
