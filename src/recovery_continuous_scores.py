"""Build target-independent, control-expression-weighted pathway scores.

The score for a perturbation is the pathway membership of its target genes
weighted by the mean normalized expression of those genes in the Norman
control cells.  Control expression is measured before any perturbed response
is used, so it is an input-side context score rather than a response-derived
feature.  All diagnostics below are restricted to the frozen training
identities and are used only to decide whether a nonlinear additive model is
identifiable.
"""
from pathlib import Path
import json, hashlib
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/NormanWeissman2019_filtered.h5ad"
DATA = ROOT / "data/processed"
OUT = ROOT / "results"

def clean(x):
    return str(x).upper().strip()

def parse_label(x):
    x = str(x)
    if x in {"nan", "*", "None"}:
        return "CONTROL"
    toks = []
    for t in x.split("_"):
        if t.startswith("p") and any(ch.isdigit() for ch in t):
            break
        if t.lower() in {"neg", "ctrl", "control", "3x", "gal4-4(mod)"}:
            continue
        toks.append(t)
    return "+".join(toks).upper() if toks else "CONTROL"

def source_arrays():
    with h5py.File(RAW, "r") as h:
        x = h["X"]
        shape = tuple(int(v) for v in x.attrs["shape"])
        indptr = np.asarray(x["indptr"][:], dtype=np.int64)
        rows = np.asarray(h["obs/perturbation/codes"][:], dtype=np.int32)
        cats = [c.decode() if isinstance(c, bytes) else str(c)
                for c in h["obs/perturbation/categories"][:]]
        labels_raw = np.asarray([cats[c] for c in rows], dtype=object)
        var = [v.decode() if isinstance(v, bytes) else str(v)
               for v in h["var/_index"][:]]
        lib = np.zeros(shape[0], dtype=np.float64)
        for c0 in range(0, shape[1], 256):
            c1 = min(c0 + 256, shape[1]); lo, hi = int(indptr[c0]), int(indptr[c1])
            rr = np.asarray(x["indices"][lo:hi], dtype=np.int64)
            vv = np.asarray(x["data"][lo:hi], dtype=np.float64)
            if rr.size:
                lib += np.bincount(rr, weights=vv, minlength=shape[0])
        return shape, indptr, labels_raw, np.asarray(var, dtype=object), lib

def sampled_rows(labels):
    rng = np.random.default_rng(17); keep = []
    for lab in sorted(set(labels)):
        ix = np.flatnonzero(labels == lab)
        n = max(8, int(60000 * len(ix) / len(labels)))
        keep.extend(rng.choice(ix, min(len(ix), n), replace=False).tolist())
    return np.asarray(sorted(set(keep)), dtype=np.int64)

def main():
    shape, indptr, raw_labels, raw_genes, lib = source_arrays()
    labels = np.asarray([parse_label(x) for x in raw_labels], dtype=object)
    keep = sampled_rows(labels)
    ctrl_rows = keep[labels[keep] == "CONTROL"]
    ctrl_mask = np.zeros(shape[0], dtype=bool); ctrl_mask[ctrl_rows] = True

    # Mean log-normalized expression over sampled control cells.  The sparse
    # CSC scan contributes only nonzero entries; unobserved genes have zero
    # expression and therefore need no explicit storage.
    baseline = np.zeros(shape[1], dtype=np.float64)
    with h5py.File(RAW, "r") as h:
        x = h["X"]
        for c0 in range(0, shape[1], 256):
            c1 = min(c0 + 256, shape[1]); lo, hi = int(indptr[c0]), int(indptr[c1])
            rr = np.asarray(x["indices"][lo:hi], dtype=np.int64)
            vv = np.asarray(x["data"][lo:hi], dtype=np.float64)
            if rr.size:
                sel = ctrl_mask[rr] & (lib[rr] > 0)
                if np.any(sel):
                    col = np.repeat(np.arange(c1-c0), np.diff(indptr[c0:c1+1]))
                    col = col[sel]; vals = np.log1p(1e4 * vv[sel] / lib[rr[sel]])
                    baseline[c0:c1] += np.bincount(col, weights=vals,
                                                    minlength=c1-c0)
    baseline /= max(1, len(ctrl_rows))

    m = json.loads((DATA / "metadata.json").read_text())
    z = np.load(DATA / "pseudobulk.npz", allow_pickle=True)
    vocab = np.asarray(m["pathway_vocab"], dtype=object)
    genes = np.asarray([clean(x) for x in raw_genes], dtype=object)
    order = {clean(g): i for i, g in enumerate(genes)}
    base_vocab = np.asarray([baseline[order[clean(g)]] for g in vocab], dtype=np.float64)
    P = z["pathways_full"].astype(np.float64)
    weighted_membership = P * base_vocab[:, None]
    vocab_idx = {clean(g): i for i, g in enumerate(vocab)}

    def identity_features(labs):
        F = np.zeros((len(labs), len(vocab)), dtype=np.float64)
        for i, lab in enumerate(labs):
            if lab == "CONTROL":
                continue
            for tok in str(lab).replace("/", "+").replace("-", "_").split("+"):
                j = vocab_idx.get(clean(tok))
                if j is not None: F[i, j] = 1.0
        return F @ weighted_membership

    all_labels = np.asarray(m["labels"]["norman"], dtype=object)
    scores = identity_features(all_labels)
    # The pathway score bundle is input-side and deterministic; keep all rows
    # for frozen evaluation, while support thresholds are computed only on the
    # predeclared training identities.
    np.savez_compressed(DATA / "pathway_scores_control_weighted.npz",
                        scores=scores.astype(np.float32), labels=all_labels,
                        pathway_names=np.asarray(m["pathway_names"], dtype=object),
                        baseline_expression=base_vocab.astype(np.float32))
    non = np.flatnonzero(all_labels != "CONTROL")
    rng = np.random.default_rng(11); rng.shuffle(non)
    ntr = int(.70 * len(non)); nva = int(.15 * len(non))
    train = np.r_[non[:ntr], np.flatnonzero(all_labels == "CONTROL")]
    val = non[ntr:ntr+nva]; train_val = np.r_[train, val]
    s = scores[train]
    uniq = np.array([len(np.unique(np.round(s[:,j], 10))) for j in range(s.shape[1])])
    qbins = np.array([len(np.unique(np.quantile(s[:,j], np.linspace(0,1,5)))) for j in range(s.shape[1])])
    rows, counts = np.unique(np.round(s, 10), axis=0, return_counts=True)
    out = {
        "input": "control-expression-weighted pathway score",
        "control_cells_sampled": int(len(ctrl_rows)),
        "n_pathways": int(s.shape[1]),
        "active_pathways_variance_gt_1e-10": int((s.var(0) > 1e-10).sum()),
        "active_fraction": float((s.var(0) > 1e-10).mean()),
        "unique_value_count_quantiles": [float(x) for x in np.quantile(uniq, [0,.25,.5,.75,1])],
        "pathways_with_at_least_four_quantile_bins": int((qbins >= 4).sum()),
        "fraction_with_at_least_four_quantile_bins": float((qbins >= 4).mean()),
        "feature_matrix_rank_train": int(np.linalg.matrix_rank(s - s.mean(0), tol=1e-8)),
        "unique_input_rows_train": int(len(rows)),
        "maximum_row_multiplicity_train": int(counts.max()),
        "n_train_rows_including_control": int(len(train)),
        "n_validation_rows": int(len(val)),
        "bundle_sha256": hashlib.sha256((DATA / "pathway_scores_control_weighted.npz").read_bytes()).hexdigest(),
        "training_identity_policy": "split seed 11; confirmation identities excluded",
    }
    (OUT / "recovery_continuous_support.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
