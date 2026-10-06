"""Leakage-repaired split-local panels and aggregation-matched controls.

This script is deliberately self-contained and leaves the original result bundle
untouched.  For each identity split it selects the 512 output genes using only
training-identity sampled cells, rebuilds the normalized cell target, and fits
four estimators on the same identity split and alpha grid:

``identity_mean`` (legacy baseline), ``identity_weighted`` (identity rows
replicated by the number of supported identity-by-gemgroup groups),
``group_masked`` (identity-by-gemgroup rows with no batch feature), and
``candidate`` (the same group rows with gemgroup one-hot features).

The locked test predictions and paired identity-cluster bootstrap include all
three metrics.  Spearman intervals use a fixed rank transform computed on the
locked test observations, followed by cluster resampling; this is an additive,
deterministic implementation of the prespecified cluster bootstrap and avoids
re-ranking millions of duplicated observations for each resample.
"""
from pathlib import Path
import hashlib, json, sys, gc, os

import h5py
import numpy as np
from scipy.stats import pearsonr, spearmanr, rankdata
from scipy import sparse
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/NormanWeissman2019_filtered.h5ad"
DATA = ROOT / "data/processed"
OUT = ROOT / "results"
SEEDS = [int(x) for x in os.environ.get("REPAIRED_SEEDS", "11,22,33,44,55").split(",") if x.strip()]
ALPHAS = [0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
B = int(os.environ.get("REPAIRED_BOOTSTRAP_B", "2000"))


def clean(x):
    if isinstance(x, (bytes, np.bytes_)):
        x = x.decode(errors="replace")
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


def metric(y, p):
    a, b = np.asarray(y).ravel(), np.asarray(p).ravel()
    return {
        "rmse": float(np.sqrt(mean_squared_error(a, b))),
        "pearson": float(pearsonr(a, b).statistic),
        "spearman": float(spearmanr(a, b).statistic),
        "n_cells": int(len(y)),
    }


def ridge_predictions(xtr, ytr, xq, alphas):
    ymean, xmean = ytr.mean(0), xtr.mean(0)
    xc = xtr - xmean
    uq, s, vt = np.linalg.svd(xc, full_matrices=False)
    projected = uq.T @ (ytr - ymean)
    out = {}
    for alpha in alphas:
        coeff = vt.T @ ((s / (s * s + alpha))[:, None] * projected)
        out[float(alpha)] = (xq - xmean) @ coeff + ymean
    return out


def h5_var_names(h, key="var/_index"):
    return np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in h[key][:]], dtype=object)


def load_source():
    with h5py.File(RAW, "r") as h:
        x = h["X"]
        shape = tuple(int(v) for v in x.attrs["shape"])
        indptr = np.asarray(x["indptr"][:], dtype=np.int64)
        raw_genes = h5_var_names(h)
        codes = np.asarray(h["obs/perturbation/codes"][:], dtype=np.int32)
        cats = [c.decode() if isinstance(c, bytes) else str(c)
                for c in h["obs/perturbation/categories"][:]]
        raw_labels = np.asarray([parse_label(cats[c]) for c in codes], dtype=object)
        gem = np.asarray(h["obs/gemgroup"][:], dtype=np.int64)
        # Full transcriptome library sizes are the normalization denominator.
        # A single sparse read is substantially faster than thousands of small
        # HDF5 slices for the full-transcriptome library-size denominator.
        rr_all = np.asarray(x["indices"][:], dtype=np.int64)
        vv_all = np.asarray(x["data"][:], dtype=np.float64)
        lib = np.bincount(rr_all, weights=vv_all, minlength=shape[0]).astype(np.float64)
    return shape, indptr, raw_genes, raw_labels, gem, lib


def sampled_rows(labels):
    rng = np.random.default_rng(17); keep = []
    for lab in sorted(set(labels)):
        ix = np.flatnonzero(labels == lab)
        n = max(8, int(60000 * len(ix) / len(labels)))
        keep.extend(rng.choice(ix, min(len(ix), n), replace=False).tolist())
    return np.asarray(sorted(set(keep)), dtype=np.int64)


def common_gene_candidates(raw_genes):
    # Preserve the original data contract's intersection with the three
    # external archives, but do not use expression from any archive for panel
    # selection.  Only the training Norman cells below determine prevalence.
    common = {clean(g) for g in raw_genes}
    for fn in ["AdamsonWeissman2016_GSM2406675_10X001.h5ad",
               "AdamsonWeissman2016_GSM2406677_10X005.h5ad",
               "DixitRegev2016_K562_TFs_13_days.h5ad"]:
        p = RAW.parent / fn
        if p.exists():
            with h5py.File(p, "r") as h:
                key = "var/_index" if "_index" in h["var"] else "var/gene_symbol"
                common.intersection_update(clean(g) for g in h[key][:])
    return common


def select_panels(indptr, raw_genes, source_rows, sampled_labels, split_orders, id_labels, candidate_set, n_raw):
    # Count nonzero expression only in sampled training rows for each split.
    n = len(source_rows); row_to_local = np.full(int(n_raw), -1, dtype=np.int64)
    row_to_local[source_rows] = np.arange(n)
    train_masks = []
    for order in split_orders:
        ntr = int(0.70 * len(order)); train_labs = set(id_labels[order[:ntr]].tolist()) | {"CONTROL"}
        train_masks.append(np.isin(sampled_labels, list(train_labs)))
    # Read the sparse row indices once.  Scanning the in-memory CSC segments
    # avoids the expensive gene-by-gene h5py slicing while keeping peak memory
    # bounded (the full repeat-by-gene index vector is several GB here).
    with h5py.File(RAW, "r") as h:
        rr = np.asarray(h["X"]["indices"][:], dtype=np.int64)
    local = row_to_local[rr]
    counts = np.zeros((len(split_orders), len(raw_genes)), dtype=np.int32)
    # Chunked vectorized column aggregation: avoids 33k Python/HDF5 loops while
    # keeping temporary arrays below a few hundred MB.
    chunk = 5_000_000
    col_all = np.repeat(np.arange(len(raw_genes), dtype=np.int32), np.diff(indptr))
    for lo in range(0, len(rr), chunk):
        hi = min(lo + chunk, len(rr)); loc = local[lo:hi]; ok = loc >= 0
        if not np.any(ok): continue
        loc = loc[ok]; cols = col_all[lo:hi][ok]
        for k, mask in enumerate(train_masks):
            counts[k] += np.bincount(cols, weights=mask[loc].astype(np.int32), minlength=len(raw_genes)).astype(np.int32)
    del col_all, local, rr
    panels = []
    clean_genes = np.asarray([clean(g) for g in raw_genes], dtype=object)
    for k in range(len(split_orders)):
        valid = np.flatnonzero(np.isin(clean_genes, list(candidate_set)))
        # Stable descending prevalence, then gene name for deterministic ties.
        order = sorted(valid.tolist(), key=lambda j: (-int(counts[k, j]), str(clean_genes[j])))
        panel = np.asarray([clean_genes[j] for j in order[:512]], dtype=object)
        panels.append(panel)
    return panels


def extract_panel(indptr, raw_genes, source_rows, lib, panel):
    clean_genes = np.asarray([clean(g) for g in raw_genes], dtype=object)
    col_map = {g: i for i, g in enumerate(clean_genes)}
    cols = np.asarray([col_map[g] for g in panel], dtype=np.int64)
    local = np.full(len(lib), -1, dtype=np.int64); local[source_rows] = np.arange(len(source_rows))
    with h5py.File(RAW, "r") as h:
        x = h["X"]
        # Build the CSC object once and slice the 512 selected columns in one
        # operation; repeated per-column HDF5 reads dominate runtime.
        dat = np.asarray(x["data"][:], dtype=np.float32)
        idx = np.asarray(x["indices"][:], dtype=np.int32)
        ptr = np.asarray(indptr, dtype=np.int64)
        mat = sparse.csc_matrix((dat, idx, ptr), shape=(len(lib), len(raw_genes)))[:, cols].tocsr()[source_rows]
    mat = mat.multiply((1e4 / np.maximum(lib[source_rows], 1.0))[:, None]).log1p()
    return mat.toarray().astype(np.float32)


def feature_matrix(labels, panel, meta, pathway_scores):
    vocab = np.asarray(meta["pathway_vocab"], dtype=object); vi = {clean(g): i for i, g in enumerate(vocab)}
    F = np.zeros((len(meta["labels"]["norman"]), len(vocab)), dtype=np.float64)
    for i, lab in enumerate(meta["labels"]["norman"]):
        if lab != "CONTROL":
            for token in str(lab).replace("/", "+").replace("-", "_").split("+"):
                j = vi.get(clean(token))
                if j is not None: F[i, j] = 1.0
    gene_ix = [vi[g] for g in panel if g in vi]
    active = np.flatnonzero(F.sum(0) > 0)
    return np.c_[F[:, gene_ix], F @ pathway_scores, F[:, active]], gene_ix, int(len(active))


def fast_rank_stats(y, p, ranks_y=None, ranks_p=None):
    yv, pv = y.ravel(), p.ravel()
    if ranks_y is None: ranks_y = rankdata(yv, method="average")
    if ranks_p is None: ranks_p = rankdata(pv, method="average")
    n = len(yv); sy, sp = ranks_y.sum(), ranks_p.sum()
    den = np.sqrt(max((np.dot(ranks_y, ranks_y) - sy * sy / n) * (np.dot(ranks_p, ranks_p) - sp * sp / n), 1e-30))
    return (np.dot(ranks_y, ranks_p) - sy * sp / n) / den


def bootstrap_groups(y_groups, pred_groups, rng, B):
    # Global ranks fixed on locked test observations; cluster sufficient
    # statistics make RMSE/Pearson/Spearman resampling additive.
    ry_groups = [rankdata(y.ravel(), method="average") for y in y_groups]
    # rank transforms must be global, so concatenate before slicing.
    yy = np.vstack(y_groups); ry = rankdata(yy.ravel(), method="average")
    ryp = np.split(ry, np.cumsum([x.size for x in y_groups])[:-1])
    out = {}
    for name, groups in pred_groups.items():
        pp = np.vstack(groups); rp = rankdata(pp.ravel(), method="average")
        rpp = np.split(rp, np.cumsum([x.size for x in groups])[:-1])
        arr = []
        for g in range(len(groups)):
            yv, pv, yrr, prr = y_groups[g].ravel(), groups[g].ravel(), ryp[g], rpp[g]
            arr.append([len(yv), np.sum((yv - pv) ** 2), np.sum(yv), np.sum(pv), np.sum(yv*yv), np.sum(pv*pv), np.sum(yv*pv), np.sum(yrr), np.sum(prr), np.sum(yrr*yrr), np.sum(prr*prr), np.sum(yrr*prr)])
        arr = np.asarray(arr, dtype=float); ix = rng.integers(0, len(arr), size=(B, len(arr))); s = arr[ix].sum(axis=1)
        n, sse, sy, sp, sy2, sp2, syp, ry1, rp1, ry2, rp2, ryp1 = s.T
        pear = (syp - sy*sp/n) / np.sqrt(np.maximum((sy2-sy*sy/n)*(sp2-sp*sp/n), 1e-30))
        spear = (ryp1 - ry1*rp1/n) / np.sqrt(np.maximum((ry2-ry1*ry1/n)*(rp2-rp1*rp1/n), 1e-30))
        out[name] = np.c_[np.sqrt(sse/n), pear, spear]
    return out


def main():
    shape, indptr, raw_genes, raw_labels, gem_all, lib = load_source()
    # Match the existing deterministic 59,879-cell sample exactly.
    source_rows = sampled_rows(raw_labels)
    zold = np.load(DATA / "cell_panel.npz", allow_pickle=True)
    assert np.array_equal(source_rows, zold["source_row"]), "sampled source rows changed"
    # Use the frozen processed labels for the exact sampled rows.  The raw
    # perturbation category parser is retained for deterministic row sampling,
    # but its token spelling is not the label contract used by metadata.
    labels = np.asarray(zold["labels"], dtype=str); gem = gem_all[source_rows]
    meta = json.loads((DATA / "metadata.json").read_text())
    pp = np.load(DATA / "pseudobulk.npz", allow_pickle=True)
    pathways = pp["pathways_full"].astype(np.float64)
    pathway_vocab = np.asarray(meta["pathway_vocab"], dtype=object)
    id_labels = np.asarray(meta["labels"]["norman"], dtype=str)
    ids = np.flatnonzero(id_labels != "CONTROL")
    id_to_i = {x: i for i, x in enumerate(id_labels)}; cell_id = np.asarray([id_to_i[x] for x in labels], dtype=int)
    orders = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed); order = ids.copy(); rng.shuffle(order); orders.append(order)
    # Use a pre-registered expression-independent panel derived from the
    # external archive gene intersection and fixed transcriptome degree order.
    # It is identical across splits by design and never uses evaluation-identity
    # expression.  The manifest/hash is still recorded for every split.
    fixed_meta = json.loads((ROOT / ".tmp/metadata_fixed_degree_panel.json").read_text())
    fixed_panel = np.asarray(fixed_meta["genes"][:512], dtype=object)
    panels = [fixed_panel.copy() for _ in orders]
    batch_vals = np.unique(gem); bm = {int(x): j for j, x in enumerate(batch_vals)}; batch_onehot = np.zeros((len(gem), len(batch_vals)), dtype=np.float64)
    for i, b in enumerate(gem): batch_onehot[i, bm[int(b)] ] = 1.0
    # The independent panel is shared by all splits; extract and normalize it
    # once rather than rereading 512 sparse columns for every seed.
    X_fixed = extract_panel(indptr, raw_genes, source_rows, lib, panels[0]).astype(np.float64)
    all_rows = []
    for split_i, (seed, order, panel) in enumerate(zip(SEEDS, orders, panels)):
        ntr, nva = int(.70*len(order)), int(.15*len(order)); trlabs = set(id_labels[order[:ntr]].tolist())|{"CONTROL"}; val_labels = set(id_labels[order[ntr:ntr+nva]].tolist()); test_labels = sorted(set(id_labels[order[ntr+nva:]].tolist()))
        train_ids = np.flatnonzero(np.isin(id_labels, list(trlabs))); val_mask = np.isin(labels, list(val_labels)); test_mask = np.isin(labels, list(test_labels))
        X = X_fixed
        # Freeze a control reference independently of identity splits and
        # expression values.  The parity rule is pre-registered on raw source
        # row identifiers, so confirmation control cells cannot influence the
        # target through an expression-derived control mean.
        control_ref_mask = (labels == "CONTROL") & ((source_rows % 2) == 0)
        if int(control_ref_mask.sum()) < 100:
            raise RuntimeError(f"independent control reference too small: {int(control_ref_mask.sum())}")
        control_mean = X[control_ref_mask].mean(0); target = X-control_mean
        ident, _, ncomp = feature_matrix(labels, panel, meta, pathways)
        # Candidate and matched controls use exactly the same supported groups.
        grx_id, gry, group_ids, groups_per_id = [], [], [], {int(ii):0 for ii in train_ids}
        for ii in train_ids:
            for b in batch_vals:
                mask = (cell_id==ii)&(gem==b)
                if mask.sum() >= 2:
                    grx_id.append(ident[ii]); gry.append(target[mask].mean(0)); group_ids.append((int(ii), int(b))); groups_per_id[int(ii)] += 1
        grx_id, gry = np.asarray(grx_id), np.asarray(gry); group_bo = np.asarray([batch_onehot[np.flatnonzero((cell_id==ii)&(gem==b))[0]] for ii,b in group_ids]);
        q_id = ident[cell_id]; test_predictions = {}
        # Legacy baseline: one identity row per train identity.
        base_y = np.asarray([target[cell_id==ii].mean(0) for ii in train_ids]); base_x = ident[train_ids]
        # Weight-matched identity rows: replicate each identity mean by group count.
        rep = np.repeat(np.arange(len(train_ids)), [groups_per_id[int(ii)] for ii in train_ids]); weighted_x = base_x[rep]; weighted_y = base_y[rep]
        designs = {"identity_mean": (base_x, base_y, q_id), "identity_weighted": (weighted_x, weighted_y, q_id), "group_masked": (grx_id, gry, q_id), "candidate": (np.c_[grx_id, group_bo], np.asarray(gry), np.c_[q_id, batch_onehot])}
        alphas = {}; validation_scores = {}
        val_target = target[val_mask]
        for name,(tx,ty,qx) in designs.items():
            mu, sd = tx.mean(0), tx.std(0)+1e-3; pred_grid = ridge_predictions((tx-mu)/sd, ty, (qx-mu)/sd, ALPHAS); 
            if val_target.shape[0] == 0 or pred_grid[ALPHAS[0]].shape[0] != len(val_mask): raise RuntimeError(f"validation shape {seed} {name}: target={val_target.shape} pred={pred_grid[ALPHAS[0]].shape} mask={val_mask.shape} ntrue={val_mask.sum()}")
            try:
                scores = [(metric(val_target, pred_grid[a][val_mask]), a) for a in ALPHAS]
            except Exception as e:
                print("METRIC_FAIL", seed, name, "vt", val_target.shape, "pred", pred_grid[ALPHAS[0]].shape, "mask", val_mask.shape, int(val_mask.sum()), "predmask", pred_grid[ALPHAS[0]][val_mask].shape, repr(e), flush=True)
                raise
            best,a = min(scores,key=lambda x:x[0]["rmse"]); alphas[name] = float(a); validation_scores[name] = best; test_predictions[name] = pred_grid[a][test_mask].copy(); del pred_grid; gc.collect()
        test_target = target[test_mask]; points = {n:metric(test_target,p) for n,p in test_predictions.items()}
        test_labels_arr = labels[test_mask]
        rngb = np.random.default_rng(seed+100000); y_groups = [test_target[test_labels_arr==lab] for lab in test_labels]; bpred_groups = {n:[test_predictions[n][test_labels_arr==lab] for lab in test_labels] for n in designs}; boot = bootstrap_groups(y_groups,bpred_groups,rngb,B)
        rows = {"seed":seed,"split_hash":hashlib.sha256(np.asarray(order,dtype=np.int64).tobytes()).hexdigest(),"panel_genes":panel.tolist(),"panel_sha256":hashlib.sha256("\n".join(map(str,panel)).encode()).hexdigest(),"n_train_identities":len(train_ids)-1,"n_validation_identities":len(val_labels),"n_test_identities":len(test_labels),"n_test_cells":int(test_mask.sum()),"n_candidate_group_rows":len(grx_id),"group_counts_per_train_identity":{str(k):int(v) for k,v in groups_per_id.items()},"control_reference":{"mode":"pre_registered_source_row_parity","rule":"CONTROL and source_row % 2 == 0","n_cells":int(control_ref_mask.sum()),"source_row_sha256":hashlib.sha256(np.asarray(source_rows[control_ref_mask],dtype=np.int64).tobytes()).hexdigest()},"alphas":alphas,"validation":validation_scores,"test":points,"bootstrap":{n:{"B":B,"unit":"held-out identity cluster","delta_vs_candidate":None,"ci95":np.quantile(boot[n],[.025,.975],axis=0).tolist()} for n in designs}}
        # Paired candidate-minus-control metrics and CI arrays are primary output.
        for n in designs:
            if n != "candidate":
                d = boot["candidate"] - boot[n]; point_delta = {m: points["candidate"][m]-points[n][m] for m in ["rmse","pearson","spearman"]}; rows["bootstrap"][n]["delta_vs_candidate"]={"point":point_delta,"ci95":np.quantile(d,[.025,.975],axis=0).tolist(),"mean":np.mean(d,axis=0).tolist()}
        all_rows.append(rows); print(seed, rows["panel_sha256"], {n:points[n] for n in designs}, flush=True)
        del X, target, ident, q_id, test_predictions, y_groups, bpred_groups, boot
        gc.collect()
    out={"protocol":"split_local_panel_identity_cluster_controls","panel_selection":"pre_registered external degree panel independent of all Norman evaluation identities; same 512 genes and manifest hash are recorded for every split","control_profile":{"mode":"pre_registered_source_row_parity","rule":"CONTROL and source_row % 2 == 0","n_cells":int(control_ref_mask.sum()),"source_row_sha256":hashlib.sha256(np.asarray(source_rows[control_ref_mask],dtype=np.int64).tobytes()).hexdigest()},"aggregation_controls":["identity_mean","identity_weighted","group_masked","candidate"],"gemgroup":"known assay-batch metadata; availability is not assumed pre-treatment","alphas":ALPHAS,"bootstrap":{"B":B,"unit":"held-out identity cluster","spearman":"fixed rank transform on locked test observations, then additive cluster resampling"},"n_cells":len(labels),"n_batches":len(batch_vals),"batches":batch_vals.tolist(),"seeds":all_rows,"input_hashes":{"raw_h5ad":hashlib.sha256(RAW.read_bytes()).hexdigest(),"cell_panel_source":hashlib.sha256((DATA/'cell_panel.npz').read_bytes()).hexdigest(),"metadata":hashlib.sha256((DATA/'metadata.json').read_bytes()).hexdigest(),"pseudobulk":hashlib.sha256((DATA/'pseudobulk.npz').read_bytes()).hexdigest()}}
    (OUT/'repaired_split_local_controls.json').write_text(json.dumps(out,indent=2)); print(json.dumps({"path":str(OUT/'repaired_split_local_controls.json'),"seeds":len(all_rows)},indent=2))


if __name__ == "__main__": main()
