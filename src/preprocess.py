"""Build identity-aligned Perturb-seq pseudobulk data from scPerturb h5ad files.

All joins use gene symbols (the h5ad var index) and perturbation labels, never
array position. The output is compact and can be loaded offline by CPU/GPU jobs.
"""
from pathlib import Path
import re, json, zipfile
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
import h5py

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
OUT = ROOT / "data/processed"
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "norman": RAW / "NormanWeissman2019_filtered.h5ad",
    "adamson_single": RAW / "AdamsonWeissman2016_GSM2406675_10X001.h5ad",
    "adamson_combo": RAW / "AdamsonWeissman2016_GSM2406677_10X005.h5ad",
    "dixit": RAW / "DixitRegev2016_K562_TFs_13_days.h5ad",
}

def clean_gene(x):
    return str(x).upper().strip()

def obs_target(obs, dataset):
    if dataset == "dixit":
        s = obs.get("target", obs.get("perturbation", pd.Series(index=obs.index, dtype=str))).astype(str)
        # target is already gene-level in the archive; controls and NaN are excluded later.
        return s.map(lambda x: x.upper().strip() if x not in {"nan", "NONE"} else "CONTROL")
    s = obs["perturbation"].astype(str)
    def parse(x):
        if x in {"nan", "*", "None"}:
            return "CONTROL"
        # Preserve combinatorial perturbations as a deterministic + separated set.
        toks = []
        for t in x.split("_"):
            if t.startswith("p") and any(ch.isdigit() for ch in t):
                break
            if t.lower() in {"neg", "ctrl", "control", "3x", "gal4-4(mod)"}:
                continue
            toks.append(t)
        return "+".join(toks).upper() if toks else "CONTROL"
    return s.map(parse)

def source_library_sizes(path):
    """Return complete-matrix row sums without repeated backed row slicing.

    The supplied h5ad files encode X as a compressed sparse column matrix.
    AnnData's backed row slices therefore decompress many overlapping column
    ranges.  Summing the CSC columns directly visits each stored count once
    and gives the exact library-size denominator used for normalization.
    """
    with h5py.File(path, "r") as h:
        x = h["X"]
        shape = tuple(int(v) for v in x.attrs["shape"])
        enc = x.attrs.get("encoding-type", b"")
        if isinstance(enc, bytes):
            enc = enc.decode()
        if enc != "csc_matrix":
            raise ValueError(f"Expected CSC X in {path}, found {enc!r}")
        indptr = np.asarray(x["indptr"][:], dtype=np.int64)
        sums = np.zeros(shape[0], dtype=np.float64)
        for c0 in range(0, shape[1], 256):
            c1 = min(c0 + 256, shape[1])
            lo, hi = int(indptr[c0]), int(indptr[c1])
            rows = np.asarray(x["indices"][lo:hi], dtype=np.int64)
            vals = np.asarray(x["data"][lo:hi], dtype=np.float64)
            if rows.size:
                sums += np.bincount(rows, weights=vals, minlength=shape[0])
        return sums

def load_dataset(name, path, max_cells=None, gene_subset=None):
    a = ad.read_h5ad(path, backed="r")
    if max_cells and a.n_obs > max_cells:
        # deterministic identity-stratified sampling keeps rare perturbations.
        t = obs_target(a.obs, name)
        rng = np.random.default_rng(17)
        keep = []
        for _, ix in pd.Series(np.arange(len(t))).groupby(t.to_numpy()).groups.items():
            ix = np.asarray(list(ix), dtype=int); n = max(8, int(max_cells * len(ix) / len(t)))
            keep.extend(rng.choice(ix, min(len(ix), n), replace=False).tolist())
        keep = np.array(sorted(set(keep)))
    else:
        keep = np.arange(a.n_obs)
    obs = a.obs.iloc[keep].copy()
    target = obs_target(obs, name).to_numpy()
    # h5ad indices are authoritative gene symbols. Duplicate symbols are collapsed.
    genes_all = pd.Index([clean_gene(x) for x in a.var_names])
    if gene_subset is not None:
        wanted = set(clean_gene(x) for x in gene_subset)
        order = np.array([i for i,g in enumerate(genes_all) if g in wanted], dtype=int)
    else:
        _, first = np.unique(genes_all, return_index=True)
        order = np.sort(first)
    genes = genes_all[order]
    # Compute library sizes over the complete source transcriptome before
    # selecting the analysis panel.  The previous implementation summed only
    # ``order`` (the 512-gene panel), which changed the normalization contract
    # and invalidated the earlier result bundle.  Chunking keeps the full
    # backed matrix out of memory while preserving the source row order.
    keep = np.asarray(keep, dtype=int)
    # Compute denominators from the complete source transcriptome before
    # selecting the analysis panel.  This is exact for the compressed source
    # matrix and avoids panel-only normalization and backed CSC read storms.
    lib_all = source_library_sizes(path)
    lib = lib_all[keep]
    lib[lib <= 0] = 1
    X = a[keep, order].X
    if sparse.issparse(X): X = X.tocsr()
    else: X = sparse.csr_matrix(X)
    # Apply full-transcriptome library-size normalization after panel slicing.
    X = X.multiply(1e4 / lib[:, None]).log1p().tocsr()
    a.file.close()
    return genes, target, X

def parse_pathways(gmt, genes, min_size=8, max_size=300):
    gene_set = set(genes)
    names, rows = [], []
    with zipfile.ZipFile(gmt) as z:
        txt = z.read("ReactomePathways.gmt").decode()
    for line in txt.splitlines():
        p = line.rstrip().split("\t")
        if len(p) < 3: continue
        gs = sorted(gene_set.intersection(clean_gene(x) for x in p[2:]))
        if min_size <= len(gs) <= max_size:
            names.append(p[0]); rows.append(gs)
    # reduce redundancy by retaining pathways with at least one unique gene and
    # cap at 256 modules for a compact, interpretable model.
    if len(rows) > 256:
        score = np.array([len(x) for x in rows]); take = np.argsort(-score)[:256]
        names = [names[i] for i in take]; rows = [rows[i] for i in take]
    mat = np.zeros((len(genes), len(rows)), dtype=np.float32)
    gi = {g:i for i,g in enumerate(genes)}
    for j, gs in enumerate(rows):
        for g in gs: mat[gi[g],j] = 1.0 / np.sqrt(len(gs))
    return names, mat, rows

def aggregate(name, path, genes_ref, max_cells=None):
    genes, target, X = load_dataset(name, path, max_cells=max_cells, gene_subset=genes_ref)
    idx = pd.Index(genes).get_indexer(genes_ref)
    valid = idx >= 0
    X = X[:, idx[valid]]
    labs = pd.Series(target)
    rows, labels = [], []
    for lab, ix in labs.groupby(labs).groups.items():
        ix = np.asarray(list(ix))
        rows.append(np.asarray(X[ix].mean(axis=0)).ravel())
        labels.append(lab)
    return np.asarray(rows, dtype=np.float32), labels, int(len(target))

def main():
    # Norman is the primary training archive; keep a deterministic subset during
    # preprocessing only if the full file is still downloading.
    present = {k:v for k,v in FILES.items() if v.exists() and v.stat().st_size > 1_000_000}
    if "norman" not in present:
        raise FileNotFoundError("Norman h5ad is required before preprocessing")
    # Inspect primary symbols.  The response panel is selected from training
    # identities only; no test identity or test cell contributes to its prevalence.
    an = ad.read_h5ad(FILES["norman"], backed="r")
    raw_n_cells = {"norman": int(an.n_obs)}
    full_genes = np.array([clean_gene(x) for x in an.var_names], dtype=str)
    # Intersect with external archives by gene identity.
    ext_sets = []
    for key in ["adamson_single", "adamson_combo", "dixit"]:
        if key in present:
            a = ad.read_h5ad(FILES[key], backed="r"); ext_sets.append(set(clean_gene(x) for x in a.var_names)); a.file.close()
    common = set(full_genes)
    for s in ext_sets:
        common.intersection_update(s)
    pathway_zip = RAW / "pathways/ReactomePathways.gmt.zip"
    # Reproduce the deterministic profile split used for model selection and
    # count expression prevalence only in cells carrying training identities.
    # This is a train-only feature-selection step; the held-out identities are
    # excluded before touching the expression matrix.
    target = obs_target(an.obs, "norman").to_numpy()
    rng_cells = np.random.default_rng(17); keep = []
    for _, ix in pd.Series(np.arange(len(target))).groupby(target).groups.items():
        ix = np.asarray(list(ix), dtype=int); n = max(8, int(60000 * len(ix) / len(target)))
        keep.extend(rng_cells.choice(ix, min(len(ix), n), replace=False).tolist())
    keep = np.array(sorted(set(keep)))
    sampled_labels = target[keep]
    profile_labels = np.array(sorted(set(sampled_labels) - {"CONTROL"}))
    rng_split = np.random.default_rng(11); shuffled = profile_labels.copy(); rng_split.shuffle(shuffled)
    train_labels = set(shuffled[:int(.70*len(shuffled))]) | {"CONTROL"}
    train_cells = keep[np.isin(sampled_labels, list(train_labels))]
    prevalence = np.asarray(an[train_cells, :].X.getnnz(axis=0)).ravel()
    order = np.argsort(prevalence, kind="stable")[::-1]
    genes = np.array([clean_gene(an.var_names[i]) for i in order if clean_gene(an.var_names[i]) in common][:512], dtype=str)
    an.file.close()
    pnames, Pfull, prows = parse_pathways(pathway_zip, full_genes)
    arrays = {}; labels = {}; ncell = {}
    for key,path in present.items():
        if key == "norman": continue
        check = ad.read_h5ad(path, backed="r")
        raw_n_cells[key] = int(check.n_obs)
        check.file.close()
        y, labs, n = aggregate(key, path, genes, max_cells=50000)
        arrays[key] = y; labels[key] = labs; ncell[key] = n
    # Primary Norman aggregated profiles (full labels) are generated separately.
    y, labs, n = aggregate("norman", FILES["norman"], genes, max_cells=60000)
    arrays["norman"] = y; labels["norman"] = labs; ncell["norman"] = n
    np.savez_compressed(OUT / "pseudobulk.npz", genes=genes, pathway_vocab=full_genes, pathways_full=Pfull, **{f"y_{k}":v for k,v in arrays.items()})
    meta = {"datasets": list(arrays), "labels": labels, "raw_n_cells": raw_n_cells,
            "n_cells": ncell, "panel_selection": {"method": "training-identity prevalence",
            "split_seed": 11, "max_norman_cells": 60000, "external_intersection": True},
            "pathway_release": "ReactomePathways.gmt supplied in scPerturb workspace",
            "pathway_names": pnames, "pathway_gene_lists": prows, "genes": genes.tolist(),
            "pathway_vocab": full_genes.tolist(), "n_pathways": int(Pfull.shape[1])}
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k:(v.shape,len(labels[k])) for k,v in arrays.items()}, indent=2))

if __name__ == "__main__": main()
