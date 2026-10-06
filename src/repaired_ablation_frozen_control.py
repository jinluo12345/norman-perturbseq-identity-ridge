"""Validation-only ablations for the frozen identity-by-gemgroup Ridge."""
from pathlib import Path
import hashlib, json, h5py
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from cell_batch_ridge_screen import ridge_predictions

ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / "data/processed"; RAW = ROOT / "data/raw/NormanWeissman2019_filtered.h5ad"; ALPHAS = [1., 3., 10., 30., 100., 300.]


def metric(y, p):
    return {"rmse": float(np.sqrt(mean_squared_error(y, p))), "pearson": float(pearsonr(y.ravel(), p.ravel()).statistic), "spearman": float(spearmanr(y.ravel(), p.ravel()).statistic), "n_cells": int(len(y))}


def main():
    z = np.load(DATA / "cell_panel.npz", allow_pickle=True); cells = z["X"].astype(float); labels = np.asarray(z["labels"], str); source = z["source_row"].astype(int)
    with h5py.File(RAW, "r") as f: gem = np.asarray(f["obs"]["gemgroup"][:], int)[source]
    m = json.loads((DATA / "metadata.json").read_text()); ids_lab = np.asarray(m["labels"]["norman"], str); pp = np.load(DATA / "pseudobulk.npz", allow_pickle=True); P = pp["pathways_full"].astype(float); v = np.asarray(m["pathway_vocab"]); genes = np.asarray(m["genes"]); gi = {g:i for i,g in enumerate(v)}
    F = np.zeros((len(ids_lab), len(v)))
    for i,l in enumerate(ids_lab):
        if l != "CONTROL":
            for t in str(l).replace('/','+').replace('-','_').split('+'):
                if t in gi: F[i,gi[t]] = 1
    active = np.flatnonzero(F.sum(0)>0); xparts = {"gene":F[:,[gi[g] for g in genes]], "pathway":F@P, "component":F[:,active]}
    ids = {l:i for i,l in enumerate(ids_lab)}; cell_id = np.asarray([ids[l] for l in labels]); control_ref=(labels=='CONTROL') & ((source%2)==0); target = cells - cells[control_ref].mean(0); batches = np.unique(gem); bm = {int(b):j for j,b in enumerate(batches)}; bo = np.zeros((len(gem),len(batches)))
    for i,b in enumerate(gem): bo[i,bm[int(b)]]=1
    variants = {"full": ["gene","pathway","component"], "no_pathway": ["gene","component"], "no_gene": ["pathway","component"], "no_component": ["gene","pathway"]}
    rows=[]; id_all=np.flatnonzero(ids_lab!='CONTROL')
    for seed in [11,22,33,44,55]:
        rng=np.random.default_rng(seed); order=id_all.copy(); rng.shuffle(order); ntr,nva=int(.7*len(order)),int(.15*len(order)); train_ids=np.flatnonzero(np.isin(ids_lab, list(set(ids_lab[order[:ntr]].tolist())|{'CONTROL'}))); val_mask=np.isin(labels, ids_lab[order[ntr:ntr+nva]])
        for name,parts in variants.items():
            ix = np.cumsum([0,512,256,102])
            # Identity feature slices are reconstructed directly to avoid any
            # dependence on a fitted model or held-out response.
            ident = np.c_[xparts["gene"] if "gene" in parts else np.empty((len(ids_lab),0)), xparts["pathway"] if "pathway" in parts else np.empty((len(ids_lab),0)), xparts["component"] if "component" in parts else np.empty((len(ids_lab),0))]
            grx=[]; gry=[]
            for ii in train_ids:
                for b in batches:
                    mask=(cell_id==ii)&(gem==b)
                    if mask.sum()>=2: grx.append(np.r_[ident[ii],bo[np.flatnonzero(mask)[0]]]); gry.append(target[mask].mean(0))
            grx,gry=np.asarray(grx),np.asarray(gry); mu,sd=grx.mean(0),grx.std(0)+1e-3
            # Ablation selection uses validation cells only.  Querying the full
            # 59,879-cell panel for every variant/alpha created a large unused
            # matrix; restricting q to the predeclared validation mask preserves
            # the protocol and makes the CPU experiment reproducible.
            q=np.c_[ident[cell_id],bo][val_mask]
            pred_grid=ridge_predictions((grx-mu)/sd,gry,(q-mu)/sd,ALPHAS); scores=[]
            for a in ALPHAS: scores.append((metric(target[val_mask],pred_grid[a]),a))
            best,a=min(scores,key=lambda x:x[0]['rmse']); rows.append({'seed':seed,'variant':name,'features':parts,'alpha':a,'validation':best})
            print(seed,name,best,a,flush=True)
    out={'protocol':'validation_only_cell_batch_ridge_input_ablation','variants':variants,'selection':'alpha by validation cell RMSE only; no test targets loaded','rows':rows,'input_hashes':{'cell_panel':hashlib.sha256((DATA/'cell_panel.npz').read_bytes()).hexdigest(),'metadata':hashlib.sha256((DATA/'metadata.json').read_bytes()).hexdigest(),'raw_h5ad':hashlib.sha256(RAW.read_bytes()).hexdigest()}}
    (ROOT/'results/repaired_ablation_frozen_control.json').write_text(json.dumps(out,indent=2))


if __name__=='__main__': main()
