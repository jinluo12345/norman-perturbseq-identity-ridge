"""Extract the deterministic cell-level 512-gene output panel offline."""
from pathlib import Path
import json, hashlib
import h5py
import numpy as np
from recovery_continuous_scores import source_arrays, sampled_rows, parse_label

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed'; RAW=ROOT/'data/raw/NormanWeissman2019_filtered.h5ad'
def main():
    shape, indptr, raw_labels, raw_genes, lib = source_arrays()
    labels=np.asarray([parse_label(x) for x in raw_labels],dtype=object); keep=sampled_rows(labels)
    meta=json.loads((DATA/'metadata.json').read_text()); panel=np.asarray(meta['genes'],dtype=object)
    raw_clean=np.asarray([str(x).upper().strip() for x in raw_genes],dtype=object); col={g:i for i,g in enumerate(raw_clean)}
    cols=np.asarray([col[str(g).upper().strip()] for g in panel],dtype=np.int64)
    local=np.full(shape[0],-1,dtype=np.int64); local[keep]=np.arange(len(keep))
    X=np.zeros((len(keep),len(panel)),dtype=np.float32)
    with h5py.File(RAW,'r') as h:
        data=h['X/data']; indices=h['X/indices']
        for j,c in enumerate(cols):
            lo,hi=int(indptr[c]),int(indptr[c+1]); rr=np.asarray(indices[lo:hi],dtype=np.int64); vv=np.asarray(data[lo:hi],dtype=np.float64)
            loc=local[rr]; sel=(loc>=0)&(lib[rr]>0)
            if np.any(sel): X[loc[sel],j]=np.log1p((1e4*vv[sel]/lib[rr[sel]])).astype(np.float32)
    out=DATA/'cell_panel.npz'; np.savez_compressed(out,X=X,labels=labels[keep],source_row=keep,panel_genes=panel)
    print(json.dumps({'shape':list(X.shape),'control_cells':int((labels[keep]=='CONTROL').sum()),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'panel_hash':hashlib.sha256('\n'.join(map(str,panel)).encode()).hexdigest()},indent=2))
if __name__=='__main__': main()
