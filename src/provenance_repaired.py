"""Write machine-readable counts for the repaired Norman preprocessing rule."""
from pathlib import Path
import json, h5py, numpy as np

ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/'data/raw/NormanWeissman2019_filtered.h5ad'; OUT=ROOT/'results'
def parse(x):
    x=str(x)
    if x in {'nan','*','None'}: return 'CONTROL'
    toks=[]
    for t in x.split('_'):
        if t.startswith('p') and any(c.isdigit() for c in t): break
        if t.lower() in {'neg','ctrl','control','3x','gal4-4(mod)'}: continue
        toks.append(t)
    return '+'.join(toks).upper() if toks else 'CONTROL'
with h5py.File(RAW,'r') as h:
    g=h['obs/perturbation']; cats=[x.decode() if isinstance(x,bytes) else str(x) for x in g['categories'][:]]
    raw=np.asarray([cats[i] for i in g['codes'][:]],dtype=object); labels=np.asarray([parse(x) for x in raw],dtype=object)
rng=np.random.default_rng(17); keep=[]
for lab in sorted(set(labels)):
    ix=np.flatnonzero(labels==lab); n=max(8,int(60000*len(ix)/len(labels))); keep.extend(rng.choice(ix,min(len(ix),n),replace=False).tolist())
keep=np.asarray(sorted(set(keep)),dtype=int); analyzed=labels[keep]
rows=[]
for lab in sorted(set(labels)):
    rows.append({'identity':lab,'raw_cells':int((labels==lab).sum()),'sampled_cells':int((analyzed==lab).sum()),'is_control':bool(lab=='CONTROL')})
out={'dataset':'NormanWeissman2019_filtered','raw_cells':int(len(labels)),'sampled_cells':int(len(keep)),'raw_identities':int(len(set(labels))),'sampled_identities':int(len(set(analyzed))),'raw_control_cells':int((labels=='CONTROL').sum()),'sampled_control_cells':int((analyzed=='CONTROL').sum()),'sampling_seed':17,'allocation':'max(8, floor(60000*n_i/n_total)) with cap at n_i','rows':rows}
(OUT/'provenance_repaired.json').write_text(json.dumps(out,indent=2)); print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
