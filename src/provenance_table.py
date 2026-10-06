"""Reconstruct per-archive and per-identity cell-count provenance."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import h5py
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results'; SRC=ROOT/'paper/source_data'
RAW=ROOT/'data/raw'
FILES={'norman':RAW/'NormanWeissman2019_filtered.h5ad',
       'adamson_single':RAW/'AdamsonWeissman2016_GSM2406675_10X001.h5ad',
       'adamson_combo':RAW/'AdamsonWeissman2016_GSM2406677_10X005.h5ad',
       'dixit':RAW/'DixitRegev2016_K562_TFs_13_days.h5ad'}
def _decode_obs_column(group, key):
    """Read a h5ad observation column without importing anndata/torch."""
    d = group[key]
    if isinstance(d, h5py.Group) and 'categories' in d and 'codes' in d:
        cats = d['categories'][...]
        cats = [x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x) for x in cats]
        codes = d['codes'][...]
        return np.asarray([cats[int(c)] if int(c) >= 0 else 'nan' for c in codes], dtype=object)
    vals = d[...]
    return np.asarray([x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x) for x in vals], dtype=object)
def _obs_target_h5(obs, dataset):
    if dataset == 'dixit':
        key = 'target' if 'target' in obs else 'perturbation'
        s = _decode_obs_column(obs, key)
        return np.asarray([str(x).upper().strip() if str(x).lower() not in {'nan','none'} else 'CONTROL' for x in s], dtype=object)
    s = _decode_obs_column(obs, 'perturbation')
    out=[]
    for x in s:
        x=str(x)
        if x in {'nan','*','None'}: out.append('CONTROL'); continue
        toks=[]
        for t in x.split('_'):
            if t.startswith('p') and any(ch.isdigit() for ch in t): break
            if t.lower() in {'neg','ctrl','control','3x','gal4-4(mod)'}: continue
            toks.append(t)
        out.append('+'.join(toks).upper() if toks else 'CONTROL')
    return np.asarray(out, dtype=object)
def selected_counts(name,path,max_cells):
    with h5py.File(path,'r') as h:
        labs=_obs_target_h5(h['obs'],name)
        raw=int(len(labs))
    if max_cells and raw>max_cells:
        rng=np.random.default_rng(17); keep=[]
        for _,ix in pd.Series(np.arange(len(labs))).groupby(labs).groups.items():
            ix=np.asarray(list(ix),dtype=int); n=max(8,int(max_cells*len(ix)/len(labs))); keep.extend(rng.choice(ix,min(len(ix),n),replace=False).tolist())
        keep=np.array(sorted(set(keep)))
    else: keep=np.arange(raw)
    vc=pd.Series(labs[keep]).value_counts().sort_index()
    rows=[{'dataset':name,'identity':str(k),'raw_cells':int((labs==k).sum()),'analyzed_cells':int(v),'is_control':bool(k=='CONTROL')} for k,v in vc.items()]
    return raw,int(len(keep)),rows
def main():
    rows=[]; summary={}
    for name,path in FILES.items():
        if not path.exists(): continue
        raw,an,rr=selected_counts(name,path,60000 if name=='norman' else 50000); rows.extend(rr); summary[name]={'raw_cells':raw,'analyzed_cells':an,'excluded_cells':raw-an,'control_cells':sum(r['analyzed_cells'] for r in rr if r['is_control']),'noncontrol_labels':sum(not r['is_control'] for r in rr),'profiles':len(rr)}
    pd.DataFrame(rows).to_csv(OUT/'per_identity_counts.csv',index=False); (OUT/'provenance_counts.json').write_text(json.dumps({'summary':summary,'rows':rows},indent=2)); (SRC/'per_identity_counts.csv').write_text(pd.DataFrame(rows).to_csv(index=False)); (SRC/'provenance_counts.json').write_text(json.dumps({'summary':summary,'rows':rows},indent=2)); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
