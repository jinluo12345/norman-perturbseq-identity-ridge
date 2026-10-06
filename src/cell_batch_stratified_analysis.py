"""Frozen-prediction error stratification for the cell-batch confirmation."""
from pathlib import Path
import hashlib, json, h5py
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from cell_batch_ridge_screen import ridge_predictions

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed'; RAW=ROOT/'data/raw/NormanWeissman2019_filtered.h5ad'

def met(y,p):
 return {'rmse':float(np.sqrt(mean_squared_error(y,p))), 'pearson':float(pearsonr(y.ravel(),p.ravel()).statistic), 'spearman':float(spearmanr(y.ravel(),p.ravel()).statistic), 'n_cells':int(len(y))}

def main():
 z=np.load(DATA/'cell_panel.npz',allow_pickle=True); cells=z['X'].astype(float); labels=np.asarray(z['labels'],str); source=z['source_row'].astype(int)
 with h5py.File(RAW,'r') as f: gem=np.asarray(f['obs']['gemgroup'][:],int)[source]
 m=json.loads((DATA/'metadata.json').read_text()); idlab=np.asarray(m['labels']['norman'],str); pp=np.load(DATA/'pseudobulk.npz',allow_pickle=True); P=pp['pathways_full'].astype(float); v=np.asarray(m['pathway_vocab']); genes=np.asarray(m['genes']); gi={g:i for i,g in enumerate(v)}; F=np.zeros((len(idlab),len(v)))
 for i,l in enumerate(idlab):
  if l!='CONTROL':
   for t in str(l).replace('/','+').replace('-','_').split('+'):
    if t in gi:F[i,gi[t]]=1
 active=np.flatnonzero(F.sum(0)>0); Ix=np.c_[F[:,[gi[g] for g in genes]],F@P,F[:,active]]; idmap={l:i for i,l in enumerate(idlab)}; cid=np.asarray([idmap[l] for l in labels]); target=cells-cells[labels=='CONTROL'].mean(0); batches=np.unique(gem); bmap={int(b):j for j,b in enumerate(batches)}; BO=np.zeros((len(gem),len(batches)))
 for i,b in enumerate(gem):BO[i,bmap[int(b)]]=1
 allrows=[]; val=json.loads((ROOT/'results/cell_batch_ridge_validation.json').read_text()); conf=json.loads((ROOT/'results/cell_batch_ridge_confirmation.json').read_text()); ids=np.flatnonzero(idlab!='CONTROL')
 for ex in val['seeds']:
  seed=int(ex['seed']); rng=np.random.default_rng(seed); order=ids.copy(); rng.shuffle(order); ntr,nva=int(.7*len(order)),int(.15*len(order)); train=set(idlab[order[:ntr]].tolist())|{'CONTROL'}; test=set(idlab[order[ntr+nva:]].tolist()); trainids=np.flatnonzero(np.isin(idlab,list(train))); testmask=np.isin(labels,list(test));
  by= np.asarray([target[cid==ii].mean(0) for ii in trainids]); xm,xs=Ix[trainids].mean(0),Ix[trainids].std(0)+1e-3; ba=float(ex['baseline_alpha']); bp=ridge_predictions((Ix[trainids]-xm)/xs,by,(Ix-xm)/xs,[ba])[ba][cid]
  gx,gy=[],[]
  for ii in trainids:
   for b in batches:
    q=(cid==ii)&(gem==b)
    if q.sum()>=2:gx.append(np.r_[Ix[ii],BO[np.flatnonzero(q)[0]]]);gy.append(target[q].mean(0))
  gx,gy=np.asarray(gx),np.asarray(gy); gm,gs=gx.mean(0),gx.std(0)+1e-3; ca=float(ex['batch_candidate_alpha']); cp=ridge_predictions((gx-gm)/gs,gy,(np.c_[Ix[cid],BO]-gm)/gs,[ca])[ca]
  strata={}
  for name,mask in [('all',testmask),('single',testmask & np.asarray(['+' not in l for l in labels])),('combination',testmask & np.asarray(['+' in l for l in labels]))]:
   if mask.sum(): strata[name]={'baseline':met(target[mask],bp[mask]),'candidate':met(target[mask],cp[mask])}
  batch_rows={}
  for b in batches:
   mask=testmask&(gem==b)
   if mask.sum():batch_rows[str(int(b))]={'baseline':met(target[mask],bp[mask]),'candidate':met(target[mask],cp[mask])}
  allrows.append({'seed':seed,'strata':strata,'batch':batch_rows,'test_identity_count':len(test)})
 out={'protocol':'frozen_cell_batch_error_stratification','selection':'all alphas and splits frozen from validation screen','rows':allrows,'input_hashes':conf['input_hashes']}; (ROOT/'results/cell_batch_stratified_analysis.json').write_text(json.dumps(out,indent=2)); print(json.dumps({'seeds':len(allrows),'strata':list(allrows[0]['strata'])},indent=2))
if __name__=='__main__':main()
