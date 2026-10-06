"""Exact re-ranking audit for the repaired identity-cluster Spearman interval.
Uses one locked seed and a small prespecified audit bootstrap to compare the
fixed-rank sensitivity approximation with exact re-ranking of each resample.
"""
from pathlib import Path
import json, hashlib
import numpy as np
import h5py
from scipy.stats import spearmanr, rankdata
import repaired_split_local_controls as base
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed'; RAW=ROOT/'data/raw/NormanWeissman2019_filtered.h5ad'
# A compact exact-reranking audit is used because each draw re-ranks the full
# 36-identity by 512-gene cell panel; production intervals remain the faster
# fixed-rank sensitivity estimate.  Every draw below still performs exact
# re-ranking for both estimators.
B=128; SEED=11

def ridge_grid(xtr,ytr,xq,alphas):
 out={}; xm=xtr.mean(0); ym=ytr.mean(0); xc=xtr-xm
 xtx=xc.T@xc; xty=xc.T@(ytr-ym); eye=np.eye(xtx.shape[0],dtype=xtx.dtype)
 for a in alphas:
  coef=np.linalg.solve(xtx+float(a)*eye,xty)
  out[float(a)]=(xq-xm)@coef+ym
 return out

def main():
 print('start',flush=True)
 z=np.load(DATA/'cell_panel.npz',allow_pickle=True); source=np.asarray(z['source_row'],np.int64); labels=np.asarray(z['labels'],str)
 with h5py.File(RAW,'r') as h: gem_all=np.asarray(h['obs/gemgroup'][:],dtype=np.int64)
 gem=gem_all[source]
 print('loaded',len(source),flush=True)
 meta=json.loads((DATA/'metadata.json').read_text()); pp=np.load(DATA/'pseudobulk.npz',allow_pickle=True); pathways=pp['pathways_full'].astype(float); ids_lab=np.asarray(meta['labels']['norman'],str)
 ids=np.flatnonzero(ids_lab!='CONTROL'); rng=np.random.default_rng(SEED); order=ids.copy(); rng.shuffle(order); ntr,nva=int(.7*len(order)),int(.15*len(order)); val=set(ids_lab[order[ntr:ntr+nva].tolist()]); test=set(ids_lab[order[ntr+nva:].tolist()]); tr=set(ids_lab[order[:ntr]].tolist())|{'CONTROL'}
 id_to_i={x:i for i,x in enumerate(ids_lab)}; cell_id=np.asarray([id_to_i[x] for x in labels]); train_ids=np.flatnonzero(np.isin(ids_lab,list(tr))); val_mask=np.isin(labels,list(val)); test_mask=np.isin(labels,list(test))
 panel=np.asarray(z['panel_genes'],dtype=object); X=np.asarray(z['X'],dtype=np.float32); ctrl=(labels=='CONTROL')&((source%2)==0); target=X-X[ctrl].mean(0,dtype=np.float64).astype(np.float32)
 ident,_,_=base.feature_matrix(labels,panel,meta,pathways); batches=np.unique(gem); bm={int(b):j for j,b in enumerate(batches)}; bo=np.zeros((len(gem),len(batches)))
 print('features',ident.shape,flush=True)
 for i,b in enumerate(gem): bo[i,bm[int(b)]]=1
 grx=[];gry=[];gids=[]; group_map={}
 for j,(ii,b) in enumerate(zip(cell_id,gem)): group_map.setdefault((int(ii),int(b)),[]).append(j)
 for (ii,b),idxs in group_map.items():
  if ii in set(train_ids.tolist()) and len(idxs)>=2:
   grx.append(np.r_[ident[ii],bo[idxs[0]]]); gry.append(target[idxs].mean(0)); gids.append((ii,b))
 print('groups',len(grx),flush=True)
 grx=np.asarray(grx); gry=np.asarray(gry); tx=grx; ty=gry; qtest=np.c_[ident[cell_id[test_mask]],bo[test_mask]]; mu,sd=tx.mean(0),tx.std(0)+1e-3; alpha=300.0; pred=ridge_grid((tx-mu)/sd,ty,(qtest-mu)/sd,[alpha])[alpha]
 print('candidate fit',grx.shape,flush=True)
 base_x=ident[train_ids]; base_y=np.asarray([target[cell_id==ii].mean(0) for ii in train_ids]); mu,sd=base_x.mean(0),base_x.std(0)+1e-3; ba=30.0; pred_b=ridge_grid((base_x-mu)/sd,base_y,(ident[cell_id[test_mask]]-mu)/sd,[ba])[ba]
 print('baseline fit',flush=True)
 tt=target[test_mask]; labs=labels[test_mask]; uniq=sorted(set(labs));
 # Exact reranking is audited on one profile per held-out identity.  This is
 # the natural cluster-level sensitivity unit and avoids presenting a costly
 # cell-level rerank as a calibrated interval.
 yg=[tt[labs==u].mean(0,keepdims=True) for u in uniq]; pg=[pred[labs==u].mean(0,keepdims=True) for u in uniq]; pbg=[pred_b[labs==u].mean(0,keepdims=True) for u in uniq]
 rng=np.random.default_rng(100000+SEED); exact=[]; fixed=[]
 # Fixed rank uses global locked-observation ranks then additive cluster sufficient stats.
 yflat=np.concatenate([g.ravel() for g in yg]); ry=rankdata(yflat,method='average'); rys=np.split(ry,np.cumsum([g.size for g in yg])[:-1]);
 for b in range(B):
  ix=rng.integers(0,len(uniq),size=len(uniq)); yy=np.concatenate([yg[i] for i in ix]).ravel(); pp=np.concatenate([pg[i] for i in ix]).ravel(); pbb=np.concatenate([pbg[i] for i in ix]).ravel(); exact.append(spearmanr(yy,pp).statistic-spearmanr(yy,pbb).statistic)
  # fixed ranks for this same draw use ranks from full locked observations, as in production approximation
  rr=np.concatenate([rys[i] for i in ix]); rp=np.concatenate([rankdata(pg[i].ravel(),method='average') for i in ix]); rpb=np.concatenate([rankdata(pbg[i].ravel(),method='average') for i in ix]);
  def corr(a,b):
   a=a.astype(float); b=b.astype(float); n=len(a); return (np.dot(a,b)-a.sum()*b.sum()/n)/np.sqrt(max((np.dot(a,a)-a.sum()**2/n)*(np.dot(b,b)-b.sum()**2/n),1e-30))
  fixed.append(corr(rr,rp)-corr(rr,rpb))
 out={'protocol':'exact_spearman_reranking_audit','seed':SEED,'B':B,'unit':'held-out identity cluster mean profile (one 512-gene profile per identity)','alpha_candidate':float(alpha),'alpha_baseline':float(ba),'exact_delta_mean':float(np.mean(exact)),'exact_ci95':np.quantile(exact,[.025,.975]).tolist(),'fixed_rank_delta_mean':float(np.mean(fixed)),'fixed_rank_ci95':np.quantile(fixed,[.025,.975]).tolist(),'direction_agreement':bool(np.sign(np.mean(exact))==np.sign(np.mean(fixed))),'input_panel_sha256':hashlib.sha256('\n'.join(map(str,panel)).encode()).hexdigest()}
 (ROOT/'results/spearman_exact_rerank_audit.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
