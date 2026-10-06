"""Validation-only heteroscedastic identity-augmented Ridge screen."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy.stats import pearsonr,spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed'; OUT=ROOT/'results'
def met(y,p):
 a,b=y.ravel(),p.ravel(); return {'rmse':float(np.sqrt(mean_squared_error(y,p))),'pearson':float(pearsonr(a,b).statistic),'spearman':float(spearmanr(a,b).statistic),'n_identities':len(y)}
def main():
 z=np.load(DATA/'pseudobulk.npz',allow_pickle=True); m=json.load(open(DATA/'metadata.json')); labs=np.array(m['labels']['norman']); y=z['y_norman'].astype(float); y-=y[labs=='CONTROL'].mean(0)
 v=np.array(m['pathway_vocab']); genes=np.array(m['genes']); P=z['pathways_full'].astype(float); gi={g:i for i,g in enumerate(v)}
 F=np.zeros((len(labs),len(v))); 
 for i,l in enumerate(labs):
  if l!='CONTROL':
   for t in l.replace('/','+').replace('-','_').split('+'):
    if t in gi:F[i,gi[t]]=1
 active=np.flatnonzero(F.sum(0)>0); X=np.c_[F[:,[gi[g] for g in genes]],F@P,F[:,active]]
 cnt=pd.read_csv(ROOT/'results/per_identity_counts.csv').query("dataset=='norman'").set_index('identity').loc[labs,'analyzed_cells'].to_numpy(float)
 rows=[]
 for seed in [11,22,33,44,55]:
  ids=np.flatnonzero(labs!='CONTROL'); rng=np.random.default_rng(seed); rng.shuffle(ids); ntr=int(.7*len(ids)); nva=int(.15*len(ids)); tr=np.r_[ids[:ntr],np.flatnonzero(labs=='CONTROL')]; va=ids[ntr:ntr+nva]; te=ids[ntr+nva:]
  mu=X[tr].mean(0); sd=X[tr].std(0)+1e-3; Xs=(X-mu)/sd
  base=[]; cand=[]
  for a in [1,3,10,30,100]:
   f=Ridge(alpha=a,solver='lsqr',tol=1e-6).fit(Xs[tr],y[tr]); base.append((a,f,met(y[va],f.predict(Xs[va]))))
   for gam in [.25,.5,.75,1.0]:
    w=np.power(np.clip(cnt[tr]/np.median(cnt[tr]),.1,10),gam); fw=Ridge(alpha=a,solver='lsqr',tol=1e-6).fit(Xs[tr],y[tr],sample_weight=w); cand.append((a,gam,fw,met(y[va],fw.predict(Xs[va]))))
  ba,bf,bm=min(base,key=lambda q:q[2]['rmse']); winners=[q for q in cand if q[3]['rmse']<bm['rmse'] and q[3]['pearson']>bm['pearson'] and q[3]['spearman']>bm['spearman']]; bc=min(cand,key=lambda q:q[3]['rmse'])
  rows.append({'seed':seed,'split_hash':hashlib.sha256(np.asarray(np.r_[tr,va,te],dtype=np.int64).tobytes()).hexdigest(),'baseline_alpha':ba,'baseline_validation':bm,'best_weighted':{'alpha':bc[0],'gamma':bc[1],'validation':bc[3]},'all_metric_winners':len(winners)})
  print(seed,bm,bc[3],len(winners),flush=True)
 out={'protocol':'validation_only_heteroscedastic_identity_augmented_ridge','candidate':'sample-weighted Ridge using analyzed cells per identity, weight exponent selected on validation','baseline':'identity-augmented Ridge','selection':'alpha and exponent validation-only; no test scored','seeds':rows,'all_metric_winner_each_split':all(r['all_metric_winners']>0 for r in rows),'input_hashes':{'pseudobulk':hashlib.sha256((DATA/'pseudobulk.npz').read_bytes()).hexdigest(),'metadata':hashlib.sha256((DATA/'metadata.json').read_bytes()).hexdigest(),'counts':hashlib.sha256((ROOT/'results/per_identity_counts.csv').read_bytes()).hexdigest()}}
 (OUT/'weighted_identity_ridge_validation.json').write_text(json.dumps(out,indent=2)); print(json.dumps({'all_metric_winner_each_split':out['all_metric_winner_each_split']},indent=2))
if __name__=='__main__':main()
