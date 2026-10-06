"""Validation-only RBF kernel decoder screen on identity-augmented features."""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.stats import pearsonr,spearmanr
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed'; OUT=ROOT/'results'
def met(y,p):
 a,b=np.asarray(y).ravel(),np.asarray(p).ravel(); return {'rmse':float(np.sqrt(mean_squared_error(y,p))),'pearson':float(pearsonr(a,b).statistic),'spearman':float(spearmanr(a,b).statistic),'n_identities':int(len(y))}
def main():
 z=np.load(DATA/'pseudobulk.npz',allow_pickle=True); m=json.load(open(DATA/'metadata.json')); labs=np.array(m['labels']['norman']); y=z['y_norman'].astype(float); y-=y[labs=='CONTROL'].mean(0)
 v=np.array(m['pathway_vocab']); genes=np.array(m['genes']); P=z['pathways_full'].astype(float); gi={g:i for i,g in enumerate(v)}
 F=np.zeros((len(labs),len(v)))
 for i,l in enumerate(labs):
  if l!='CONTROL':
   for t in l.replace('/','+').replace('-','_').split('+'):
    if t in gi:F[i,gi[t]]=1
 active=np.flatnonzero(F.sum(0)>0); X=np.c_[F[:,[gi[g] for g in genes]],F@P,F[:,active]]
 rows=[]
 for seed in [11,22,33,44,55]:
  ids=np.flatnonzero(labs!='CONTROL'); rng=np.random.default_rng(seed); rng.shuffle(ids); ntr=int(.7*len(ids)); nva=int(.15*len(ids)); tr=np.r_[ids[:ntr],np.flatnonzero(labs=='CONTROL')]; va=ids[ntr:ntr+nva]; te=ids[ntr+nva:]
  mu=X[tr].mean(0); sd=X[tr].std(0)+1e-3; Xs=(X-mu)/sd
  bg=[]
  for a in [1,3,10,30,100]:
   f=Ridge(alpha=a,solver='lsqr',tol=1e-6).fit(Xs[tr],y[tr]); bg.append((a,met(y[va],f.predict(Xs[va]))))
  ba,bm=min(bg,key=lambda q:q[1]['rmse']); cand=[]
  # gamma uses squared Euclidean distance of standardized 870-D identity features.
  for gamma in [1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1]:
   for a in [0.1,1,3,10,30,100]:
    f=KernelRidge(alpha=a,kernel='rbf',gamma=gamma).fit(Xs[tr],y[tr]); cand.append((gamma,a,met(y[va],f.predict(Xs[va]))))
  winners=[q for q in cand if q[2]['rmse']<bm['rmse'] and q[2]['pearson']>bm['pearson'] and q[2]['spearman']>bm['spearman']]; bc=min(cand,key=lambda q:q[2]['rmse'])
  rows.append({'seed':seed,'split_hash':hashlib.sha256(np.asarray(np.r_[tr,va,te],dtype=np.int64).tobytes()).hexdigest(),'baseline_alpha':ba,'baseline_validation':bm,'best_rbf':{'gamma':bc[0],'alpha':bc[1],'validation':bc[2]},'all_metric_winners':len(winners)})
  print(seed,bm,bc[2],len(winners),flush=True)
 out={'protocol':'validation_only_rbf_kernel_identity_augmented_screen','candidate':'RBF KernelRidge on standardized 870-dimensional identity-augmented features','baseline':'identity-augmented Ridge','selection':'gamma and alpha validation-only; locked test identities not scored','seeds':rows,'all_metric_winner_each_split':all(r['all_metric_winners']>0 for r in rows),'input_hashes':{'pseudobulk':hashlib.sha256((DATA/'pseudobulk.npz').read_bytes()).hexdigest(),'metadata':hashlib.sha256((DATA/'metadata.json').read_bytes()).hexdigest()}}
 (OUT/'kernel_identity_validation.json').write_text(json.dumps(out,indent=2)); print(json.dumps({'all_metric_winner_each_split':out['all_metric_winner_each_split'],'winner_counts':[r['all_metric_winners'] for r in rows]},indent=2))
if __name__=='__main__':main()
