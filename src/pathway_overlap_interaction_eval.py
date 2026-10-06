"""Validation audit for pathway-overlap interactions in combinatorial perturbations."""
from pathlib import Path
import hashlib,json,itertools
import numpy as np
from scipy.stats import pearsonr,spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/processed';OUT=ROOT/'results'
def tok(l):return [x for x in str(l).replace('/','+').replace('-','_').split('+') if x and x not in {'ONLY','MOD'}]
def met(y,p):return {'rmse':float(np.sqrt(mean_squared_error(y,p))),'pearson':float(pearsonr(y.ravel(),p.ravel()).statistic),'spearman':float(spearmanr(y.ravel(),p.ravel()).statistic),'n_identities':int(len(y))}
def main():
 z=np.load(DATA/'pseudobulk.npz',allow_pickle=True);m=json.loads((DATA/'metadata.json').read_text());labs=np.asarray(m['labels']['norman'],str);y=z['y_norman'].astype(np.float32);ctrl=np.flatnonzero(labs=='CONTROL');y-=y[ctrl].mean(0);P=z['pathways_full'].astype(np.float32);v=np.asarray(m['pathway_vocab']);genes=np.asarray(m['genes']);gi={x:i for i,x in enumerate(v)};F=np.zeros((len(labs),len(v)),np.float32); H=np.zeros((len(labs),P.shape[1]),np.float32)
 for i,l in enumerate(labs):
  ts=[t for t in tok(l) if t in gi]
  for t in ts:F[i,gi[t]]=1
  for a,b in itertools.combinations(ts,2):H[i]+=P[gi[a]]*P[gi[b]]
 active=np.flatnonzero(F.sum(0)>0);A=F[:,[gi[x] for x in genes]];B=F@P;C=F[:,active];ids=np.flatnonzero(labs!='CONTROL');rng=np.random.default_rng(11);rng.shuffle(ids);ntr,nva=int(.7*len(ids)),int(.15*len(ids));tr=np.r_[ids[:ntr],ctrl];va=ids[ntr:ntr+nva];te=ids[ntr+nva:]
 rows=[]
 for name,X in [('augmented',np.c_[A,B,C]),('overlap',np.c_[A,B,C,H])]:
  mu=X[tr].mean(0);sd=X[tr].std(0)+1e-3;Xs=(X-mu)/sd
  for a in [1,3,10,30,100,300]:
   q=Ridge(alpha=a,solver='cholesky').fit(Xs[tr],y[tr]);rows.append({'model':name,'alpha':a,'validation':met(y[va],q.predict(Xs[va])),'test':met(y[te],q.predict(Xs[te]))})
 best=min([r for r in rows if r['model']=='overlap'],key=lambda r:r['validation']['rmse'])
 out={'protocol':'pathway_overlap_interaction_locked_split11','n_active_identity_features':int(len(active)),'n_overlap_features':int(H.shape[1]),'n_combo_rows':int(np.sum(H.sum(1)>0)),'selected_overlap':best,'augmented_reference':min([r for r in rows if r['model']=='augmented'],key=lambda r:r['validation']['rmse']),'grid_results':rows,'input_hashes':{'pseudobulk':hashlib.sha256((DATA/'pseudobulk.npz').read_bytes()).hexdigest(),'metadata':hashlib.sha256((DATA/'metadata.json').read_bytes()).hexdigest()}}
 (OUT/'pathway_overlap_interaction_eval.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
