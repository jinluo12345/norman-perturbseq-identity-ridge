"""Repeated validation for a fixed low-complexity pathway dosage expansion."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.stats import pearsonr,spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/processed';OUT=ROOT/'results';SEEDS=[11,22,33,44,55];ALPHAS=[1,3,10,30,100]
def tok(l):return [x for x in str(l).replace('/','+').replace('-','_').split('+') if x and x not in {'ONLY','MOD'}]
def met(y,p):return {'rmse':float(np.sqrt(mean_squared_error(y,p))),'pearson':float(pearsonr(y.ravel(),p.ravel()).statistic),'spearman':float(spearmanr(y.ravel(),p.ravel()).statistic),'n_identities':int(len(y))}
def main():
 z=np.load(DATA/'pseudobulk.npz',allow_pickle=True);m=json.loads((DATA/'metadata.json').read_text());labs=np.asarray(m['labels']['norman'],str);y=z['y_norman'].astype(np.float32);ctrl=np.flatnonzero(labs=='CONTROL');y-=y[ctrl].mean(0);P=z['pathways_full'].astype(np.float32);v=np.asarray(m['pathway_vocab']);genes=np.asarray(m['genes']);gi={x:i for i,x in enumerate(v)};F=np.zeros((len(labs),len(v)),np.float32)
 for i,l in enumerate(labs):
  for t in tok(l):
   if t in gi:F[i,gi[t]]=1
 active=np.flatnonzero(F.sum(0)>0);A=F[:,[gi[x] for x in genes]];B=F@P;C=F[:,active];D=B**2;ids=np.flatnonzero(labs!='CONTROL');rows=[]
 for seed in SEEDS:
  rng=np.random.default_rng(seed);o=ids.copy();rng.shuffle(o);ntr,nva=int(.7*len(o)),int(.15*len(o));tr=np.r_[o[:ntr],ctrl];va=o[ntr:ntr+nva];X0=np.c_[A,B,C];X1=np.c_[A,B,C,D];mu0=X0[tr].mean(0);sd0=X0[tr].std(0)+1e-3;mu1=X1[tr].mean(0);sd1=X1[tr].std(0)+1e-3;X0=(X0-mu0)/sd0;X1=(X1-mu1)/sd1;base=Ridge(alpha=30,solver='cholesky').fit(X0[tr],y[tr]);b=met(y[va],base.predict(X0[va]));cand=[]
  for a in ALPHAS:
   q=Ridge(alpha=a,solver='cholesky').fit(X1[tr],y[tr]);cand.append((met(y[va],q.predict(X1[va])),a,met(y[o[ntr+nva:]],q.predict(X1[o[ntr+nva:]]))))
  best=min(cand,key=lambda t:t[0]['rmse']);rows.append({'seed':seed,'baseline_validation':b,'nonlinear_validation':best[0],'selected_alpha':best[1],'nonlinear_test_exploratory':best[2]})
 out={'protocol':'repeated_pathway_dosage_square_validation','seeds':rows,'all_metrics_improved':all(r['nonlinear_validation']['rmse']<r['baseline_validation']['rmse'] and r['nonlinear_validation']['pearson']>r['baseline_validation']['pearson'] and r['nonlinear_validation']['spearman']>r['baseline_validation']['spearman'] for r in rows),'input_hashes':{'pseudobulk':hashlib.sha256((DATA/'pseudobulk.npz').read_bytes()).hexdigest(),'metadata':hashlib.sha256((DATA/'metadata.json').read_bytes()).hexdigest()}}
 (OUT/'pathway_dosage_nonlinear.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
