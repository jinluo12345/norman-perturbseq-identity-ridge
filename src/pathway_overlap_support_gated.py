"""Support-gated pathway overlap interaction validation."""
from pathlib import Path
import json,hashlib,itertools
import numpy as np
from scipy.stats import pearsonr,spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/processed';OUT=ROOT/'results';SEEDS=[11,22,33,44,55];ALPHAS=[1,3,10,30,100]
def tok(l):return [x for x in str(l).replace('/','+').replace('-','_').split('+') if x and x not in {'ONLY','MOD'}]
def met(y,p):return {'rmse':float(np.sqrt(mean_squared_error(y,p))),'pearson':float(pearsonr(y.ravel(),p.ravel()).statistic),'spearman':float(spearmanr(y.ravel(),p.ravel()).statistic),'n_identities':int(len(y))}
def main():
 z=np.load(DATA/'pseudobulk.npz',allow_pickle=True);m=json.loads((DATA/'metadata.json').read_text());labs=np.asarray(m['labels']['norman'],str);y=z['y_norman'].astype(np.float32);ctrl=np.flatnonzero(labs=='CONTROL');y-=y[ctrl].mean(0);P=z['pathways_full'].astype(np.float32);v=np.asarray(m['pathway_vocab']);genes=np.asarray(m['genes']);gi={x:i for i,x in enumerate(v)};F=np.zeros((len(labs),len(v)),np.float32);pairs=[]
 for i,l in enumerate(labs):
  ts=[t for t in tok(l) if t in gi]; pairs.append(tuple(sorted(ts)))
  for t in ts:F[i,gi[t]]=1
 active=np.flatnonzero(F.sum(0)>0);A=F[:,[gi[x] for x in genes]];B=F@P;C=F[:,active];ids=np.flatnonzero(labs!='CONTROL');rows=[]
 for seed in SEEDS:
  rng=np.random.default_rng(seed);o=ids.copy();rng.shuffle(o);ntr,nva=int(.7*len(o)),int(.15*len(o));tr0=o[:ntr];tr=np.r_[tr0,ctrl];va=o[ntr:ntr+nva];te=o[ntr+nva:];X0=np.c_[A,B,C];mu=X0[tr].mean(0);sd=X0[tr].std(0)+1e-3;X0=(X0-mu)/sd
  # overlap vector only for pairs seen among training perturbations
  seen=set(p for i in tr0 for p in itertools.combinations(pairs[i],2)); H=np.zeros((len(labs),P.shape[1]),np.float32)
  for i,ts in enumerate(pairs):
   for a,b in itertools.combinations(ts,2):
    if (a,b) in seen:H[i]+=P[gi[a]]*P[gi[b]]
  X1=np.c_[A,B,C,H];mu1=X1[tr].mean(0);sd1=X1[tr].std(0)+1e-3;X1=(X1-mu1)/sd1
  base=Ridge(alpha=30,solver='cholesky').fit(X0[tr],y[tr]);b=met(y[va],base.predict(X0[va]));cand=[]
  for a in ALPHAS:
   q=Ridge(alpha=a,solver='cholesky').fit(X1[tr],y[tr]);cand.append((met(y[va],q.predict(X1[va])),a,met(y[te],q.predict(X1[te]))))
  best=min(cand,key=lambda t:t[0]['rmse']);rows.append({'seed':seed,'n_seen_pairs':len(seen),'baseline_validation':b,'gated_validation':best[0],'selected_alpha':best[1],'gated_test_exploratory':best[2]})
 out={'protocol':'support_gated_pathway_overlap_repeated_validation','seeds':rows,'all_metrics_improved':all(r['gated_validation']['rmse']<r['baseline_validation']['rmse'] and r['gated_validation']['pearson']>r['baseline_validation']['pearson'] and r['gated_validation']['spearman']>r['baseline_validation']['spearman'] for r in rows),'input_hashes':{'pseudobulk':hashlib.sha256((DATA/'pseudobulk.npz').read_bytes()).hexdigest(),'metadata':hashlib.sha256((DATA/'metadata.json').read_bytes()).hexdigest()}}
 (OUT/'pathway_overlap_support_gated.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
