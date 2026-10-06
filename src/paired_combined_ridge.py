"""Paired identity bootstrap for validation-selected linear controls."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed'; OUT=ROOT/'results'
def toks(label):
    if label in {'CONTROL','NAN','*'}: return []
    return [x for x in str(label).replace('/','+').replace('-','_').split('+') if x and x not in {'ONLY','MOD'}]
def metric(y,p):
    return {'rmse':float(np.sqrt(mean_squared_error(y,p))), 'pearson':float(pearsonr(y.ravel(),p.ravel()).statistic)}
def boot(y,a,b,B=3000,seed=20261005):
    rng=np.random.default_rng(seed); ix=rng.integers(0,len(y),(B,len(y))); yy=y[ix]; aa=a[ix]; bb=b[ix]
    dr=np.sqrt(np.mean((yy-aa)**2,axis=(1,2)))-np.sqrt(np.mean((yy-bb)**2,axis=(1,2)))
    def pr(q):
        q=q.reshape(B,-1); z=yy.reshape(B,-1); q-=q.mean(1,keepdims=True); z-=z.mean(1,keepdims=True)
        return np.sum(q*z,1)/np.sqrt(np.sum(q*q,1)*np.sum(z*z,1))
    dp=pr(aa)-pr(bb)
    def ci(x): return {'estimate':float(np.mean(x)),'ci95':[float(np.quantile(x,.025)),float(np.quantile(x,.975))]}
    return {'rmse_neat_minus_linear':ci(dr),'pearson_neat_minus_linear':ci(dp),'n_bootstrap':B,'unit':'held-out perturbation identity'}
z=np.load(DATA/'pseudobulk.npz',allow_pickle=True); m=json.loads((DATA/'metadata.json').read_text()); labels=np.asarray(m['labels']['norman']); y=z['y_norman'].astype(float); P=z['pathways_full'].astype(float); vocab=np.asarray(m['pathway_vocab']); genes=np.asarray(m['genes'])
gi={g:i for i,g in enumerate(vocab)}; Ffull=np.zeros((len(labels),len(vocab)),float)
for i,l in enumerate(labels):
    for t in toks(l):
        if t in gi: Ffull[i,gi[t]]=1
Fp=Ffull@P; Fg=Ffull[:,np.asarray([gi[g] for g in genes])]; ctrl=np.flatnonzero(labels=='CONTROL'); y-=y[ctrl].mean(0)
ids=np.asarray([i for i,l in enumerate(labels) if l!='CONTROL']); rng=np.random.default_rng(11); rng.shuffle(ids); ntr,nva=int(.7*len(ids)),int(.15*len(ids)); tr=np.r_[ids[:ntr],ctrl]; va=ids[ntr:ntr+nva]; te=ids[ntr+nva:]
mu=Fp[tr].mean(0); sd=Fp[tr].std(0)+1e-3; Fs=(Fp-mu)/sd
alpha=3.0; pathway=Ridge(alpha=alpha).fit(Fs[tr],y[tr]); combined=Ridge(alpha=alpha).fit(np.c_[Fs,Fg][tr],y[tr]);
nd=json.loads((ROOT/'results/neat_diagnostics.json').read_text()); neat=np.mean([np.asarray(q['pred_neat']) for q in nd['runs']],axis=0); yy=np.asarray(nd['runs'][0]['y_test'])
out={'split_seed':11,'alpha':alpha,'test_n':len(te),'pathway_ridge':metric(y[te],pathway.predict(Fs[te])),'combined_ridge':metric(y[te],combined.predict(np.c_[Fs,Fg][te])),'pathway_ridge_test_predictions':pathway.predict(Fs[te]).tolist(),'combined_ridge_test_predictions':combined.predict(np.c_[Fs,Fg][te]).tolist(),'neat_test_predictions':neat.tolist(),'neat_metrics':metric(yy,neat),'paired_bootstrap_neat_minus_pathway':boot(yy,neat,pathway.predict(Fs[te])),'paired_bootstrap_neat_minus_combined':boot(yy,neat,combined.predict(np.c_[Fs,Fg][te]))}
(OUT/'combined_ridge_paired_bootstrap.json').write_text(json.dumps(out,indent=2)); print(json.dumps({k:out[k] for k in out if 'paired' in k or k.endswith('metrics')},indent=2))
