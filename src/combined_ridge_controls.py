"""Validation-selected additive pathway+identity ridge control-relative baseline."""
from pathlib import Path
import json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

ROOT=Path(__file__).resolve().parents[1]
z=np.load(ROOT/'data/processed/pseudobulk.npz',allow_pickle=True)
m=json.loads((ROOT/'data/processed/metadata.json').read_text())
P=z['pathways_full'].astype('float32'); tv=np.asarray(m['pathway_vocab']); labels=np.asarray(m['labels']['norman']); y=z['y_norman'].astype('float32'); genes=np.asarray(m['genes'])
def toks(l):
    return [] if l in {'CONTROL','NAN','*'} else [x for x in str(l).replace('/','+').replace('-','_').split('+') if x and x not in {'ONLY','MOD'}]
gi={g:i for i,g in enumerate(tv)}; Fg=np.zeros((len(labels),len(tv)),np.float32)
for i,l in enumerate(labels):
    for t in toks(l):
        if t in gi: Fg[i,gi[t]]=1
Fp=Fg@P; out=np.asarray([np.where(tv==g)[0][0] for g in genes]); Fg=Fg[:,out]
c=np.flatnonzero(labels=='CONTROL'); y=y-y[c].mean(0)
idx=np.asarray([i for i,l in enumerate(labels) if l!='CONTROL']); rng=np.random.default_rng(11); rng.shuffle(idx); ntr=int(.7*len(idx)); nva=int(.15*len(idx)); tr=np.r_[idx[:ntr],c]; va=idx[ntr:ntr+nva]; te=idx[ntr+nva:]
mu=Fp[tr].mean(0); sd=Fp[tr].std(0)+1e-3; Fps=(Fp-mu)/sd
def met(a,b):
    return {'rmse':float(np.sqrt(mean_squared_error(a,b))),'pearson':float(pearsonr(a.ravel(),b.ravel()).statistic),'spearman':float(spearmanr(a.ravel(),b.ravel()).statistic)}
rows=[]
for alpha in [0.01,.03,.1,.3,1,3,10,30,100,300,1000]:
    for mode,X in [('pathway',Fps),('combined',np.c_[Fps,Fg])]:
        q=Ridge(alpha=alpha).fit(X[tr],y[tr]); rows.append({'alpha':alpha,'mode':mode,'val':met(y[va],q.predict(X[va])),'test':met(y[te],q.predict(X[te]))})
best={}
for mode in ['pathway','combined']:
    best[mode]=min((r for r in rows if r['mode']==mode),key=lambda r:r['val']['rmse'])
out={'split_seed':11,'train_n':int(len(tr)),'validation_n':int(len(va)),'test_n':int(len(te)),'grid':rows,'best_by_validation_rmse':best}
(ROOT/'results/combined_ridge_control_relative.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
