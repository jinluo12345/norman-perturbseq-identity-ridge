"""Repeated validation for weighted cross-dataset transfer."""
from pathlib import Path
import json, hashlib
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed'; OUT=ROOT/'results'
WEIGHTS=[0.0,0.025,0.05,0.1,0.25,0.5,1.0]; ALPHAS=[1,3,10,30,100]; SEEDS=[11,22,33,44,55]
def tok(l): return [x for x in str(l).replace('/','+').replace('-','_').replace('(MOD)','').split('+') if x and x not in {'ONLY','MOD'}]
def feat(labels,vocab,genes,P,active):
    gi={x:i for i,x in enumerate(vocab)};F=np.zeros((len(labels),len(vocab)),np.float32)
    for i,l in enumerate(labels):
        if l!='CONTROL':
            for t in tok(l):
                if t in gi:F[i,gi[t]]=1
    return np.c_[F[:,[gi[x] for x in genes]],F@P,F[:,active]]
def met(y,p): return {'rmse':float(np.sqrt(mean_squared_error(y,p))),'pearson':float(pearsonr(y.ravel(),p.ravel()).statistic),'spearman':float(spearmanr(y.ravel(),p.ravel()).statistic),'n_identities':int(len(y))}
def main():
    z=np.load(DATA/'pseudobulk.npz',allow_pickle=True);m=json.loads((DATA/'metadata.json').read_text()); vocab=np.asarray(m['pathway_vocab']);genes=np.asarray(m['genes']);P=z['pathways_full'].astype(np.float32);ds=['norman','adamson_single','adamson_combo','dixit']; all_tokens=set(t for d in ds for l in m['labels'][d] if l!='CONTROL' for t in tok(l)); vset=set(vocab);active=np.asarray(sorted({int(np.where(vocab==t)[0][0]) for t in all_tokens if t in vset}),int)
    X={};Y={}
    for d in ds:
        labs=np.asarray(m['labels'][d],str); X[d]=feat(labs,vocab,genes,P,active); c=np.flatnonzero(labs=='CONTROL');Y[d]=z['y_'+d].astype(np.float32)-z['y_'+d].astype(np.float32)[c].mean(0)
    labs=np.asarray(m['labels']['norman'],str);ids=np.flatnonzero(labs!='CONTROL');ctrl=np.flatnonzero(labs=='CONTROL');rows=[]
    for seed in SEEDS:
        rng=np.random.default_rng(seed);o=ids.copy();rng.shuffle(o);ntr,nva=int(.7*len(o)),int(.15*len(o));tr=o[:ntr];va=o[ntr:ntr+nva];ref=np.r_[tr,ctrl];mu=X['norman'][ref].mean(0);sd=X['norman'][ref].std(0)+1e-3;Xs={d:(X[d]-mu)/sd for d in ds}
        base=Ridge(alpha=30,solver='cholesky').fit(Xs['norman'][ref],Y['norman'][ref]); b=met(Y['norman'][va],base.predict(Xs['norman'][va])); cand=[]
        for w in WEIGHTS:
            for a in ALPHAS:
                xx=[Xs['norman'][ref]]; yy=[Y['norman'][ref]]; ww=[np.ones(len(ref))]
                for d in ds[1:]: xx.append(Xs[d]);yy.append(Y[d]);ww.append(np.full(len(Y[d]),w))
                q=Ridge(alpha=a,solver='cholesky').fit(np.vstack(xx),np.vstack(yy),sample_weight=np.concatenate(ww)); cand.append((met(Y['norman'][va],q.predict(Xs['norman'][va])),w,a))
        best=min(cand,key=lambda t:t[0]['rmse']); rows.append({'seed':seed,'baseline_validation':b,'transfer_validation':best[0],'selected_external_weight':best[1],'selected_alpha':best[2]})
    out={'protocol':'repeated_cross_dataset_transfer_validation','seeds':rows,'all_metrics_improved':all(r['transfer_validation']['rmse']<r['baseline_validation']['rmse'] and r['transfer_validation']['pearson']>r['baseline_validation']['pearson'] and r['transfer_validation']['spearman']>r['baseline_validation']['spearman'] for r in rows),'input_hashes':{'pseudobulk':hashlib.sha256((DATA/'pseudobulk.npz').read_bytes()).hexdigest(),'metadata':hashlib.sha256((DATA/'metadata.json').read_bytes()).hexdigest()}}
    (OUT/'cross_dataset_transfer_repeated.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__':main()
