"""Validation-only cross-dataset transfer audit for Norman identity response prediction."""
from pathlib import Path
import hashlib, json
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
OUT = ROOT / "results"

def metric(y,p):
    return {"rmse":float(np.sqrt(mean_squared_error(y,p))),"pearson":float(pearsonr(y.ravel(),p.ravel()).statistic),"spearman":float(spearmanr(y.ravel(),p.ravel()).statistic),"n_identities":int(len(y))}

def tokens(label):
    return [x for x in str(label).replace('/','+').replace('-','_').replace('(MOD)','').split('+') if x and x not in {'ONLY','MOD'}]

def feature(labels,vocab,genes,P,active):
    gi={x:i for i,x in enumerate(vocab)}
    F=np.zeros((len(labels),len(vocab)),np.float32)
    for i,l in enumerate(labels):
        if l!='CONTROL':
            for t in tokens(l):
                if t in gi:F[i,gi[t]]=1
    return np.c_[F[:,[gi[x] for x in genes]],F@P,F[:,active]],active

def center(y,labels):
    c=np.flatnonzero(np.asarray(labels)=="CONTROL")
    return y-y[c].mean(0)

def run():
    z=np.load(DATA/'pseudobulk.npz',allow_pickle=True);m=json.loads((DATA/'metadata.json').read_text())
    vocab=np.asarray(m['pathway_vocab']); genes=np.asarray(m['genes']); P=z['pathways_full'].astype(np.float32)
    datasets=['norman','adamson_single','adamson_combo','dixit']
    allx={}; ally={}; active_counts={}
    # Use one shared perturbation-component vocabulary across datasets.
    all_tokens=set()
    for ds in datasets:
        for label in m['labels'][ds]:
            if label!='CONTROL': all_tokens.update(tokens(label))
    active=np.asarray(sorted({int(np.where(vocab==t)[0][0]) for t in all_tokens if t in set(vocab)}),dtype=int)
    for ds in datasets:
        labs=np.asarray(m['labels'][ds],str); allx[ds],act=feature(labs,vocab,genes,P,active); ally[ds]=center(z['y_'+ds].astype(np.float32),labs); active_counts[ds]=int(len(act))
    labs=np.asarray(m['labels']['norman'],str); ids=np.flatnonzero(labs!='CONTROL'); ctrl=np.flatnonzero(labs=='CONTROL'); rng=np.random.default_rng(11);rng.shuffle(ids);ntr,nva=int(.7*len(ids)),int(.15*len(ids));tr=ids[:ntr];va=ids[ntr:ntr+nva];te=ids[ntr+nva:]
    # Scale using Norman train + controls; external rows are transformed with identical statistics.
    Xn=allx['norman']; train_ref=np.r_[tr,ctrl]; mu=Xn[train_ref].mean(0); sd=Xn[train_ref].std(0)+1e-3
    Xns=(Xn-mu)/sd
    Xext={ds:(allx[ds]-mu)/sd for ds in datasets if ds!='norman'}
    rows=[]
    # external weighting is selected only on Norman validation.
    for w in [0.0,0.05,0.1,0.25,0.5,1.0,2.0,4.0]:
        for alpha in [1,3,10,30,100]:
            xs=[Xns[train_ref]]; ys=[ally['norman'][train_ref]]; ws=[np.ones(len(train_ref))]
            for ds in ['adamson_single','adamson_combo','dixit']:
                xs.append(Xext[ds]); ys.append(ally[ds]); ws.append(np.full(len(ally[ds]),w))
            X=np.vstack(xs); Y=np.vstack(ys); weights=np.concatenate(ws)
            model=Ridge(alpha=alpha,solver='cholesky').fit(X,Y,sample_weight=weights)
            rows.append({'external_weight':w,'alpha':alpha,'validation':metric(ally['norman'][va],model.predict(Xns[va]))})
    best=min(rows,key=lambda r:r['validation']['rmse'])
    w=best['external_weight']; alpha=best['alpha']
    xs=[Xns[train_ref]];ys=[ally['norman'][train_ref]];ws=[np.ones(len(train_ref))]
    for ds in ['adamson_single','adamson_combo','dixit']:
        xs.append(Xext[ds]);ys.append(ally[ds]);ws.append(np.full(len(ally[ds]),w))
    model=Ridge(alpha=alpha,solver='cholesky').fit(np.vstack(xs),np.vstack(ys),sample_weight=np.concatenate(ws))
    baseline=Ridge(alpha=30,solver='cholesky').fit(Xns[train_ref],ally['norman'][train_ref])
    out={'protocol':'cross_dataset_transfer_validation_then_locked_test','split_seed':11,'feature_definition':'512-gene panel + Reactome pathway scores + active perturbation-component indicators','datasets':datasets,'external_weight_grid':[0.0,0.05,0.1,0.25,0.5,1.0,2.0,4.0],'alpha_grid':[1,3,10,30,100],'selected_by_validation':best,'baseline_identity_augmented_ridge':{'validation':metric(ally['norman'][va],baseline.predict(Xns[va])),'test':metric(ally['norman'][te],baseline.predict(Xns[te]))},'transfer_model':{'external_weight':w,'alpha':alpha,'validation':metric(ally['norman'][va],model.predict(Xns[va])),'test':metric(ally['norman'][te],model.predict(Xns[te]))},'n_external_rows':{ds:int(len(ally[ds])) for ds in datasets if ds!='norman'},'active_features':active_counts,'split_hash':hashlib.sha256(np.asarray(np.r_[train_ref,va,te],dtype=np.int64).tobytes()).hexdigest(),'input_hashes':{'pseudobulk':hashlib.sha256((DATA/'pseudobulk.npz').read_bytes()).hexdigest(),'metadata':hashlib.sha256((DATA/'metadata.json').read_bytes()).hexdigest()},'grid_results':rows}
    (OUT/'cross_dataset_transfer_eval.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':run()
