"""Evaluate identity-held-out cell-level baselines without endpoint inputs."""
from pathlib import Path
import json, csv
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed'; OUT=ROOT/'results'
def metric(y,p):
    return {'rmse':float(np.sqrt(mean_squared_error(y,p))), 'pearson':float(pearsonr(y.ravel(),p.ravel()).statistic), 'spearman':float(spearmanr(y.ravel(),p.ravel()).statistic), 'n_cells':int(len(y))}
def main():
    z=np.load(DATA/'cell_panel.npz',allow_pickle=True); X=z['X'].astype(np.float64); labs=np.asarray(z['labels'],dtype=str); meta=json.loads((DATA/'metadata.json').read_text()); P=np.load(DATA/'pseudobulk.npz',allow_pickle=True)['pathways_full'].astype(float); vocab=np.asarray(meta['pathway_vocab']); genes=np.asarray(meta['genes']); gi={g:i for i,g in enumerate(vocab)}
    F=np.zeros((len(meta['labels']['norman']),len(vocab)),float); id_labs=np.asarray(meta['labels']['norman'],dtype=str)
    for i,l in enumerate(id_labs):
        if l!='CONTROL':
            for t in str(l).replace('/','+').replace('-','_').split('+'):
                if t in gi:F[i,gi[t]]=1
    Fg=F[:,[gi[g] for g in genes]]; Fp=F@P
    ids=np.flatnonzero(id_labs!='CONTROL'); rng=np.random.default_rng(11); rng.shuffle(ids); ntr=int(.70*len(ids)); nva=int(.15*len(ids)); tr_ids=ids[:ntr]; va_ids=ids[ntr:ntr+nva]; te_ids=ids[ntr+nva:]
    train_set=set(id_labs[tr_ids].tolist()+['CONTROL']); val_set=set(id_labs[va_ids]); test_set=set(id_labs[te_ids]);
    # Identity means are used for fitting so cell count imbalance cannot make
    # one perturbation dominate.  Cell-level scores are evaluated only after
    # the identity split is frozen.
    means={l:X[labs==l].mean(0) for l in np.unique(labs)}; counts={l:int((labs==l).sum()) for l in np.unique(labs)}
    ctrl=means['CONTROL']; Yid=np.vstack([means[l] for l in id_labs]); Yid-=ctrl
    ycell=X-ctrl
    rows=[]; full={}
    tr_idx=np.array([i for i,l in enumerate(id_labs) if l in train_set]); va_idx=np.array([i for i,l in enumerate(id_labs) if l in val_set]); te_idx=np.array([i for i,l in enumerate(id_labs) if l in test_set])
    for name,A in {'pathway':Fp,'gene':Fg,'joint':np.c_[Fg,Fp]}.items():
        mu=A[tr_idx].mean(0); sd=A[tr_idx].std(0)+1e-3; As=(A-mu)/sd
        best=None
        for alpha in [0.03,.1,.3,1,3,10,30,100,300]:
            mod=Ridge(alpha=alpha,solver='cholesky').fit(As[tr_idx],Yid[tr_idx]); pred=mod.predict(As[va_idx]); score=metric(Yid[va_idx],pred)
            if best is None or score['rmse']<best[0]: best=(score['rmse'],alpha)
        alpha=best[1]; mod=Ridge(alpha=alpha,solver='cholesky').fit(As[tr_idx],Yid[tr_idx]); pred_id=mod.predict(As); id_index={l:i for i,l in enumerate(id_labs)}; pred_cells=pred_id[np.asarray([id_index[l] for l in labs])]
        tm=metric(ycell[labs!='CONTROL'],pred_cells[labs!='CONTROL']); tm['test_identities']=int(len(te_ids)); tm['alpha']=float(alpha); tm['model']=name
        tm_test=metric(ycell[np.isin(labs,list(test_set))],pred_cells[np.isin(labs,list(test_set))]); tm_test.update({'test_identities':int(len(te_ids)),'alpha':float(alpha),'model':name}); rows.append(tm_test); full[name]={'test':tm_test,'test_predictions':pred_cells[np.isin(labs,list(test_set))].tolist()}
    # Distributional baseline: a global diagonal variance estimated from
    # training cells around their training identity means.
    train_cells=np.isin(labs,list(train_set)); resid=[]
    for l in train_set:
        if l in means: resid.append(X[labs==l]-means[l])
    resid=np.vstack(resid); var=np.maximum(resid.var(0),1e-4); pred=full['pathway']['test_predictions']; ytest=ycell[np.isin(labs,list(test_set))]; nll=float(0.5*np.mean(np.log(2*np.pi*var)+((ytest-np.asarray(pred))**2)/var)); full['pathway']['diagonal_gaussian_nll']=nll; full['diagonal_variance_summary']={'median':float(np.median(var)),'mean':float(np.mean(var)),'train_cells':int(train_cells.sum())}
    with (OUT/'cell_level_baseline_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    full['split']={'seed':11,'train_identities':sorted(train_set),'validation_identities':sorted(val_set),'test_identities':sorted(test_set)}; full['n_cells']=int(len(X)); full['output_panel_genes']=int(X.shape[1]); (OUT/'cell_level_baselines.json').write_text(json.dumps(full,indent=2)); print(json.dumps(rows,indent=2)); print(json.dumps({'gaussian_nll':nll,'variance':full['diagonal_variance_summary']},indent=2))
if __name__=='__main__': main()
