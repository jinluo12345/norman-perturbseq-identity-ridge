"""Locked repaired-bundle evaluation of linear input-contract baselines."""
from pathlib import Path
import json, csv
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed'; OUT=ROOT/'results'
def metrics(y,p):
    return {'rmse':float(np.sqrt(mean_squared_error(y,p))), 'pearson':float(pearsonr(y.ravel(),p.ravel()).statistic), 'spearman':float(spearmanr(y.ravel(),p.ravel()).statistic), 'n_profiles':int(len(y))}
def scale(X,tr):
    mu=X[tr].mean(0); sd=X[tr].std(0)+1e-3; return (X-mu)/sd,mu,sd
def main():
    z=np.load(DATA/'pseudobulk.npz',allow_pickle=True); m=json.loads((DATA/'metadata.json').read_text())
    labs=np.asarray(m['labels']['norman']); y=z['y_norman'].astype(np.float64); P=z['pathways_full'].astype(np.float64); v=np.asarray(m['pathway_vocab']); genes=np.asarray(m['genes']); gi={g:i for i,g in enumerate(v)}
    F=np.zeros((len(labs),len(v)),float)
    for i,l in enumerate(labs):
        if l=='CONTROL': continue
        for t in str(l).replace('/','+').replace('-','_').split('+'):
            if t in gi:F[i,gi[t]]=1
    Fg=F[:,[gi[g] for g in genes]]; Fp=F@P; ctrl=np.flatnonzero(labs=='CONTROL'); y-=y[ctrl].mean(0)
    W=np.load(DATA/'pathway_scores_control_weighted.npz',allow_pickle=True)['scores'].astype(float)
    ids=np.flatnonzero(labs!='CONTROL'); rng=np.random.default_rng(11); rng.shuffle(ids); ntr=int(.70*len(ids)); nva=int(.15*len(ids)); tr=np.r_[ids[:ntr],ctrl]; va=ids[ntr:ntr+nva]; te=ids[ntr+nva:]
    candidates={'pathway_ridge':Fp,'gene_ridge':Fg,'joint_ridge':np.c_[Fg,Fp],'weighted_pathway_ridge':W,'weighted_joint_ridge':np.c_[Fg,W]}; rows=[]; full={}
    for name,X in candidates.items():
        Xs,mu,sd=scale(X,tr); best=None
        for alpha in [0.01,.03,.1,.3,1,3,10,30,100,300,1000]:
            model=Ridge(alpha=alpha).fit(Xs[tr],y[tr]); vm=metrics(y[va],model.predict(Xs[va]))
            if best is None or vm['rmse']<best[0]: best=(vm['rmse'],alpha,vm)
        alpha=best[1]; model=Ridge(alpha=alpha).fit(Xs[tr],y[tr]); tm=metrics(y[te],model.predict(Xs[te])); vm=metrics(y[va],model.predict(Xs[va]))
        row={'model':name,'alpha':float(alpha),'train_n':int(len(tr)),'validation_n':int(len(va)),'test_n':int(len(te)),**{f'val_{k}':v for k,v in vm.items() if k!='n_profiles'},**{f'test_{k}':v for k,v in tm.items() if k!='n_profiles'}}; rows.append(row); full[name]={'row':row,'test_predictions':model.predict(Xs[te]).tolist()}
    with (OUT/'repaired_baseline_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    full['split']={'seed':11,'train_indices':tr.tolist(),'validation_indices':va.tolist(),'test_indices':te.tolist(),'control_indices':ctrl.tolist()}; full['bundle']='data/processed/pseudobulk.npz and pathway_scores_control_weighted.npz'; (OUT/'repaired_baseline_eval.json').write_text(json.dumps(full,indent=2)); print(json.dumps(rows,indent=2))
if __name__=='__main__': main()
