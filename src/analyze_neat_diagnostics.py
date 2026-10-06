"""Summarize saved GPU NEAT predictions with paired identity uncertainty."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
ROOT=Path(__file__).resolve().parents[1]

def ci(x): return {'estimate':float(np.mean(x)),'ci95':[float(np.quantile(x,.025)),float(np.quantile(x,.975))],'p_two_sided':float(2*min(np.mean(x<=0),np.mean(x>=0)))}
def paired(y,a,b,B=3000,seed=20261006):
    rng=np.random.default_rng(seed); n=len(y); ix=rng.integers(0,n,(B,n)); yy=y[ix]; aa=a[ix]; bb=b[ix]
    dr=np.sqrt(np.mean((yy-aa)**2,(1,2)))-np.sqrt(np.mean((yy-bb)**2,(1,2)))
    def pr(x):
      q=x.reshape(B,-1); t=yy.reshape(B,-1); q-=q.mean(1,keepdims=True); t-=t.mean(1,keepdims=True); return np.sum(q*t,1)/np.sqrt(np.sum(q*q,1)*np.sum(t*t,1))
    dp=pr(aa)-pr(bb); sb=min(2000,B); ds=np.array([spearmanr(yy[i].ravel(),aa[i].ravel()).statistic-spearmanr(yy[i].ravel(),bb[i].ravel()).statistic for i in range(sb)])
    return {'rmse_neat_minus_ridge':ci(dr),'pearson_neat_minus_ridge':ci(dp),'spearman_neat_minus_ridge':ci(ds),'n_bootstrap':B,'spearman_bootstrap':sb,'unit':'held-out perturbation identity'}
def main():
    x=json.load(open(ROOT/'results/neat_diagnostics.json')); r0=x['runs'][0]; y=np.array(r0['y_test']); rows=[]; 
    for run in x['runs']:
      yy=np.array(run['y_test']); a=np.array(run['pred_neat']); b=np.array(run['pred_ridge']);
      for i,lab in enumerate(x['split']['test_labels']):
        rows.append({'seed':run['seed'],'identity':lab,'rmse_neat':float(np.sqrt(np.mean((yy[i]-a[i])**2))),'rmse_ridge':float(np.sqrt(np.mean((yy[i]-b[i])**2))),'pearson_neat':float(pearsonr(yy[i],a[i]).statistic),'pearson_ridge':float(pearsonr(yy[i],b[i]).statistic),'delta_rmse':float(np.sqrt(np.mean((yy[i]-a[i])**2))-np.sqrt(np.mean((yy[i]-b[i])**2))),'delta_pearson':float(pearsonr(yy[i],a[i]).statistic-pearsonr(yy[i],b[i]).statistic)})
    pd.DataFrame(rows).to_csv(ROOT/'results/neat_profile_deltas.csv',index=False)
    curve=[]
    for run in x['runs']:
      h=pd.DataFrame(run['learning_curve']); curve.append({'seed':run['seed'],'epochs_run':run['epochs_run'],'best_epoch':run['best_epoch'],'parameter_count':run['parameter_count'],'train_mse_at_best':float(h.loc[h.val_mse_std.idxmin(),'train_mse_std']),'val_mse_at_best':float(h.val_mse_std.min()),'train_mse_final':float(h.train_mse_std.iloc[-1]),'val_mse_final':float(h.val_mse_std.iloc[-1]),'train_val_gap_final':float(h.val_mse_std.iloc[-1]-h.train_mse_std.iloc[-1])})
    pd.DataFrame(curve).to_csv(ROOT/'results/neat_training_diagnostics.csv',index=False)
    out={'paired_bootstrap_seed11':paired(y,np.array(r0['pred_neat']),np.array(r0['pred_ridge'])),'training_diagnostics':curve,'parameter_count':r0['parameter_count'],'test_labels':x['split']['test_labels']}
    (ROOT/'results/neat_paired_bootstrap.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
