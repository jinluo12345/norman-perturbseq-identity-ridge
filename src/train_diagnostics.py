"""GPU-only diagnostics for Pathway-NEAT: paired predictions and learning curves."""
from pathlib import Path
import json, random
import sys
import numpy as np
import torch
from sklearn.linear_model import Ridge
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed'; OUT=ROOT/'results'
sys.path.insert(0,str(ROOT/'src'))
from train_models import PathwayNEAT, features, set_seed, metrics

def split(labels, seed=11):
    rng=np.random.default_rng(seed); idx=np.array([i for i,l in enumerate(labels) if l!='CONTROL']); rng.shuffle(idx)
    ntr=int(.70*len(idx)); nva=int(.15*len(idx)); te=idx[ntr+nva:]; tr=idx[:ntr]; va=idx[ntr:ntr+nva]; c=np.array([i for i,l in enumerate(labels) if l=='CONTROL']); return np.concatenate([tr,c]),va,te

def main():
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for this diagnostic job')
    device='cuda'; z=np.load(DATA/'pseudobulk.npz',allow_pickle=True); meta=json.loads((DATA/'metadata.json').read_text()); genes=np.array(meta['genes']); tv=np.array(meta['pathway_vocab']); P=z['pathways_full'].astype(np.float32); labels=meta['labels']['norman']; y=z['y_norman'].astype(np.float32); c=np.array([i for i,l in enumerate(labels) if l=='CONTROL']); y=y-y[c].mean(0); Fgfull,Fp=features(labels,tv,P); out_idx=np.array([int(np.where(tv==g)[0][0]) for g in genes]); Fg=Fgfull[:,out_idx]; tr,va,te=split(labels,11); fpmean=Fp[tr].mean(0); fpstd=Fp[tr].std(0)+1e-3; Fp=(Fp-fpmean)/fpstd; ym=y[tr].mean(0); ysdev=y[tr].std(0)+1e-3; ys=(y-ym)/ysdev
    result={'split':{'train':tr.tolist(),'val':va.tolist(),'test':te.tolist(),'test_labels':[labels[i] for i in te]},'runs':[]}
    for seed in [11,22,33]:
        set_seed(seed); model=PathwayNEAT(P.shape[1],y.shape[1],rank=16,use_path=True,use_gene=True,use_direct=True).to(device)
        init=Ridge(alpha=.1).fit(Fp[tr],ys[tr]); model.path_direct.data.copy_(torch.tensor(init.coef_.T,dtype=torch.float32,device=device))
        Xg=torch.tensor(Fg,dtype=torch.float32,device=device); Xp=torch.tensor(Fp,dtype=torch.float32,device=device); Y=torch.tensor(ys,dtype=torch.float32,device=device); opt=torch.optim.AdamW(model.parameters(),lr=7e-4,weight_decay=1e-3); best=float('inf'); best_state=None; stall=0; hist=[]; epochs=1000
        for ep in range(epochs):
            model.train(); opt.zero_grad(); pred,_=model(Xg[tr],Xp[tr]); loss=((pred-Y[tr])**2).mean(); loss.backward(); opt.step(); model.eval()
            with torch.no_grad(): tl=((model(Xg[tr],Xp[tr])[0]-Y[tr])**2).mean().item(); vl=((model(Xg[va],Xp[va])[0]-Y[va])**2).mean().item()
            hist.append({'epoch':ep+1,'train_mse_std':tl,'val_mse_std':vl})
            if vl<best: best=vl; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stall=0
            else: stall+=1
            if stall>=70: break
        if best_state: model.load_state_dict(best_state)
        model.eval();
        with torch.no_grad(): pred_test=model(Xg[te],Xp[te])[0].detach().cpu().numpy()*ysdev+ym
        ridge=Ridge(alpha=.1).fit(Fp[tr],y[tr]); ridge_test=ridge.predict(Fp[te])
        result['runs'].append({'seed':seed,'epochs_run':len(hist),'best_epoch':int(np.argmin([h['val_mse_std'] for h in hist])+1),'parameter_count':int(sum(p.numel() for p in model.parameters())),'test_metrics_neat':metrics(y[te],pred_test),'test_metrics_ridge':metrics(y[te],ridge_test),'y_test':y[te].tolist(),'pred_neat':pred_test.tolist(),'pred_ridge':ridge_test.tolist(),'learning_curve':hist})
    (OUT/'neat_diagnostics.json').write_text(json.dumps(result))
    print(json.dumps({'runs':[{'seed':r['seed'],'epochs_run':r['epochs_run'],'best_epoch':r['best_epoch'],'parameter_count':r['parameter_count'],'neat':r['test_metrics_neat'],'ridge':r['test_metrics_ridge']} for r in result['runs']]},indent=2))
if __name__=='__main__': main()
