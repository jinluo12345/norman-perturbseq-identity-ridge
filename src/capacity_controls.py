"""Capacity and regularization controls for the fixed Norman confirmation split."""
from pathlib import Path
import json, random, sys
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / "data/processed"; OUT = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))
from train_models import PathwayNEAT, features  # noqa: E402


class LowResidualNEAT(nn.Module):
    """Pathway-NEAT with an explicit low-rank target-gene residual."""
    def __init__(self, n_path, n_out, path_rank=16, residual_rank=16):
        super().__init__(); self.path_u=nn.Parameter(torch.randn(n_path,path_rank)*.03); self.path_v=nn.Parameter(torch.randn(path_rank,n_out)*.03); self.path_gate=nn.Parameter(torch.zeros(n_path)); self.path_direct=nn.Parameter(torch.zeros(n_path,n_out)); self.intercept=nn.Parameter(torch.zeros(n_out)); self.gene_down=nn.Linear(n_out,residual_rank,bias=False); self.gene_up=nn.Linear(residual_rank,n_out,bias=False)
    def forward(self,Fg,Fp):
        path=(Fp*torch.nn.functional.softplus(self.path_gate))@self.path_u@self.path_v + Fp@self.path_direct
        return self.intercept + path + self.gene_up(self.gene_down(Fg)), [path]


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def metric(y, p):
    fy, fp = y.ravel(), p.ravel()
    return {"rmse": float(np.sqrt(mean_squared_error(y, p))),
            "pearson": float(pearsonr(fy, fp).statistic),
            "spearman": float(spearmanr(fy, fp).statistic), "n_profiles": int(len(y))}


def split(labels, seed=11):
    ids = np.array([i for i, l in enumerate(labels) if l != "CONTROL"]); rng = np.random.default_rng(seed); rng.shuffle(ids)
    ntr, nva = int(.70 * len(ids)), int(.15 * len(ids)); tr, va, te = ids[:ntr], ids[ntr:ntr+nva], ids[ntr+nva:]
    c = np.array([i for i, l in enumerate(labels) if l == "CONTROL"]); return np.concatenate([tr, c]), va, te


def fit_variant(name, cfg, Fg, Fp, y, tr, va, te, n_path, device, opt_seeds):
    out = {"name": name, "config": cfg, "runs": []}
    fpmean, fpstd = Fp[tr].mean(0), Fp[tr].std(0) + 1e-3; Fps = (Fp - fpmean) / fpstd
    ym, ysdev = y[tr].mean(0), y[tr].std(0) + 1e-3; ys = (y - ym) / ysdev
    for seed in opt_seeds:
        set_seed(seed)
        if cfg.get("residual_rank"):
            m = LowResidualNEAT(n_path, y.shape[1], path_rank=cfg["rank"], residual_rank=cfg["residual_rank"]).to(device)
        else:
            m = PathwayNEAT(n_path, y.shape[1], rank=cfg["rank"], use_path=True,
                            use_gene=cfg["use_gene"], use_direct=cfg["use_direct"]).to(device)
        if cfg["use_direct"]:
            init = Ridge(alpha=.1).fit(Fps[tr], ys[tr]); m.path_direct.data.copy_(torch.tensor(init.coef_.T, dtype=torch.float32, device=device))
        Xg = torch.tensor(Fg, dtype=torch.float32, device=device); Xp = torch.tensor(Fps, dtype=torch.float32, device=device); Y = torch.tensor(ys, dtype=torch.float32, device=device)
        opt = torch.optim.AdamW(m.parameters(), lr=7e-4, weight_decay=cfg["weight_decay"]); best=float('inf'); best_state=None; stall=0; hist=[]
        for ep in range(1000):
            m.train(); opt.zero_grad(); pred,_=m(Xg[tr],Xp[tr]); loss=((pred-Y[tr])**2).mean(); loss.backward(); opt.step(); m.eval()
            with torch.no_grad(): tl=((m(Xg[tr],Xp[tr])[0]-Y[tr])**2).mean().item(); vl=((m(Xg[va],Xp[va])[0]-Y[va])**2).mean().item()
            hist.append({"epoch":ep+1,"train_mse_std":float(tl),"val_mse_std":float(vl)})
            if vl<best: best=vl; stall=0; best_state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
            else: stall+=1
            if stall>=70: break
        if best_state: m.load_state_dict(best_state)
        m.eval();
        with torch.no_grad(): p=m(Xg[te],Xp[te])[0].detach().cpu().numpy()*ysdev+ym
        out["runs"].append({"optimization_seed":int(seed),"epochs_run":len(hist),"best_epoch":int(np.argmin([h['val_mse_std'] for h in hist])+1),"parameter_count":int(sum(q.numel() for q in m.parameters())),"metrics":metric(y[te],p),"learning_curve":hist})
    out["summary"]={k:{"mean":float(np.mean([r['metrics'][k] for r in out['runs']])),"sd":float(np.std([r['metrics'][k] for r in out['runs']],ddof=1))} for k in ['rmse','pearson','spearman']}
    return out


def main():
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    device='cuda'; z=np.load(DATA/'pseudobulk.npz',allow_pickle=True); meta=json.loads((DATA/'metadata.json').read_text()); labels=np.asarray(meta['labels']['norman']); tv=np.asarray(meta['pathway_vocab']); P=z['pathways_full'].astype(np.float32); y=z['y_norman'].astype(np.float32); c=np.flatnonzero(labels=='CONTROL'); y=y-y[c[0]]
    Fgfull,Fp=features(labels.tolist(),tv,P); out_idx=np.asarray([int(np.where(tv==g)[0][0]) for g in np.asarray(meta['genes'])]); Fg=Fgfull[:,out_idx]; tr,va,te=split(labels.tolist(),11)
    variants={
      'full_rank16_direct_gene':{'rank':16,'use_direct':True,'use_gene':True,'weight_decay':1e-3},
      'rank4_direct_gene':{'rank':4,'use_direct':True,'use_gene':True,'weight_decay':1e-3},
      'rank1_direct_gene':{'rank':1,'use_direct':True,'use_gene':True,'weight_decay':1e-3},
      'rank16_no_direct_gene':{'rank':16,'use_direct':False,'use_gene':True,'weight_decay':1e-3},
      'rank16_direct_no_gene':{'rank':16,'use_direct':True,'use_gene':False,'weight_decay':1e-3},
      'rank16_direct_lowres16':{'rank':16,'use_direct':True,'use_gene':True,'residual_rank':16,'weight_decay':1e-3},
      'rank16_direct_lowres64':{'rank':16,'use_direct':True,'use_gene':True,'residual_rank':64,'weight_decay':1e-3},
      'rank16_direct_gene_wd1e2':{'rank':16,'use_direct':True,'use_gene':True,'weight_decay':1e-2},
    }
    result={'device':device,'split_seed':11,'split':{'train':tr.tolist(),'validation':va.tolist(),'test':te.tolist(),'test_labels':labels[te].tolist()},'runs':[fit_variant(n,cfg,Fg,Fp,y,tr,va,te,P.shape[1],device,[11,22,33]) for n,cfg in variants.items()]}
    (OUT/'capacity_controls.json').write_text(json.dumps(result)); print(json.dumps({'runs':[(r['name'],r['summary'],r['runs'][0]['parameter_count']) for r in result['runs']]}))
    rows=[]
    for r in result['runs']:
        for q in r['runs']:
            rows.append({'variant':r['name'],'optimization_seed':q['optimization_seed'],'parameter_count':q['parameter_count'],'epochs_run':q['epochs_run'],'best_epoch':q['best_epoch'],**q['metrics']})
    pd.DataFrame(rows).to_csv(OUT/'capacity_controls.csv',index=False)


if __name__=='__main__': main()
