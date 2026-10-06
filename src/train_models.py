"""Train pathway-additive perturbation response models and baselines."""
from pathlib import Path
import json, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def target_tokens(label):
    if label in {"CONTROL", "NAN", "*"}: return []
    return [x for x in str(label).replace("/", "+").replace("-", "_").split("+") if x and x not in {"ONLY", "MOD"}]

def features(labels, target_vocab, P):
    gidx = {g:i for i,g in enumerate(target_vocab)}
    Fg = np.zeros((len(labels), len(target_vocab)), np.float32)
    for i,l in enumerate(labels):
        for tok in target_tokens(l):
            if tok in gidx: Fg[i,gidx[tok]] = 1
    # pathway membership is an explicit, identity-aligned transformation.
    Fp = Fg @ P
    return Fg, Fp

class PathwayNEAT(nn.Module):
    """Neural additive module decoder.

    Each pathway has an independent low-rank contribution. The prediction is
    the sum of pathway contributions plus a global intercept; no post-hoc
    attribution is required because each term is exposed in forward().
    """
    def __init__(self, n_path, n_out, rank=16, use_path=True, use_gene=True, use_direct=False):
        super().__init__(); self.use_path=use_path; self.use_gene=use_gene
        self.intercept=nn.Parameter(torch.zeros(n_out)); self.use_direct=use_direct
        self.path_direct=nn.Parameter(torch.zeros(n_path,n_out)) if use_direct else None
        self.path_u=nn.Parameter(torch.randn(n_path,rank)*0.03)
        self.path_v=nn.Parameter(torch.randn(rank,n_out)*0.03)
        self.path_gate=nn.Parameter(torch.zeros(n_path))
        self.gene=nn.Linear(n_out,n_out,bias=False) if use_gene else None
    def forward(self, Fg, Fp):
        terms=[]
        if self.use_path:
            # positive gates improve stability and make contribution magnitudes comparable.
            terms.append((Fp * torch.nn.functional.softplus(self.path_gate)) @ self.path_u @ self.path_v)
            if self.use_direct: terms.append(Fp @ self.path_direct)
        if self.use_gene:
            terms.append(self.gene(Fg))
        y=self.intercept
        if terms: y=y+sum(terms)
        return y, terms

class MLP(nn.Module):
    def __init__(self,n_in,n_out):
        super().__init__(); self.net=nn.Sequential(nn.Linear(n_in,128),nn.GELU(),nn.Dropout(0.1),nn.Linear(128,n_out))
    def forward(self,Fg,Fp): return self.net(Fp), []

def metrics(y,p):
    y=np.asarray(y); p=np.asarray(p); flat_y=y.ravel(); flat_p=p.ravel()
    pr=pearsonr(flat_y,flat_p).statistic if np.std(flat_p)>1e-8 else 0.
    sr=spearmanr(flat_y,flat_p).statistic if np.std(flat_p)>1e-8 else 0.
    return {"rmse":float(np.sqrt(mean_squared_error(y,p))),"pearson":float(pr),"spearman":float(sr),"n_profiles":int(len(y))}

def fit_torch(model, Xg, Xp, y, train_idx, val_idx, epochs=600, lr=2e-3, device="cpu"):
    model.to(device); Xg=torch.tensor(Xg,dtype=torch.float32,device=device); Xp=torch.tensor(Xp,dtype=torch.float32,device=device); y=torch.tensor(y,dtype=torch.float32,device=device)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-3); best=None; best_state=None; patience=70; stall=0
    for ep in range(epochs):
        model.train(); opt.zero_grad(); pred,_=model(Xg[train_idx],Xp[train_idx]); loss=((pred-y[train_idx])**2).mean(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): vp,_=model(Xg[val_idx],Xp[val_idx]); vl=((vp-y[val_idx])**2).mean().item()
        if best is None or vl<best: best=vl; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stall=0
        else: stall+=1
        if stall>=patience: break
    if best_state: model.load_state_dict(best_state)
    return model, ep+1

def pred(model,Fg,Fp,device="cpu"):
    model.eval();
    with torch.no_grad(): p,_=model(torch.tensor(Fg,dtype=torch.float32,device=device),torch.tensor(Fp,dtype=torch.float32,device=device))
    return p.detach().cpu().numpy()

def run(seed, device="cpu", split_seed=11):
    set_seed(seed)
    z=np.load(DATA/"pseudobulk.npz",allow_pickle=True); meta=json.loads((DATA/"metadata.json").read_text()); genes=np.array(meta["genes"]); target_vocab=np.array(meta["pathway_vocab"]); P=z["pathways_full"].astype(np.float32)
    labels=meta["labels"]["norman"]; y=z["y_norman"].astype(np.float32)
    # Predict perturbation response relative to the observed Norman control
    # profile, so metrics quantify biology rather than shared housekeeping signal.
    cidx=[i for i,l in enumerate(labels) if l=="CONTROL"]
    control=y[cidx].mean(0) if cidx else y.mean(0)
    y=y-control
    Fg_full,Fp=features(labels,target_vocab,P); out_idx=np.array([int(np.where(target_vocab==g)[0][0]) for g in genes]); Fg=Fg_full[:,out_idx]
    # identity-level split; no test profiles influence fitting or early stopping.
    # Freeze identity partitions across optimization seeds so all model
    # differences are paired on exactly the same confirmation labels.
    rng=np.random.default_rng(split_seed)
    # Controls define the centering reference and are always retained in the
    # training fit; only non-control perturbation identities are confirmation
    # units.  This prevents a zero-response control profile from inflating test
    # correlations or RMSE.
    idx=np.array([i for i,l in enumerate(labels) if l != "CONTROL"],dtype=int); rng.shuffle(idx)
    ntr=int(.70*len(idx)); nva=int(.15*len(idx)); tr,va,te=idx[:ntr],idx[ntr:ntr+nva],idx[ntr+nva:]
    cidx=np.array([i for i,l in enumerate(labels) if l == "CONTROL"],dtype=int)
    tr=np.concatenate([tr,cidx])
    results={"seed":seed,"split_seed":split_seed,"split":{"train":tr.tolist(),"val":va.tolist(),"test":te.tolist()},"models":{}}
    fpmean=Fp[tr].mean(0); fpstd=Fp[tr].std(0)+1e-3; Fp=(Fp-fpmean)/fpstd
    # Optimize on gene-wise standardized responses to prevent high-expression
    # housekeeping genes from dominating the neural objective.
    ymean=y[tr].mean(0); ystd=y[tr].std(0)+1e-3; ys=(y-ymean)/ystd
    mean_pred=np.repeat(y[tr].mean(0,keepdims=True),len(te),0); results["models"]["mean"]={**metrics(y[te],mean_pred),"trainable":False}
    ridge=Ridge(alpha=0.1).fit(Fp[tr],y[tr]); results["models"]["ridge_pathway"]={**metrics(y[te],ridge.predict(Fp[te])),"trainable":True,"alpha":0.1}
    # target-identity ridge is the conventional non-pathway additive baseline.
    rg=Ridge(alpha=10.0).fit(Fg[tr],y[tr]); results["models"]["ridge_gene"]={**metrics(y[te],rg.predict(Fg[te])),"trainable":True}
    mlp=MLP(Fp.shape[1],y.shape[1]); mlp,ep=fit_torch(mlp,Fg*0,Fp,ys,tr,va,device=device); results["models"]["mlp_pathway"]={**metrics(y[te],pred(mlp,Fg[te]*0,Fp[te],device)*ystd+ymean),"epochs":ep}
    main=PathwayNEAT(P.shape[1],y.shape[1],rank=16,use_path=True,use_gene=True,use_direct=True)
    # Warm start the additive decoder from a regularized pathway fit, then allow
    # the neural gates and low-rank residual to refine it on validation data.
    init_r=Ridge(alpha=0.1).fit(Fp[tr],ys[tr]); main.path_direct.data.copy_(torch.tensor(init_r.coef_.T,dtype=torch.float32))
    main,ep=fit_torch(main,Fg,Fp,ys,tr,va,epochs=1000,lr=7e-4,device=device); p=pred(main,Fg[te],Fp[te],device)*ystd+ymean; results["models"]["pathway_neat"]={**metrics(y[te],p),"epochs":ep}
    # Save the signed per-module decoder matrix for stability audits.  This is
    # a model output, not a post-hoc attribution map.
    with torch.no_grad():
        module_coef = (torch.nn.functional.softplus(main.path_gate)[:,None] * (main.path_u @ main.path_v))
        if main.use_direct:
            module_coef = module_coef + main.path_direct
    results["pathway_coefficients"] = module_coef.detach().cpu().numpy().astype(np.float32).tolist()
    # Key ablations: remove pathway modules and remove target-gene residual.
    for name,up,ug in [("ablate_no_pathway",False,True),("ablate_no_gene_residual",True,False)]:
        m=PathwayNEAT(P.shape[1],y.shape[1],rank=16,use_path=up,use_gene=ug); m,ep=fit_torch(m,Fg,Fp,ys,tr,va,epochs=1000,device=device); results["models"][name]={**metrics(y[te],pred(m,Fg[te],Fp[te],device)*ystd+ymean),"epochs":ep}
    # External validation on Dixit: gene identities are aligned by symbols and
    # the model is frozen; no external labels are used for selection.
    ld=meta["labels"]["dixit"]; yd=z["y_dixit"].astype(np.float32); dc=[i for i,l in enumerate(ld) if l=="CONTROL"]; yd=yd-(yd[dc].mean(0) if dc else yd.mean(0)); dg_full,dp=features(ld,target_vocab,P); dg=dg_full[:,out_idx]; keep=np.array([i for i,l in enumerate(ld) if l!="CONTROL"])
    ext={"n_profiles":int(len(keep))}
    ext["mean"] = metrics(yd[keep],np.repeat(y[tr].mean(0,keepdims=True),len(keep),0))
    ext["pathway_neat"] = metrics(yd[keep],pred(main,dg[keep],dp[keep],device)*ystd+ymean)
    ext["ridge_pathway"] = metrics(yd[keep],ridge.predict(dp[keep]))
    results["external_dixit"]=ext
    for dname in ["adamson_single","adamson_combo"]:
        if f"y_{dname}" not in z: continue
        la=meta["labels"][dname]; ya=z[f"y_{dname}"].astype(np.float32); ac=[i for i,l in enumerate(la) if l=="CONTROL"]; ya=ya-(ya[ac].mean(0) if ac else ya.mean(0)); agf,ap=features(la,target_vocab,P); ag=agf[:,out_idx]; kk=np.array([i for i,l in enumerate(la) if l!="CONTROL"])
        results[f"external_{dname}"]={"n_profiles":int(len(kk)),"mean":metrics(ya[kk],np.repeat(y[tr].mean(0,keepdims=True),len(kk),0)),"pathway_neat":metrics(ya[kk],pred(main,ag[kk],ap[kk],device)*ystd+ymean),"ridge_pathway":metrics(ya[kk],ridge.predict(ap[kk]))}
    # contribution stability: bootstrap perturbation resampling of held-out profiles.
    contrib=[]
    main.eval();
    with torch.no_grad():
        _,terms=main(torch.tensor(Fg[te],dtype=torch.float32,device=device),torch.tensor(Fp[te],dtype=torch.float32,device=device))
    if terms:
        a=terms[0].detach().cpu().numpy()*ystd;
        for j in range(a.shape[0]): contrib.append(float(np.linalg.norm(a[j])))
    results["contribution_norms"] = contrib
    return results

def main():
    device="cuda" if torch.cuda.is_available() else "cpu"; allr=[]
    for seed in [11,22,33]: allr.append(run(seed,device))
    (OUT/"model_results.json").write_text(json.dumps({"device":device,"runs":allr},indent=2))
    print(json.dumps({"device":device,"seeds":len(allr),"example":allr[0]["models"]},indent=2))

if __name__=="__main__": main()
