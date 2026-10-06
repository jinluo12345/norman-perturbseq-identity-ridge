"""Identity-level uncertainty, repeated splits and pathway-null diagnostics.

All calculations use the frozen processed Norman panel.  The script deliberately
keeps the test identities out of feature selection and reports the perturbation
identity as the statistical unit.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
OUT = ROOT / "results"

def target_tokens(label):
    if label in {"CONTROL", "NAN", "*"}: return []
    return [x for x in str(label).replace("/", "+").replace("-", "_").split("+") if x and x not in {"ONLY", "MOD"}]

def design(labels, vocab, P, out_vocab):
    gi = {g:i for i,g in enumerate(vocab)}
    F = np.zeros((len(labels), len(vocab)), np.float32)
    for i, lab in enumerate(labels):
        for tok in target_tokens(lab):
            if tok in gi: F[i, gi[tok]] = 1.0
    return F[:, np.array([gi[g] for g in out_vocab])] , F @ P

def metrics(y, p):
    flat_y, flat_p = np.asarray(y).ravel(), np.asarray(p).ravel()
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "pearson": float(pearsonr(flat_y, flat_p).statistic) if np.std(flat_p)>1e-10 else 0.0,
        "spearman": float(spearmanr(flat_y, flat_p).statistic) if np.std(flat_p)>1e-10 else 0.0,
        "n_profiles": int(len(y)),
    }

def split_ids(labels, seed):
    non = np.array([i for i,l in enumerate(labels) if l != "CONTROL"], dtype=int)
    rng = np.random.default_rng(seed); rng.shuffle(non)
    ntr, nva = int(.70*len(non)), int(.15*len(non))
    tr, va, te = non[:ntr], non[ntr:ntr+nva], non[ntr+nva:]
    ctrl = np.array([i for i,l in enumerate(labels) if l == "CONTROL"], dtype=int)
    return np.concatenate([tr, ctrl]), va, te

def bootstrap_deltas(y, pred_a, pred_b, B=3000, seed=20261005):
    """Paired identity bootstrap for metric A minus metric B."""
    rng=np.random.default_rng(seed); n=len(y); ix=rng.integers(0,n,(B,n))
    ya=y[ix]; pa=pred_a[ix]; pb=pred_b[ix]
    d_rmse=np.sqrt(np.mean((ya-pa)**2,axis=(1,2)))-np.sqrt(np.mean((ya-pb)**2,axis=(1,2)))
    # Vectorized Pearson over bootstrap matrices; this is equivalent to the
    # flattened profile-level statistic and avoids thousands of scipy calls.
    def vp(a):
        aa=a.reshape(B,-1); yy=ya.reshape(B,-1)
        aa=aa-aa.mean(1,keepdims=True); yy=yy-yy.mean(1,keepdims=True)
        return np.sum(aa*yy,1)/np.sqrt(np.sum(aa*aa,1)*np.sum(yy*yy,1))
    d_pr=vp(pa)-vp(pb)
    # Spearman is retained as an approximate paired rank statistic; exact
    # ranks are computed for a deterministic subset to bound CPU cost.
    sb=min(B,2000); d_sr=np.empty(sb)
    for b in range(sb):
        yy=ya[b].ravel(); d_sr[b]=spearmanr(yy,pa[b].ravel()).statistic-spearmanr(yy,pb[b].ravel()).statistic
    def ci(x): return {"estimate":float(np.mean(x)), "ci95":[float(np.quantile(x,.025)),float(np.quantile(x,.975))], "p_two_sided":float(2*min(np.mean(x<=0),np.mean(x>=0)))}
    return {"rmse_A_minus_B":ci(d_rmse),"pearson_A_minus_B":ci(d_pr),"spearman_A_minus_B":ci(d_sr),"n_bootstrap":B,"spearman_bootstrap":sb,"unit":"held-out perturbation identity"}

def main():
    z=np.load(DATA/"pseudobulk.npz",allow_pickle=True); meta=json.loads((DATA/"metadata.json").read_text())
    labels=np.array(meta["labels"]["norman"]); y=z["y_norman"].astype(np.float64); ctrl=np.where(labels=="CONTROL")[0]; y=y-y[ctrl].mean(0)
    genes=np.array(meta["genes"]); vocab=np.array(meta["pathway_vocab"]); P=z["pathways_full"].astype(np.float64)
    Fg,Fp=design(labels,vocab,P,genes)
    tr,va,te=split_ids(labels,11)
    fpmean=Fp[tr].mean(0); fpstd=Fp[tr].std(0)+1e-3
    Fp=(Fp-fpmean)/fpstd
    out={"protocol":{"split_seed":11,"train_rows_including_control":int(len(tr)),"validation_rows":int(len(va)),"test_rows":int(len(te)),"test_labels":[str(x) for x in labels[te]],"panel_genes":int(len(genes)),"pathways":int(P.shape[1])}}
    # Main closed-form models and paired identity-level bootstrap.
    rp=Ridge(alpha=.1).fit(Fp[tr],y[tr]); rg=Ridge(alpha=10.0).fit(Fg[tr],y[tr])
    mean=np.repeat(y[tr].mean(0,keepdims=True),len(te),0); pp=rp.predict(Fp[te]); pg=rg.predict(Fg[te])
    out["fixed_split_metrics"]={"global_mean":metrics(y[te],mean),"pathway_ridge":metrics(y[te],pp),"gene_ridge":metrics(y[te],pg)}
    out["paired_bootstrap"]={"pathway_ridge_vs_global_mean":bootstrap_deltas(y[te],pp,mean),"pathway_ridge_vs_gene_ridge":bootstrap_deltas(y[te],pp,pg)}
    # Profile-level table supports stratified error and source-data plotting.
    rows=[]
    for j,i in enumerate(te):
        obs=y[i]; pred=pp[j]; toks=target_tokens(labels[i]);
        rows.append({"identity":str(labels[i]),"index":int(i),"n_targets":len(toks),"single_or_combo":"combo" if len(toks)>1 else "single","response_norm":float(np.linalg.norm(obs)),"rmse_pathway_ridge":float(np.sqrt(np.mean((obs-pred)**2))),"pearson_pathway_ridge":float(pearsonr(obs,pred).statistic) if np.std(pred)>1e-10 else 0.0})
    prof=pd.DataFrame(rows); prof.to_csv(OUT/"profile_error_stratification.csv",index=False)
    out["profile_strata"]={}
    for key,g in prof.groupby("single_or_combo"):
        out["profile_strata"][key]={"n_profiles":int(len(g)),"rmse_mean":float(g.rmse_pathway_ridge.mean()),"rmse_sd":float(g.rmse_pathway_ridge.std(ddof=1)) if len(g)>1 else 0.0,"pearson_mean":float(g.pearson_pathway_ridge.mean()),"response_norm_mean":float(g.response_norm.mean())}
    out["profile_strata"]["all"]={"n_profiles":int(len(prof)),"rmse_mean":float(prof.rmse_pathway_ridge.mean()),"rmse_sd":float(prof.rmse_pathway_ridge.std(ddof=1)),"pearson_mean":float(prof.pearson_pathway_ridge.mean()),"response_norm_mean":float(prof.response_norm.mean())}
    # Repeated identity splits quantify split-to-split variation without ever
    # selecting a model on confirmation identities.
    reps=[]
    for seed in [11,22,33,44,55,66,77,88,99,111]:
        tr0,va0,te0=split_ids(labels,seed); a=Ridge(alpha=.1).fit(Fp[tr0],y[tr0]); b=Ridge(alpha=10.).fit(Fg[tr0],y[tr0]);
        ma=metrics(y[te0],a.predict(Fp[te0])); mb=metrics(y[te0],b.predict(Fg[te0]));
        reps.append({"split_seed":seed,"n_train":int(len(tr0)),"n_test":int(len(te0)),"pathway_ridge":ma,"gene_ridge":mb,"delta_pearson":ma["pearson"]-mb["pearson"],"delta_rmse":ma["rmse"]-mb["rmse"]})
    out["repeated_splits"]={"runs":reps,"summary":{"pathway_ridge_pearson_mean":float(np.mean([r["pathway_ridge"]["pearson"] for r in reps])),"pathway_ridge_pearson_sd":float(np.std([r["pathway_ridge"]["pearson"] for r in reps],ddof=1)),"pathway_ridge_rmse_mean":float(np.mean([r["pathway_ridge"]["rmse"] for r in reps])),"pathway_ridge_rmse_sd":float(np.std([r["pathway_ridge"]["rmse"] for r in reps],ddof=1)),"delta_pearson_mean":float(np.mean([r["delta_pearson"] for r in reps])),"delta_pearson_sd":float(np.std([r["delta_pearson"] for r in reps],ddof=1))}}
    # Randomized pathway incidence controls preserve the observed module-size
    # distribution but destroy gene-to-pathway identity.
    null=[]
    sizes=(P>0).sum(0); rng=np.random.default_rng(314159)
    for k in range(20):
        Pn=np.zeros_like(P)
        for j,s in enumerate(sizes.astype(int)):
            ix=rng.choice(P.shape[0],size=int(s),replace=False); Pn[ix,j]=1/np.sqrt(max(int(s),1))
        _,Fn=design(labels,vocab,Pn,genes); fnmean=Fn[tr].mean(0); fnstd=Fn[tr].std(0)+1e-3; Fn=(Fn-fnmean)/fnstd; m=Ridge(alpha=.1).fit(Fn[tr],y[tr]); met=metrics(y[te],m.predict(Fn[te])); null.append({"seed":k,"rmse":met["rmse"],"pearson":met["pearson"],"spearman":met["spearman"]})
    out["random_pathway_null"]={"runs":null,"observed_pathway_ridge":out["fixed_split_metrics"]["pathway_ridge"],"null_pearson_mean":float(np.mean([x["pearson"] for x in null])),"null_pearson_sd":float(np.std([x["pearson"] for x in null],ddof=1))}
    (OUT/"paired_diagnostics.json").write_text(json.dumps(out,indent=2))
    print(json.dumps({"fixed":out["fixed_split_metrics"],"bootstrap":out["paired_bootstrap"],"repeated":out["repeated_splits"]["summary"],"null":out["random_pathway_null"]},indent=2))

if __name__=="__main__": main()
