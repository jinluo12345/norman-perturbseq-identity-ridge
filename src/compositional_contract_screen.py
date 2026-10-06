"""Validation-only screen for a compositional Perturb-seq contract.

The contract holds out perturbation *combinations* while retaining every
component's singleton profile in training.  This asks a distinct scientific
question from random identity splits: can a decoder extrapolate response to a
new pair from observed singleton effects and training combinations?  The
locked test combinations are never loaded or scored.  Candidate selection is
validation-only and compared with the identity-augmented Ridge reference.
"""
from pathlib import Path
import hashlib, json
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"; OUT = ROOT / "results"
SEEDS = [11,22,33,44,55]

def metric(y,p):
    a,b=np.asarray(y).ravel(),np.asarray(p).ravel()
    return {"rmse":float(np.sqrt(mean_squared_error(y,p))),
            "pearson":float(pearsonr(a,b).statistic),
            "spearman":float(spearmanr(a,b).statistic),"n_identities":int(len(y))}

def toks(l): return [] if l == "CONTROL" else str(l).split("+")

def build():
    z=np.load(DATA/"pseudobulk.npz",allow_pickle=True); m=json.loads((DATA/"metadata.json").read_text())
    labs=np.asarray(m["labels"]["norman"],str); y=z["y_norman"].astype(np.float64)
    c=np.flatnonzero(labs=="CONTROL"); y=y-y[c].mean(0)
    vocab=np.asarray(m["pathway_vocab"]); genes=np.asarray(m["genes"]); P=z["pathways_full"].astype(np.float64)
    gi={g:i for i,g in enumerate(vocab)}
    comps=sorted({t for l in labs for t in toks(l) if t in gi}); ci={g:i for i,g in enumerate(comps)}
    F=np.zeros((len(labs),len(comps)),float)
    for i,l in enumerate(labs):
        for t in toks(l):
            if t in ci: F[i,ci[t]]=1
    # Same identity-augmented baseline contract as the strongest reference.
    # Component-to-gene incidence and the derived pathway score are fixed
    # input transforms; F itself remains the 105-dimensional component block.
    vg={g:i for i,g in enumerate(vocab)}
    T=np.zeros((len(comps),len(vocab)),float)
    for k,g in enumerate(comps):
        if g in vg: T[k,vg[g]]=1.0
    G=F@T[:,[vg[g] for g in genes]]
    X=np.c_[G,F@T@P,F]
    # Pair-overlap pathway descriptor used only in the residual candidate.
    H=np.zeros((len(labs),P.shape[1]),float)
    for i,l in enumerate(labs):
        ts=[t for t in toks(l) if t in vg]
        if len(ts)>=2:
            H[i]=sum((P[vg[a]]*P[vg[b]]) for a in ts for b in ts if a<b)
    return labs,y,F,X,H,comps

def split(labs,seed):
    singles=np.flatnonzero(np.array([l!='CONTROL' and '+' not in l for l in labs]))
    combos=np.flatnonzero(np.array(['+' in l for l in labs])); rng=np.random.default_rng(seed); q=combos.copy(); rng.shuffle(q)
    ntr=int(.60*len(q)); nva=int(.20*len(q)); trc,va,te=q[:ntr],q[ntr:ntr+nva],q[ntr+nva:]
    ctrl=np.flatnonzero(labs=='CONTROL'); tr=np.r_[ctrl,singles,trc]
    return tr,va,te,singles,trc

def main():
    labs,y,F,X,H,comps=build(); rows=[]
    for seed in SEEDS:
        tr,va,te,singles,trc=split(labs,seed)
        mu=X[tr].mean(0); sd=X[tr].std(0)+1e-3; Xs=(X-mu)/sd
        base=[]
        for a in [1,3,10,30,100]:
            fit=Ridge(alpha=a,solver='cholesky').fit(Xs[tr],y[tr]); base.append((a,fit,metric(y[va],fit.predict(Xs[va]))))
        ba, bf, bm=min(base,key=lambda q:q[2]['rmse'])
        # Singleton composition: estimate each component effect from its singleton,
        # then sum the observed effects for a held-out pair.
        E=np.zeros((len(comps),y.shape[1]))
        for k,g in enumerate(comps):
            ix=np.flatnonzero(labs==g)
            if len(ix): E[k]=y[ix[0]]
        add=F@E
        addrows=[]
        for lam in [0.25,0.5,0.75,1.0,1.25]: addrows.append((lam,metric(y[va],lam*add[va])))
        la,am=min(addrows,key=lambda q:q[1]['rmse'])
        # Residual pathway-overlap correction: train only on observed training
        # combinations, with no held-out combination targets entering the fit.
        combo_tr=np.asarray([i for i in tr if '+' in labs[i]],int)
        cand=[]
        if len(combo_tr)>=5:
            hm=H[combo_tr].mean(0); hs=H[combo_tr].std(0)+1e-3; Hs=(H-hm)/hs
            residual=y[combo_tr]-la*add[combo_tr]
            for a in [1,3,10,30,100,300]:
                rf=Ridge(alpha=a,solver='cholesky').fit(Hs[combo_tr],residual)
                for lamr in [0.25,0.5,0.75,1.0]:
                    pp=la*add[va]+lamr*rf.predict(Hs[va]); cand.append((a,lamr,rf,hm,hs,metric(y[va],pp)))
        winners=[q for q in cand if q[5]['rmse']<bm['rmse'] and q[5]['pearson']>bm['pearson'] and q[5]['spearman']>bm['spearman']]
        bestc=min(cand,key=lambda q:q[5]['rmse']) if cand else None
        rows.append({'seed':seed,'split_hash':hashlib.sha256(np.asarray(np.r_[tr,va,te],dtype=np.int64).tobytes()).hexdigest(),'n_train_singletons':int(len(singles)),'n_train_combos':int(len(trc)),'n_validation_combos':int(len(va)),'n_test_combos_locked':int(len(te)),'baseline_alpha':ba,'baseline_validation':bm,'additive_lambda':la,'additive_validation':am,'best_residual_overlap':None if bestc is None else {'alpha':bestc[0],'residual_scale':bestc[1],'validation':bestc[5]},'residual_all_metric_winners':len(winners)})
        print(seed, bm, 'add',am, 'best_residual',None if bestc is None else bestc[5], 'winners',len(winners),flush=True)
    out={'protocol':'validation_only_compositional_generalization_screen','contract':'hold out 20% of Norman two-component identities while retaining every singleton component; test combinations locked','candidate':'singleton response composition plus pathway-overlap residual correction','baseline':'identity-augmented Ridge on gene panel + Reactome scores + component indicators','selection':'all alpha and composition scales selected on validation only; locked combinations not scored','seeds':rows,'all_metric_winner_each_split':all(r['residual_all_metric_winners']>0 for r in rows),'input_hashes':{'pseudobulk':hashlib.sha256((DATA/'pseudobulk.npz').read_bytes()).hexdigest(),'metadata':hashlib.sha256((DATA/'metadata.json').read_bytes()).hexdigest()}}
    (OUT/'compositional_contract_validation.json').write_text(json.dumps(out,indent=2)); print(json.dumps({'all_metric_winner_each_split':out['all_metric_winner_each_split'],'summary':rows},indent=2))
if __name__=='__main__': main()
