"""Locked confirmation of the validation-screened identity-block low-rank model.

For each pre-registered identity split, alpha and rank are selected on the
validation identities only.  Both the baseline (joint identity-augmented
Ridge) and candidate are then evaluated once on the held-out test identities.
No test scores are used in model selection.
"""
from pathlib import Path
import hashlib, json
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT/'data/processed'; OUT = ROOT/'results'
SEEDS = [11,22,33,44,55]; ALPHAS = [3,10,30,100]; RANKS = [4,8,16,32,64]

def metric(y,p):
    a,b=np.asarray(y).ravel(),np.asarray(p).ravel()
    return {'rmse':float(np.sqrt(mean_squared_error(y,p))), 'pearson':float(pearsonr(a,b).statistic), 'spearman':float(spearmanr(a,b).statistic), 'n_identities':int(len(y))}

def build():
    z=np.load(DATA/'pseudobulk.npz',allow_pickle=True); m=json.loads((DATA/'metadata.json').read_text())
    labs=np.asarray(m['labels']['norman'],str); y=z['y_norman'].astype(np.float64); ctrl=np.flatnonzero(labs=='CONTROL'); y=y-y[ctrl].mean(0)
    P=z['pathways_full'].astype(np.float64); vocab=np.asarray(m['pathway_vocab']); genes=np.asarray(m['genes']); gi={g:i for i,g in enumerate(vocab)}
    F=np.zeros((len(labs),len(vocab)),np.float64)
    for i,l in enumerate(labs):
        if l=='CONTROL': continue
        for t in str(l).replace('/','+').replace('-','_').split('+'):
            if t in gi: F[i,gi[t]]=1
    active=np.flatnonzero(F.sum(0)>0); X=np.c_[F[:,[gi[g] for g in genes]], F@P, F[:,active]]
    return labs,y,X,ctrl,active

def split(labs,seed):
    ids=np.flatnonzero(labs!='CONTROL'); ctrl=np.flatnonzero(labs=='CONTROL'); o=ids.copy(); np.random.default_rng(seed).shuffle(o); ntr,nva=int(.70*len(o)),int(.15*len(o)); return np.r_[o[:ntr],ctrl],o[ntr:ntr+nva],o[ntr+nva:]

def trunc_identity(W,rank,n_a=512,n_b=256):
    Wi=W[:,n_a+n_b:].T; u,s,vt=np.linalg.svd(Wi,full_matrices=False); r=min(rank,len(s)); return np.c_[W[:,:n_a+n_b],((u[:,:r]*s[:r])@vt[:r]).T]

def main():
    labs,y,X,ctrl,active=build(); rows=[]
    for seed in SEEDS:
        tr,va,te=split(labs,seed); mu=X[tr].mean(0); sd=X[tr].std(0)+1e-3; Xs=(X-mu)/sd
        # Baseline alpha and candidate alpha/rank are selected only on VA.
        bgrid=[]; cgrid=[]
        for a in ALPHAS:
            b=Ridge(alpha=a,solver='cholesky').fit(Xs[tr],y[tr]); bgrid.append((a,metric(y[va],b.predict(Xs[va]))))
            W=b.coef_.copy()
            for r in RANKS:
                Wr=trunc_identity(W,r); p=Xs[va]@Wr.T+b.intercept_; cgrid.append((a,r,metric(y[va],p)))
        ba=min(bgrid,key=lambda q:q[1]['rmse']); ca=min(cgrid,key=lambda q:q[2]['rmse'])
        # Refit each locked choice on the same training rows and score test.
        bm=Ridge(alpha=ba[0],solver='cholesky').fit(Xs[tr],y[tr]); btest=metric(y[te],bm.predict(Xs[te]))
        cm=Ridge(alpha=ca[0],solver='cholesky').fit(Xs[tr],y[tr]); W=trunc_identity(cm.coef_.copy(),ca[1]); ctest=metric(y[te],Xs[te]@W.T+cm.intercept_)
        rows.append({'split_seed':seed,'split_hash':hashlib.sha256(np.asarray(np.r_[tr,va,te],dtype=np.int64).tobytes()).hexdigest(),'selected_baseline_alpha':ba[0],'selected_candidate_alpha':ca[0],'selected_candidate_rank':ca[1],'validation_baseline':ba[1],'validation_candidate':ca[2],'test_baseline':btest,'test_candidate':ctest,'test_delta_candidate_minus_baseline':{k:ctest[k]-btest[k] for k in ['rmse','pearson','spearman']},'n_train_identities':int(len(tr)-len(ctrl)),'n_validation_identities':int(len(va)),'n_test_identities':int(len(te) )})
        print(seed, 'val',ba[0],ba[1],ca[0],ca[1],ca[2], 'test',btest,ctest)
    out={'protocol':'locked_confirmation_structured_lowrank_identity','candidate':'rank-constrained active identity-component coefficient block; gene/pathway blocks unchanged','selection':'alpha and rank selected by validation RMSE only; test scored once after lock','seeds':rows,'all_test_metrics_improved_each_split':all(r['test_candidate']['rmse']<r['test_baseline']['rmse'] and r['test_candidate']['pearson']>r['test_baseline']['pearson'] and r['test_candidate']['spearman']>r['test_baseline']['spearman'] for r in rows),'input_hashes':{'pseudobulk':hashlib.sha256((DATA/'pseudobulk.npz').read_bytes()).hexdigest(),'metadata':hashlib.sha256((DATA/'metadata.json').read_bytes()).hexdigest()}}
    (OUT/'lowrank_identity_confirmation.json').write_text(json.dumps(out,indent=2)); print(json.dumps({'all_test_metrics_improved_each_split':out['all_test_metrics_improved_each_split'],'deltas':[{'seed':r['split_seed'],**r['test_delta_candidate_minus_baseline']} for r in rows]},indent=2))

if __name__=='__main__': main()
