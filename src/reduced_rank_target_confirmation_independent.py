"""Independent locked confirmation for the target-PCA reduced-rank decoder.

The target PCA basis is fitted on training responses only.  Validation chooses
rank and Ridge penalty; test identities are touched only for final metrics.
"""
from pathlib import Path
import hashlib, json
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed'; OUT=ROOT/'results'
SEEDS=[11,22,33,44,55]; ALPHAS=[1,3,10,30,100]; RANKS=[4,8,16,32,64,96,128,160]

def metric(y,p):
    a,b=np.asarray(y).ravel(),np.asarray(p).ravel()
    return {'rmse':float(np.sqrt(mean_squared_error(y,p))),'pearson':float(pearsonr(a,b).statistic),'spearman':float(spearmanr(a,b).statistic),'n_identities':int(len(y))}

def tok(l): return [t for t in str(l).replace('/','+').replace('-','_').split('+') if t and t not in {'ONLY','MOD'}]

def build():
    z=np.load(DATA/'pseudobulk.npz',allow_pickle=True); m=json.loads((DATA/'metadata.json').read_text())
    labels=np.asarray(m['labels']['norman'],str); y=z['y_norman'].astype(np.float64); ctrl=np.flatnonzero(labels=='CONTROL'); y=y-y[ctrl].mean(0)
    vocab=np.asarray(m['pathway_vocab']); genes=np.asarray(m['genes']); P=z['pathways_full'].astype(np.float64); gi={g:i for i,g in enumerate(vocab)}
    F=np.zeros((len(labels),len(vocab)),np.float64)
    for i,l in enumerate(labels):
        for t in tok(l):
            if t in gi: F[i,gi[t]]=1
    active=np.flatnonzero(F.sum(0)>0); X=np.c_[F[:,[gi[g] for g in genes]],F@P,F[:,active]]
    return labels,y,X,ctrl,active

def split(labels,seed):
    ids=np.flatnonzero(labels!='CONTROL'); ctrl=np.flatnonzero(labels=='CONTROL'); o=ids.copy(); np.random.default_rng(seed).shuffle(o); ntr,nva=int(.70*len(o)),int(.15*len(o)); return np.r_[o[:ntr],ctrl],o[ntr:ntr+nva],o[ntr+nva:]

def main():
    labels,y,X,ctrl,active=build(); rows=[]
    for seed in SEEDS:
        tr,va,te=split(labels,seed); xm=X[tr].mean(0); xs=X[tr].std(0)+1e-3; Xs=(X-xm)/xs
        bgrid=[]
        for a in ALPHAS:
            b=Ridge(alpha=a,solver='cholesky').fit(Xs[tr],y[tr]); bgrid.append((metri:=metric(y[va],b.predict(Xs[va])),a))
        bval,ba=min(bgrid,key=lambda z:z[0]['rmse']); ymean=y[tr].mean(0); yc=y[tr]-ymean
        pca=PCA(n_components=min(max(RANKS),len(tr)-1),svd_solver='full').fit(yc)
        cand=[]
        for r in RANKS:
            U=pca.components_[:r]; scores=yc@U.T
            for a in ALPHAS:
                q=Ridge(alpha=a,solver='cholesky').fit(Xs[tr],scores); pred=q.predict(Xs[va])@U+ymean
                cand.append((metric(y[va],pred),r,a))
        cval,cr,ca=min(cand,key=lambda z:z[0]['rmse'])
        # Lock choices and score the test exactly once.
        bm=Ridge(alpha=ba,solver='cholesky').fit(Xs[tr],y[tr]); btest=metric(y[te],bm.predict(Xs[te]))
        U=pca.components_[:cr]; scores=yc@U.T; cm=Ridge(alpha=ca,solver='cholesky').fit(Xs[tr],scores); ctest=metric(y[te],cm.predict(Xs[te])@U+ymean)
        rows.append({'split_seed':seed,'split_hash':hashlib.sha256(np.asarray(np.r_[tr,va,te],dtype=np.int64).tobytes()).hexdigest(),'selected_baseline_alpha':ba,'selected_rank':cr,'selected_candidate_alpha':ca,'validation_baseline':bval,'validation_candidate':cval,'target_pca_explained_variance_ratio_sum':float(pca.explained_variance_ratio_[:cr].sum()),'target_pca_first160_variance_ratio_sum':float(pca.explained_variance_ratio_.sum()),'test_baseline':btest,'test_candidate':ctest,'test_delta_candidate_minus_baseline':{k:ctest[k]-btest[k] for k in ('rmse','pearson','spearman')},'n_train_identities':int(len(tr)-len(ctrl)),'n_validation_identities':int(len(va)),'n_test_identities':int(len(te))})
        print(seed,'val',ba,bval,'candidate',cr,ca,cval,'test',btest,ctest,flush=True)
    out={'protocol':'independent_locked_confirmation_reduced_rank_target_decoder','feature_definition':'512 gene membership + 256 Reactome pathway scores + active perturbation components','candidate':'PCA of training responses followed by Ridge score decoder; target PCA fit on train only','selection':'alpha and target rank selected by validation RMSE; test identities scored only after lock','seeds':rows,'all_test_metrics_improved_each_split':all(r['test_candidate']['rmse']<r['test_baseline']['rmse'] and r['test_candidate']['pearson']>r['test_baseline']['pearson'] and r['test_candidate']['spearman']>r['test_baseline']['spearman'] for r in rows),'input_hashes':{'pseudobulk':hashlib.sha256((DATA/'pseudobulk.npz').read_bytes()).hexdigest(),'metadata':hashlib.sha256((DATA/'metadata.json').read_bytes()).hexdigest()}}
    (OUT/'reduced_rank_target_confirmation_independent.json').write_text(json.dumps(out,indent=2)); print(json.dumps({'all_test_metrics_improved_each_split':out['all_test_metrics_improved_each_split'],'deltas':[{'seed':r['split_seed'],**r['test_delta_candidate_minus_baseline']} for r in rows]},indent=2))

if __name__=='__main__': main()
