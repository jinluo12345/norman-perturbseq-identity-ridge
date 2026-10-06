"""Fair locked confirmation for Ridge + target-PCA decoder blending.

The baseline Ridge penalty is selected independently by validation RMSE and
then frozen. Candidate rank/penalty and convex weight are selected on the same
validation identities, always blending against that frozen strongest baseline.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.stats import pearsonr,spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/processed';OUT=ROOT/'results';SEEDS=[11,22,33,44,55];ALPHAS=[1,3,10,30,100];RANKS=[4,8,16,32,64,96,128,160];WEIGHTS=np.linspace(0,1,11)
def met(y,p):
 a,b=np.asarray(y).ravel(),np.asarray(p).ravel();return {'rmse':float(np.sqrt(mean_squared_error(y,p))),'pearson':float(pearsonr(a,b).statistic),'spearman':float(spearmanr(a,b).statistic),'n_identities':int(len(y))}
def tok(l):return [t for t in str(l).replace('/','+').replace('-','_').split('+') if t and t not in {'ONLY','MOD'}]
def build():
 z=np.load(DATA/'pseudobulk.npz',allow_pickle=True);m=json.loads((DATA/'metadata.json').read_text());lab=np.asarray(m['labels']['norman'],str);y=z['y_norman'].astype(float);ctrl=np.flatnonzero(lab=='CONTROL');y-=y[ctrl].mean(0);v=np.asarray(m['pathway_vocab']);g=np.asarray(m['genes']);P=z['pathways_full'].astype(float);gi={x:i for i,x in enumerate(v)};F=np.zeros((len(lab),len(v)))
 for i,l in enumerate(lab):
  for t in tok(l):
   if t in gi:F[i,gi[t]]=1
 active=np.flatnonzero(F.sum(0)>0);X=np.c_[F[:,[gi[x] for x in g]],F@P,F[:,active]];return lab,y,X,ctrl
def split(lab,s):
 ids=np.flatnonzero(lab!='CONTROL');ctrl=np.flatnonzero(lab=='CONTROL');o=ids.copy();np.random.default_rng(s).shuffle(o);ntr,nva=int(.7*len(o)),int(.15*len(o));return np.r_[o[:ntr],ctrl],o[ntr:ntr+nva],o[ntr+nva:]
def main():
 lab,y,X,ctrl=build();rows=[]
 for seed in SEEDS:
  tr,va,te=split(lab,seed);xm=X[tr].mean(0);sd=X[tr].std(0)+1e-3;Xs=(X-xm)/sd;ym=y[tr].mean(0);yc=y[tr]-ym;pca=PCA(n_components=min(max(RANKS),len(tr)-1),svd_solver='full').fit(yc)
  bg=[]
  for a in ALPHAS:
   bm=Ridge(alpha=a,solver='cholesky').fit(Xs[tr],y[tr]);bg.append((met(y[va],bm.predict(Xs[va])),a))
  bval,ba=min(bg,key=lambda q:q[0]['rmse']);bmodel=Ridge(alpha=ba,solver='cholesky').fit(Xs[tr],y[tr]);bpv=bmodel.predict(Xs[va]);bpt=bmodel.predict(Xs[te])
  grid=[]
  for a in ALPHAS:
   for r in RANKS:
    U=pca.components_[:r];sm=Ridge(alpha=a,solver='cholesky').fit(Xs[tr],yc@U.T);cpv=sm.predict(Xs[va])@U+ym
    for w in WEIGHTS:grid.append((met(y[va],(1-w)*bpv+w*cpv),a,r,float(w)))
  best=min(grid,key=lambda q:q[0]['rmse']); cval,ca,cr,w=best;U=pca.components_[:cr];sm=Ridge(alpha=ca,solver='cholesky').fit(Xs[tr],yc@U.T);cpt=sm.predict(Xs[te])@U+ym;ct=met(y[te],(1-w)*bpt+w*cpt);bt=met(y[te],bpt)
  rows.append({'split_seed':seed,'split_hash':hashlib.sha256(np.asarray(np.r_[tr,va,te],dtype=np.int64).tobytes()).hexdigest(),'baseline_alpha_locked':ba,'candidate_alpha':ca,'candidate_rank':cr,'blend_weight':w,'validation_baseline':bval,'validation_blend':cval,'test_baseline':bt,'test_blend':ct,'test_delta_blend_minus_baseline':{k:ct[k]-bt[k] for k in ('rmse','pearson','spearman')},'n_test_identities':int(len(te))});print(seed,'base',ba,bval,'blend',ca,cr,w,cval,'test',bt,ct,flush=True)
 out={'protocol':'fair_locked_validation_selected_blend_ridge_target_pca','candidate':'convex blend of independently locked identity-augmented Ridge baseline and target-PCA low-rank decoder','selection':'baseline alpha independently selected by validation RMSE; candidate alpha/rank/weight selected against frozen baseline; test after lock','blend_weights':[float(w) for w in WEIGHTS],'seeds':rows,'all_test_metrics_improved_each_split':all(r['test_blend']['rmse']<r['test_baseline']['rmse'] and r['test_blend']['pearson']>r['test_baseline']['pearson'] and r['test_blend']['spearman']>r['test_baseline']['spearman'] for r in rows),'input_hashes':{'pseudobulk':hashlib.sha256((DATA/'pseudobulk.npz').read_bytes()).hexdigest(),'metadata':hashlib.sha256((DATA/'metadata.json').read_bytes()).hexdigest()}};(OUT/'reduced_rank_blend_confirmation_fair.json').write_text(json.dumps(out,indent=2));print(json.dumps({'all_test_metrics_improved_each_split':out['all_test_metrics_improved_each_split'],'deltas':[{'seed':r['split_seed'],**r['test_delta_blend_minus_baseline']} for r in rows]},indent=2))
if __name__=='__main__':main()
