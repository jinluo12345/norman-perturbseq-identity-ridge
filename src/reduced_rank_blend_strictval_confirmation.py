"""Locked test after strict all-metric validation-only blend selection."""
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
 active=np.flatnonzero(F.sum(0)>0);return lab,y,np.c_[F[:,[gi[x] for x in g]],F@P,F[:,active]],ctrl
def split(lab,s):
 ids=np.flatnonzero(lab!='CONTROL');ctrl=np.flatnonzero(lab=='CONTROL');o=ids.copy();np.random.default_rng(s).shuffle(o);ntr,nva=int(.7*len(o)),int(.15*len(o));return np.r_[o[:ntr],ctrl],o[ntr:ntr+nva],o[ntr+nva:]
def main():
 lab,y,X,ctrl=build();rows=[]
 for seed in SEEDS:
  tr,va,te=split(lab,seed);xm=X[tr].mean(0);sd=X[tr].std(0)+1e-3;Xs=(X-xm)/sd;ym=y[tr].mean(0);yc=y[tr]-ym;pca=PCA(n_components=min(max(RANKS),len(tr)-1),svd_solver='full').fit(yc)
  bg=[]
  for a in ALPHAS:
   bm=Ridge(alpha=a,solver='cholesky').fit(Xs[tr],y[tr]);bg.append((met(y[va],bm.predict(Xs[va])),a))
  bval,ba=min(bg,key=lambda q:q[0]['rmse']);bm=Ridge(alpha=ba,solver='cholesky').fit(Xs[tr],y[tr]);bpv=bm.predict(Xs[va]);cands=[]
  for a in ALPHAS:
   for r in RANKS:
    U=pca.components_[:r];cm=Ridge(alpha=a,solver='cholesky').fit(Xs[tr],yc@U.T);cpv=cm.predict(Xs[va])@U+ym
    for w in WEIGHTS:
     m=met(y[va],(1-w)*bpv+w*cpv);ok=m['rmse']<bval['rmse'] and m['pearson']>bval['pearson'] and m['spearman']>bval['spearman'];
     if ok:
      si=((bval['rmse']-m['rmse'])/bval['rmse']+(m['pearson']-bval['pearson'])/abs(bval['pearson'])+(m['spearman']-bval['spearman'])/abs(bval['spearman']))/3;cands.append((m,a,r,float(w),si))
  if cands:
   # Pre-registered priority: maximum Spearman, then standardized mean gain,
   # then RMSE and smallest complexity for deterministic ties.
   cval,ca,cr,w,si=max(cands,key=lambda q:(q[0]['spearman'],q[4],-q[0]['rmse'],-q[2],-q[1],-q[3]))
   U=pca.components_[:cr];cm=Ridge(alpha=ca,solver='cholesky').fit(Xs[tr],yc@U.T);bpt=bm.predict(Xs[te]);cpt=cm.predict(Xs[te])@U+ym;bt=met(y[te],bpt);ct=met(y[te],(1-w)*bpt+w*cpt)
  else: ca=cr=w=si=None;cval=None;bt=met(y[te],bm.predict(Xs[te]));ct=None
  rows.append({'split_seed':seed,'baseline_alpha':ba,'validation_baseline':bval,'n_strict_validation_candidates':len(cands),'selected_candidate_alpha':ca,'selected_rank':cr,'selected_weight':w,'selected_standardized_gain':si,'validation_candidate':cval,'test_baseline':bt,'test_candidate':ct,'test_delta_candidate_minus_baseline':None if ct is None else {k:ct[k]-bt[k] for k in ('rmse','pearson','spearman')},'n_test_identities':int(len(te))});print(seed,'base',bval,'n',len(cands),'selected',ca,cr,w,cval,'test',bt,ct,flush=True)
 out={'protocol':'strict_validation_all_metric_blend_selection','selection':'baseline alpha independently selected by validation RMSE; candidate only if validation RMSE lower and Pearson/Spearman higher; max validation Spearman then standardized mean gain; test after lock','seeds':rows,'all_test_metrics_improved_each_split':all(r['test_candidate'] is not None and r['test_candidate']['rmse']<r['test_baseline']['rmse'] and r['test_candidate']['pearson']>r['test_baseline']['pearson'] and r['test_candidate']['spearman']>r['test_baseline']['spearman'] for r in rows),'input_hashes':{'pseudobulk':hashlib.sha256((DATA/'pseudobulk.npz').read_bytes()).hexdigest(),'metadata':hashlib.sha256((DATA/'metadata.json').read_bytes()).hexdigest()}};(OUT/'reduced_rank_blend_strictval_confirmation.json').write_text(json.dumps(out,indent=2));print(json.dumps({'all_test_metrics_improved_each_split':out['all_test_metrics_improved_each_split'],'deltas':[{'seed':r['split_seed'],**r['test_delta_candidate_minus_baseline']} for r in rows if r['test_delta_candidate_minus_baseline'] is not None]},indent=2))
if __name__=='__main__':main()
