"""Global hyperparameter confirmation for Ridge + target-PCA blend.

Each split independently selects the strongest baseline Ridge penalty on its
validation identities. A single candidate (alpha, target rank, blend weight)
is selected across all five validation splits, requiring all validation
metrics to improve on every split and maximizing mean standardized gain.
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
 active=np.flatnonzero(F.sum(0)>0);return lab,y,np.c_[F[:,[gi[x] for x in g]],F@P,F[:,active]],ctrl
def split(lab,s):
 ids=np.flatnonzero(lab!='CONTROL');ctrl=np.flatnonzero(lab=='CONTROL');o=ids.copy();np.random.default_rng(s).shuffle(o);ntr,nva=int(.7*len(o)),int(.15*len(o));return np.r_[o[:ntr],ctrl],o[ntr:ntr+nva],o[ntr+nva:]
def main():
 lab,y,X,ctrl=build();cache=[];global_rows=[]
 for seed in SEEDS:
  tr,va,te=split(lab,seed);xm=X[tr].mean(0);sd=X[tr].std(0)+1e-3;Xs=(X-xm)/sd;ym=y[tr].mean(0);yc=y[tr]-ym;pca=PCA(n_components=min(max(RANKS),len(tr)-1),svd_solver='full').fit(yc)
  bg=[]
  for a in ALPHAS:
   bm=Ridge(alpha=a,solver='cholesky').fit(Xs[tr],y[tr]);bg.append((met(y[va],bm.predict(Xs[va])),a,bm))
  bval,ba,bm=min(bg,key=lambda q:q[0]['rmse']);bpv=bm.predict(Xs[va]); cache.append((seed,tr,va,te,Xs,ym,yc,pca,bval,ba,bm,bpv))
  print('split',seed,'baseline',ba,bval,flush=True)
 # Evaluate same candidate configuration in every validation split.
 for a in ALPHAS:
  for r in RANKS:
   for w in WEIGHTS:
    ss=[]; ok=True
    for seed,tr,va,te,Xs,ym,yc,pca,bval,ba,bm,bpv in cache:
     U=pca.components_[:r];cm=Ridge(alpha=a,solver='cholesky').fit(Xs[tr],yc@U.T);cpv=cm.predict(Xs[va])@U+ym;cv=met(y[va],(1-w)*bpv+w*cpv);ok &= cv['rmse']<bval['rmse'] and cv['pearson']>bval['pearson'] and cv['spearman']>bval['spearman'];ss.append((bval,cv))
    gains=[]
    for b,c in ss:gains.append(((b['rmse']-c['rmse'])/b['rmse']+(c['pearson']-b['pearson'])/abs(b['pearson'])+(c['spearman']-b['spearman'])/abs(b['spearman']))/3)
    global_rows.append({'alpha':a,'rank':r,'weight':float(w),'strict_validation_all_splits':bool(ok),'mean_standardized_gain':float(np.mean(gains)),'mean_validation_rmse':float(np.mean([c['rmse'] for b,c in ss])),'split_validation':[{'baseline':b,'candidate':c,'standardized_gain':g} for (b,c),g in zip(ss,gains)]})
 strict=[q for q in global_rows if q['strict_validation_all_splits']]
 selected=max(strict,key=lambda q:(q['mean_standardized_gain'],q['mean_validation_rmse']*-1)) if strict else min(global_rows,key=lambda q:q['mean_validation_rmse'])
 a,r,w=selected['alpha'],selected['rank'],selected['weight'];rows=[]
 for seed,tr,va,te,Xs,ym,yc,pca,bval,ba,bm,bpv in cache:
  U=pca.components_[:r];cm=Ridge(alpha=a,solver='cholesky').fit(Xs[tr],yc@U.T);bpt=bm.predict(Xs[te]);cpt=cm.predict(Xs[te])@U+ym;bt=met(y[te],bpt);ct=met(y[te],(1-w)*bpt+w*cpt);rows.append({'split_seed':seed,'baseline_alpha':ba,'test_baseline':bt,'test_candidate':ct,'test_delta_candidate_minus_baseline':{k:ct[k]-bt[k] for k in ('rmse','pearson','spearman')}});print('test',seed,bt,ct,flush=True)
 out={'protocol':'global_validation_selected_blend_target_pca','selection':'single alpha/rank/weight shared across splits; baseline alpha independently validation-selected; candidates required all three validation improvements in every split; maximize mean standardized gain','selected_global_candidate':{'alpha':a,'rank':r,'weight':w,'strict_candidate_pool_size':len(strict),'mean_standardized_gain':selected['mean_standardized_gain']},'validation_grid':global_rows,'test_rows':rows,'all_test_metrics_improved_each_split':all(r['test_candidate']['rmse']<r['test_baseline']['rmse'] and r['test_candidate']['pearson']>r['test_baseline']['pearson'] and r['test_candidate']['spearman']>r['test_baseline']['spearman'] for r in rows),'input_hashes':{'pseudobulk':hashlib.sha256((DATA/'pseudobulk.npz').read_bytes()).hexdigest(),'metadata':hashlib.sha256((DATA/'metadata.json').read_bytes()).hexdigest()}};(OUT/'reduced_rank_blend_global_confirmation.json').write_text(json.dumps(out,indent=2));print(json.dumps({'selected':out['selected_global_candidate'],'all_test_metrics_improved_each_split':out['all_test_metrics_improved_each_split'],'deltas':[{'seed':r['split_seed'],**r['test_delta_candidate_minus_baseline']} for r in rows]},indent=2))
if __name__=='__main__':main()
