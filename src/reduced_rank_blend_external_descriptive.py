"""Descriptive external archive check for the fair validation-locked blend.

Norman validation chooses all configurations.  Dixit/Adamson rows are only
scored after the model and feature standardization statistics are frozen.
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
 a,b=np.asarray(y).ravel(),np.asarray(p).ravel();return {'rmse':float(np.sqrt(mean_squared_error(y,p))),'pearson':float(pearsonr(a,b).statistic) if np.std(b)>1e-12 else 0.,'spearman':float(spearmanr(a,b).statistic) if np.std(b)>1e-12 else 0.,'n_identities':int(len(y))}
def tok(l):return [t for t in str(l).replace('/','+').replace('-','_').split('+') if t and t not in {'ONLY','MOD'}]
def feature(labels,vocab,genes,P,active=None):
 gi={g:i for i,g in enumerate(vocab)};F=np.zeros((len(labels),len(vocab)),float)
 for i,l in enumerate(labels):
  for t in tok(l):
   if t in gi:F[i,gi[t]]=1
 if active is None: active=np.flatnonzero(F.sum(0)>0)
 return np.c_[F[:,[gi[g] for g in genes]],F@P,F[:,active]]
def metric_data(z,m,dname,vocab,genes,P,active=None):
 labels=np.asarray(m['labels'][dname],str); y=z['y_'+dname].astype(float); c=np.flatnonzero(labels=='CONTROL'); y=y-(y[c].mean(0) if len(c) else y.mean(0)); return labels,y,feature(labels,vocab,genes,P,active)
def split(labels,s):
 ids=np.flatnonzero(labels!='CONTROL');ctrl=np.flatnonzero(labels=='CONTROL');o=ids.copy();np.random.default_rng(s).shuffle(o);ntr,nva=int(.7*len(o)),int(.15*len(o));return np.r_[o[:ntr],ctrl],o[ntr:ntr+nva],o[ntr+nva:]
def main():
 z=np.load(DATA/'pseudobulk.npz',allow_pickle=True);m=json.loads((DATA/'metadata.json').read_text());v=np.asarray(m['pathway_vocab']);g=np.asarray(m['genes']);P=z['pathways_full'].astype(float);lab,y,X=metric_data(z,m,'norman',v,g,P)
 gi={q:i for i,q in enumerate(v)}; Fn=np.zeros((len(lab),len(v)))
 for i,l in enumerate(lab):
  for t in tok(l):
   if t in gi: Fn[i,gi[t]]=1
 active=np.flatnonzero(Fn.sum(0)>0); external={d:metric_data(z,m,d,v,g,P,active) for d in ('dixit','adamson_single','adamson_combo')};rows=[]
 for seed in SEEDS:
  tr,va,te=split(lab,seed);xm=X[tr].mean(0);sd=X[tr].std(0)+1e-3;Xs=(X-xm)/sd;ym=y[tr].mean(0);yc=y[tr]-ym;pca=PCA(n_components=min(max(RANKS),len(tr)-1),svd_solver='full').fit(yc)
  bg=[]
  for a in ALPHAS:
   bm=Ridge(alpha=a,solver='cholesky').fit(Xs[tr],y[tr]);bg.append((met(y[va],bm.predict(Xs[va])),a))
  bval,ba=min(bg,key=lambda q:q[0]['rmse']);bm=Ridge(alpha=ba,solver='cholesky').fit(Xs[tr],y[tr]);bpv=bm.predict(Xs[va]);grid=[]
  for a in ALPHAS:
   for r in RANKS:
    U=pca.components_[:r];cm=Ridge(alpha=a,solver='cholesky').fit(Xs[tr],yc@U.T);cpv=cm.predict(Xs[va])@U+ym
    for w in WEIGHTS:grid.append((met(y[va],(1-w)*bpv+w*cpv),a,r,float(w)))
  cval,ca,cr,w=min(grid,key=lambda q:q[0]['rmse']);U=pca.components_[:cr];cm=Ridge(alpha=ca,solver='cholesky').fit(Xs[tr],yc@U.T);exrows={}
  for d,(elab,ey,eX) in external.items():
   keep=np.flatnonzero(elab!='CONTROL');eXs=(eX[keep]-xm)/sd;bp=bm.predict(eXs);cp=cm.predict(eXs)@U+ym;blend=(1-w)*bp+w*cp;exrows[d]={'n_profiles':int(len(keep)),'baseline':met(ey[keep],bp),'blend':met(ey[keep],blend),'delta_blend_minus_baseline':{k:float(met(ey[keep],blend)[k]-met(ey[keep],bp)[k]) for k in ('rmse','pearson','spearman')}}
  rows.append({'split_seed':seed,'baseline_alpha':ba,'candidate_alpha':ca,'candidate_rank':cr,'blend_weight':w,'external':exrows})
  print(seed,ba,ca,cr,w,json.dumps(exrows),flush=True)
 out={'protocol':'descriptive_external_stress_test_fair_locked_blend','selection':'all configurations selected from Norman train/validation only; external rows scored after lock','archive_rows':rows,'external_input_hashes':{'pseudobulk':hashlib.sha256((DATA/'pseudobulk.npz').read_bytes()).hexdigest(),'metadata':hashlib.sha256((DATA/'metadata.json').read_bytes()).hexdigest()}};(OUT/'reduced_rank_blend_external_descriptive.json').write_text(json.dumps(out,indent=2));print('saved')
if __name__=='__main__':main()
