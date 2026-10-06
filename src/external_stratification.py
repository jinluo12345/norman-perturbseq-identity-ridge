"""Descriptive external metrics stratified by exact gene identity overlap."""
from pathlib import Path
import json
import numpy as np
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr, spearmanr
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed'; OUT=ROOT/'results'
def tok(x): return [] if x in {'CONTROL','NAN','*'} else [t for t in str(x).replace('/','+').replace('-','_').split('+') if t and t not in {'ONLY','MOD'}]
def feat(labels,vocab,P):
 gi={g:i for i,g in enumerate(vocab)}; F=np.zeros((len(labels),len(vocab)),np.float32)
 for i,l in enumerate(labels):
  for t in tok(l):
   if t in gi:F[i,gi[t]]=1
 return F@P
def met(y,p):
 return {'rmse':float(np.sqrt(np.mean((y-p)**2))),'pearson':float(pearsonr(y.ravel(),p.ravel()).statistic),'spearman':float(spearmanr(y.ravel(),p.ravel()).statistic),'n_profiles':int(len(y))}
def main():
 z=np.load(DATA/'pseudobulk.npz',allow_pickle=True); m=json.load(open(DATA/'metadata.json')); labels=np.array(m['labels']['norman']); y=z['y_norman'].astype(np.float32); c=np.where(labels=='CONTROL')[0]; y-=y[c].mean(0); vocab=np.array(m['pathway_vocab']); P=z['pathways_full']; tr=np.array(json.load(open(ROOT/'results/model_results.json'))['runs'][0]['split']['train']); F=feat(labels,vocab,P); mu=F[tr].mean(0); sd=F[tr].std(0)+1e-3; F=(F-mu)/sd; model=Ridge(alpha=.1).fit(F[tr],y[tr]); match=json.load(open(ROOT/'results/external_matched_identity.json')); out={}
 for name in ['dixit','adamson_single','adamson_combo']:
  labs=np.array(m['labels'][name]); yy=z['y_'+name].astype(np.float32); cc=np.where(labs=='CONTROL')[0]; yy-=yy[cc].mean(0); ff=feat(labs,vocab,P); pred=model.predict((ff-mu)/sd); non=np.where(labs!='CONTROL')[0]; matched=set(match[name]['matched_gene_identities']); groups={'matched':np.array([i for i in non if labs[i] in matched]),'unmatched':np.array([i for i in non if labs[i] not in matched])}; out[name]={'n_total_noncontrol':int(len(non)),'n_exact_matched':int(len(groups['matched'])),'matched_identities':sorted(matched),'groups':{}}
  for g,ix in groups.items(): out[name]['groups'][g]={'labels':[str(labs[i]) for i in ix],**(met(yy[ix],pred[ix]) if len(ix) else {'n_profiles':0})}
 (OUT/'external_stratification.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
