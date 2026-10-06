from pathlib import Path
import json,h5py,numpy as np
from scipy.stats import spearmanr
from repaired_split_local_controls import load_source, extract_panel
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed'; RAW=ROOT/'data/raw/NormanWeissman2019_filtered.h5ad'
def met(y,p):
 a,b=y.ravel(),p.ravel(); return {'rmse':float(np.sqrt(np.mean((a-b)**2))),'pearson':float(np.corrcoef(a,b)[0,1]),'spearman':float(spearmanr(a,b).statistic),'n_cells':int(len(y))}
def ridge(x,y,q,alpha):
 xm=x.mean(0); xs=x.std(0)+1e-3; X=(x-xm)/xs; Q=(q-xm)/xs; ym=y.mean(0); Xc=X-X.mean(0); Y=y-ym
 A=Xc.T@Xc + alpha*np.eye(Xc.shape[1]); W=np.linalg.solve(A,Xc.T@Y); return (Q-X.mean(0))@W+ym
def main():
 z=np.load(DATA/'cell_panel.npz',allow_pickle=True); labels=np.asarray(z['labels'],str); source=z['source_row'].astype(int)
 shape,indptr,raw_genes,raw_labels,gem_all,lib=load_source(); gem=gem_all[source]; fixed_panel=np.asarray(json.loads((ROOT/'.tmp/metadata_fixed_degree_panel.json').read_text())['genes'][:512],dtype=object); cells=extract_panel(indptr,raw_genes,source,lib,fixed_panel).astype(np.float64)
 m=json.loads((DATA/'metadata.json').read_text()); idlab=np.asarray(m['labels']['norman'],str); pp=np.load(DATA/'pseudobulk.npz',allow_pickle=True); P=pp['pathways_full'].astype(float); v=np.asarray(m['pathway_vocab']); genes=fixed_panel; gi={g:i for i,g in enumerate(v)}; F=np.zeros((len(idlab),len(v)))
 for i,l in enumerate(idlab):
  if l!='CONTROL':
   for t in str(l).replace('/','+').replace('-','_').split('+'):
    if t in gi:F[i,gi[t]]=1
 active=np.flatnonzero(F.sum(0)>0); gene_ix=[gi[g] for g in genes if g in gi]; Ix=np.c_[F[:,gene_ix],F@P,F[:,active]]; idmap={l:i for i,l in enumerate(idlab)}; cid=np.asarray([idmap[l] for l in labels]); control=(labels=='CONTROL')&((source%2)==0); target=cells-cells[control].mean(0); batches=np.unique(gem); bm={int(b):j for j,b in enumerate(batches)}; BO=np.eye(len(batches))[np.array([bm[int(b)] for b in gem])]
 conf=json.loads((ROOT/'results/repaired_split_local_controls.json').read_text()); ids=np.flatnonzero(idlab!='CONTROL'); rows=[]
 for ex in conf['seeds']:
  seed=int(ex['seed']); rng=np.random.default_rng(seed); order=ids.copy(); rng.shuffle(order); ntr,nva=int(.7*len(order)),int(.15*len(order)); train=set(idlab[order[:ntr]].tolist())|{'CONTROL'}; test=set(idlab[order[ntr+nva:]].tolist()); trainids=np.flatnonzero(np.isin(idlab,list(train))); testmask=np.isin(labels,list(test))
  by=np.asarray([target[cid==ii].mean(0) for ii in trainids]); qtest=Ix[cid[testmask]]; bptest=ridge(Ix[trainids],by,qtest,float(ex['alphas']['identity_mean'])); gx=[];gy=[]
  for ii in trainids:
   for b in batches:
    q=(cid==ii)&(gem==b)
    if q.sum()>=2: gx.append(np.r_[Ix[ii],BO[np.flatnonzero(q)[0]]]); gy.append(target[q].mean(0))
  gx,gy=np.asarray(gx),np.asarray(gy); cptest=ridge(gx,gy,np.c_[Ix[cid[testmask]],BO[testmask]],float(ex['alphas']['candidate'])); strata={}
  for name,mask in [('all',testmask),('single',testmask&np.asarray(['+' not in l for l in labels])),('combination',testmask&np.asarray(['+' in l for l in labels]))]:
   local=np.flatnonzero(mask[testmask]); labs=np.unique(labels[mask]); strata[name]={'baseline':met(target[testmask][local],bptest[local]),'candidate':met(target[testmask][local],cptest[local]),'n_identities':int(len(labs))}
  batch_rows={}
  for b in batches:
   mask=testmask&(gem==b)
   if mask.sum(): local=np.flatnonzero(mask[testmask]); batch_rows[str(int(b))]={'baseline':met(target[testmask][local],bptest[local]),'candidate':met(target[testmask][local],cptest[local]),'n_identities':int(len(np.unique(labels[mask])))}
  rows.append({'seed':seed,'strata':strata,'batch':batch_rows,'test_identity_count':len(test)}); print(seed,flush=True)
 out={'protocol':'repaired_frozen_control_error_stratification','control_reference':{'rule':'CONTROL and source_row % 2 == 0','n_cells':int(control.sum())},'selection':'alphas copied from repaired validation-only locked bundle','rows':rows}; (ROOT/'results/repaired_stratified_analysis.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__': main()
