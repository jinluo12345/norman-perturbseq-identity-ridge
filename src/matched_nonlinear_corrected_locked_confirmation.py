"""Locked test confirmation of validation-selected matched nonlinear models."""
from pathlib import Path
import hashlib,json,random,gc
import numpy as np
from scipy.stats import pearsonr,spearmanr,rankdata
from sklearn.metrics import mean_squared_error
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed'; RAW=ROOT/'data/raw/NormanWeissman2019_filtered.h5ad'; OUT=ROOT/'results'; B=2000; SPLITS=[11,22,33,44,55]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metric(y,p):
 a,b=np.asarray(y),np.asarray(p); return {'rmse':float(np.sqrt(np.mean((a-b)**2))),'pearson':float(pearsonr(a.ravel(),b.ravel()).statistic),'spearman':float(spearmanr(a.ravel(),b.ravel()).statistic),'n_cells':int(len(a))}
def perout(y,p):
 vals={'rmse':np.sqrt(np.mean((y-p)**2,axis=0)).tolist(),'pearson':[],'spearman':[]}
 for j in range(y.shape[1]):
  # Correlation routines can still flag numerically constant vectors when
  # their floating-point range is non-zero but below numerical resolution.
  # Encode all such undefined/non-finite per-output statistics as JSON null.
  if np.ptp(y[:,j]) <= 1e-12 or np.ptp(p[:,j]) <= 1e-12:
   vals['pearson'].append(None); vals['spearman'].append(None)
  else:
   pr=float(pearsonr(y[:,j],p[:,j]).statistic); sr=float(spearmanr(y[:,j],p[:,j]).statistic)
   vals['pearson'].append(pr if np.isfinite(pr) else None); vals['spearman'].append(sr if np.isfinite(sr) else None)
 return vals
def clean(x): return str(x.decode() if isinstance(x,(bytes,np.bytes_)) else x).upper().strip()
def ridge(xtr,ytr,xq,a):
 ym,xm=ytr.mean(0),xtr.mean(0);u,s,vt=np.linalg.svd(xtr-xm,full_matrices=False); return (xq-xm)@(vt.T@((s/(s*s+a))[:,None]*(u.T@(ytr-ym))))+ym
def bootstrap(y, preds, labels, seed):
 labs=np.unique(labels); rng=np.random.default_rng(seed+800000); arr=[]; yy=[]; pp={k:[] for k in preds}
 for lab in labs:
  m=labels==lab; yy.append(y[m]); [pp[k].append(preds[k][m]) for k in preds]
 out={}
 # fixed global ranks before cluster resampling
 yr=rankdata(y.ravel()); prs={k:rankdata(preds[k].ravel()) for k in preds}; offsets=np.cumsum([0]+[a.size for a in yy]); stats=[]
 for k in preds:
  pa=preds[k]; vals=[]
  for i,m in enumerate([labels==x for x in labs]):
   sl=slice(offsets[i],offsets[i+1]); yv=y[m].ravel(); pv=pa[m].ravel(); vals.append([len(yv),np.sum((yv-pv)**2),np.sum(yv),np.sum(pv),np.sum(yv*yv),np.sum(pv*pv),np.sum(yv*pv),np.sum(yr[sl]),np.sum(prs[k][sl]),np.sum(yr[sl]**2),np.sum(prs[k][sl]**2),np.sum(yr[sl]*prs[k][sl])])
  v=np.asarray(vals); ix=rng.integers(0,len(v),(B,len(v))); s=v[ix].sum(1); n,sse,sy,sp,sy2,sp2,syp,ry,rp,ry2,rp2,ryp=s.T; out[k]=np.c_[np.sqrt(sse/n),(syp-sy*sp/n)/np.sqrt(np.maximum((sy2-sy*sy/n)*(sp2-sp*sp/n),1e-30)),(ryp-ry*rp/n)/np.sqrt(np.maximum((ry2-ry*ry/n)*(rp2-rp*rp/n),1e-30))]
 return out

def prepare():
 import matched_nonlinear_validation as v
 z=np.load(DATA/'cell_panel.npz',allow_pickle=True); labels=np.asarray(z['labels'],str); source=np.asarray(z['source_row'],int); valres=json.loads((OUT/'matched_nonlinear_validation_corrected.json').read_text()); panel=np.asarray(json.loads((OUT/'repaired_split_local_controls.json').read_text())['seeds'][0]['panel_genes'][:512],object); indptr,raw,gemall,lib=v.load_raw(); gene=v.extract(indptr,raw,source,lib,panel).astype(np.float64); meta=json.loads((DATA/'metadata.json').read_text()); pp=np.load(DATA/'pseudobulk.npz',allow_pickle=True); ident,idlabels=v.make_ident(meta,pp['pathways_full'].astype(np.float64),panel); im={x:i for i,x in enumerate(idlabels)}; cid=np.asarray([im[x] for x in labels]); gem=gemall[source]; batches=np.unique(gem); bm={int(b):j for j,b in enumerate(batches)}; bo=np.zeros((len(gem),len(batches))); [bo.__setitem__((i,bm[int(b)]),1.) for i,b in enumerate(gem)]; cref=(labels=='CONTROL')&((source%2)==0); control_mean=gene[cref].mean(0); return labels,source,ident,idlabels,cid,gem,batches,bo,gene,control_mean,valres

def fit_mlp(X,Y,skip,seed,epochs,Xq=None):
 import torch
 from torch import nn
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 class Net(nn.Module):
  def __init__(self):
   super().__init__(); self.t=nn.Sequential(nn.Linear(X.shape[1],256),nn.GELU(),nn.Linear(256,256),nn.GELU(),nn.Linear(256,Y.shape[1]));self.s=nn.Linear(X.shape[1],Y.shape[1],bias=False) if skip else None
  def forward(self,x):return self.t(x)+(self.s(x) if self.s is not None else 0.)
 net=Net().cuda(); opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4);tx=torch.tensor(X,dtype=torch.float32,device='cuda');ty=torch.tensor(Y,dtype=torch.float32,device='cuda')
 for _ in range(epochs):opt.zero_grad(); loss=((net(tx)-ty)**2).mean();loss.backward();opt.step()
 with torch.no_grad():
  pred=net(torch.tensor(X if Xq is None else Xq,dtype=torch.float32,device='cuda')).cpu().numpy()
 npar=sum(p.numel() for p in net.parameters());del net,tx,ty,opt;torch.cuda.empty_cache();return pred,npar

def main():
 import torch; assert torch.cuda.is_available(); labels,source,ident,idlabels,cid,gem,batches,bo,gene,control_mean,valres=prepare(); rows=[]
 for sr,vr in zip(SPLITS,valres['seeds']):
  rng=np.random.default_rng(sr); ids=np.flatnonzero(idlabels!='CONTROL'); order=ids.copy();rng.shuffle(order);ntr,nva=int(.7*len(order)),int(.15*len(order));trlabs=set(idlabels[order[:ntr]].tolist())|{'CONTROL'};vallabs=set(idlabels[order[ntr:ntr+nva]].tolist());testlabs=set(idlabels[order[ntr+nva:]].tolist()); fitids=np.flatnonzero(np.isin(idlabels,list(trlabs))); testmask=np.isin(labels,list(testlabs)); gx=[];gy=[]
  train_cell_mask=np.isin(labels,list(trlabs)); target_train_full=np.zeros_like(gene); target_train_full[train_cell_mask]=gene[train_cell_mask]-control_mean
  for ii in fitids:
   for b in batches:
    m=(cid==ii)&(gem==b)
    if m.sum()>=2:gx.append(np.r_[ident[ii],bo[np.flatnonzero(m)[0]]]);gy.append(target_train_full[m].mean(0))
  gx=np.asarray(gx);gy=np.asarray(gy);q=np.c_[ident[cid],bo];raw_sd=gx.std(0);constant=(raw_sd==0);mu,sd=gx.mean(0),raw_sd+1e-3;ym,ys=gy.mean(0),gy.std(0)+1e-3;gxz=(gx-mu)/sd;qz=(q-mu)/sd;gxz[:,constant]=0.;qz[:,constant]=0.;gyz=(gy-ym)/ys
  conf=min(vr['models'].items(),key=lambda kv:kv[1]['validation']['rmse']); key, cfg=conf; fam=cfg['family']; ts=int(cfg['training_seed']); ep=int(cfg['best_epoch']); skip=fam=='mlp_linear_skip'; rp=ridge(gxz,gy,qz,float(vr['ridge_alpha'])); mp,_=fit_mlp(gxz,gyz,skip,ts,ep,qz);mp=mp*ys+ym; y=gene[testmask]-control_mean; preds={'ridge':rp[testmask],'nonlinear':mp[testmask]}; labs=labels[testmask]; points={k:metric(y,p) for k,p in preds.items()}; per={k:perout(y,p) for k,p in preds.items()}; boot=bootstrap(y,preds,labs,sr); delta=boot['nonlinear']-boot['ridge']; rows.append({'split_seed':sr,'split_hash':vr['split_hash'],'selected_family':fam,'training_seed':ts,'selected_epoch':ep,'ridge_alpha':float(vr['ridge_alpha']),'parameter_count':int(cfg['parameter_count']),'n_fit_group_rows':len(gx),'n_test_cells':len(y),'n_test_identities':len(np.unique(labs)),'n_train_constant_features':int(constant.sum()),'test':points,'per_output_512':per,'paired_bootstrap_nonlinear_minus_ridge':{'B':B,'unit':'held-out identity cluster','ci95':np.quantile(delta,[.025,.975],axis=0).tolist(),'mean':delta.mean(0).tolist(),'point':{m:points['nonlinear'][m]-points['ridge'][m] for m in ['rmse','pearson','spearman']}}});print(sr,rows[-1]['selected_family'],points,flush=True);del gx,gy;gc.collect()
 out={'protocol':'corrected_locked_confirmation_matched_nonlinear_878_to_512','selection':'family, training seed, epoch and ridge alpha copied from corrected validation-only minima; models fit on training identities only and scored once on locked confirmation identities','feature_contract':'878=512 panel genes+256 pathways+102 components+8 gemgroup','output_dim':512,'bootstrap':{'B':B,'unit':'held-out identity cluster'},'seeds':rows,'input_hashes':{'script':sha(__file__),'validation':sha(OUT/'matched_nonlinear_validation_corrected.json'),'metadata':sha(DATA/'metadata.json'),'pseudobulk':sha(DATA/'pseudobulk.npz'),'cell_panel':sha(DATA/'cell_panel.npz'),'raw_h5ad':sha(RAW)},'protocol_hash':sha(ROOT/'experiments/matched_nonlinear_corrected_locked_confirmation_protocol_20261008.md')};(OUT/'matched_nonlinear_corrected_locked_confirmation.json').write_text(json.dumps(out,indent=2));print('WROTE corrected locked confirmation')
if __name__=='__main__':main()
