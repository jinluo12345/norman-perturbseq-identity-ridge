"""Validation-only matched nonlinear baseline for repaired Norman contract.
No locked-test responses are read for model selection or scored here.
"""
from pathlib import Path
import hashlib,json,os,random,gc
import numpy as np
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import mean_squared_error

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed'; RAW=ROOT/'data/raw/NormanWeissman2019_filtered.h5ad'; OUT=ROOT/'results'; CURVES=OUT/'matched_nonlinear_corrected_curves'; CURVES.mkdir(exist_ok=True)
SEEDS=[11,22,33,44,55]; TRAIN_SEEDS=[101,202,303]; ALPHAS=[.1,1.,3.,10.,30.,100.,300.]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metric(y,p):
 a,b=np.asarray(y).ravel(),np.asarray(p).ravel(); return {'rmse':float(np.sqrt(mean_squared_error(a,b))),'pearson':float(pearsonr(a,b).statistic),'spearman':float(spearmanr(a,b).statistic),'n_cells':int(len(y))}
def clean(x): return str(x.decode() if isinstance(x,(bytes,np.bytes_)) else x).upper().strip()
def ridge_predictions(xtr,ytr,xq,alphas):
 ym,xm=ytr.mean(0),xtr.mean(0); u,s,vt=np.linalg.svd(xtr-xm,full_matrices=False); proj=u.T@(ytr-ym); return {float(a):(xq-xm)@(vt.T@((s/(s*s+a))[:,None]*proj))+ym for a in alphas}

def load_raw():
 import h5py
 with h5py.File(RAW,'r') as h:
  x=h['X']; shape=tuple(int(v) for v in x.attrs['shape']); indptr=np.asarray(x['indptr'][:],np.int64); raw=np.asarray([clean(x) for x in h['var/_index'][:]],object); gemall=np.asarray(h['obs/gemgroup'][:],np.int64); rr=np.asarray(x['indices'][:],np.int64); vv=np.asarray(x['data'][:],np.float64); lib=np.bincount(rr,weights=vv,minlength=shape[0]);
 return indptr,raw,gemall,lib

def extract(indptr,raw,source,lib,panel):
 import h5py
 from scipy import sparse
 cmap={g:i for i,g in enumerate(raw)}; cols=np.asarray([cmap[g] for g in panel]);
 with h5py.File(RAW,'r') as h:
  x=h['X']; dat=np.asarray(x['data'][:],np.float32); idx=np.asarray(x['indices'][:],np.int32); mat=sparse.csc_matrix((dat,idx,indptr),shape=(len(lib),len(raw)))[:,cols].tocsr()[source]
 mat=mat.multiply((1e4/np.maximum(lib[source],1.))[:,None]).log1p(); return mat.toarray().astype(np.float32)

def make_ident(meta,pathways,panel):
 labs=np.asarray(meta['labels']['norman'],str); V=np.asarray(meta['pathway_vocab'],object); vi={clean(g):i for i,g in enumerate(V)}; F=np.zeros((len(labs),len(V)),np.float64)
 for i,l in enumerate(labs):
  if l!='CONTROL':
   for t in str(l).replace('/','+').replace('-','_').split('+'):
    j=vi.get(clean(t));
    if j is not None:F[i,j]=1
 active=np.flatnonzero(F.sum(0)>0); gix=[vi[g] for g in panel if g in vi]; return np.c_[F[:,gix],F@pathways,F[:,active]],labs

class MLP:
 def __init__(self,d,o,skip=False):
  import torch
  from torch import nn
  self.skip=skip; self.net=nn.Sequential(nn.Linear(d,256),nn.GELU(),nn.Linear(256,256),nn.GELU(),nn.Linear(256,o)); self.lin=nn.Linear(d,o) if skip else None
 def module(self):
  import torch.nn as nn
  return nn.Module() # unused

def train_model(X,Y,VX,VY,ys,seed,skip):
 import torch
 from torch import nn
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
 class Net(nn.Module):
  def __init__(self):
   super().__init__(); self.trunk=nn.Sequential(nn.Linear(X.shape[1],256),nn.GELU(),nn.Linear(256,256),nn.GELU(),nn.Linear(256,Y.shape[1])); self.skip=nn.Linear(X.shape[1],Y.shape[1],bias=False) if skip else None
  def forward(self,z):
   q=self.trunk(z); return q+(self.skip(z) if self.skip is not None else 0.)
 net=Net().cuda(); opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4)
 tx=torch.as_tensor(X,dtype=torch.float32,device='cuda'); ty=torch.as_tensor(Y,dtype=torch.float32,device='cuda'); vx=torch.as_tensor(VX,dtype=torch.float32,device='cuda'); vy=torch.as_tensor(VY,dtype=torch.float32,device='cuda'); best=float('inf'); beststate=None; bestepoch=None; stall=0; curve=[]
 for ep in range(1,301):
  net.train(); opt.zero_grad(); loss=(((net(tx)-ty)*torch.as_tensor(ys,dtype=torch.float32,device='cuda'))**2).mean(); loss.backward(); opt.step(); net.eval()
  with torch.no_grad(): vl=float((((net(vx)-vy)*torch.as_tensor(ys,dtype=torch.float32,device='cuda'))**2).mean().sqrt().cpu())
  curve.append({'epoch':ep,'train_mse':float(loss.detach().cpu()),'val_raw_rmse':vl})
  if vl<best-1e-6: best=vl; bestepoch=ep; stall=0; beststate={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
  else: stall+=1
  if stall>=40: break
 net.load_state_dict(beststate); net.eval();
 with torch.no_grad(): pred=net(vx).cpu().numpy()
 nparam=sum(p.numel() for p in net.parameters()); del net,tx,ty,vx,vy,opt; torch.cuda.empty_cache(); return pred,curve,nparam,bestepoch,best,beststate

def main():
 import torch
 assert torch.cuda.is_available(), 'GPU required for this job'
 z=np.load(DATA/'cell_panel.npz',allow_pickle=True); labels=np.asarray(z['labels'],str); source=np.asarray(z['source_row'],int)
 repaired=json.loads((OUT/'repaired_split_local_controls.json').read_text()); panel=np.asarray(repaired['seeds'][0]['panel_genes'][:512],object); assert len(panel)==512 and len(set(panel))==512
 indptr,raw,gemall,lib=load_raw(); Xgene=extract(indptr,raw,source,lib,panel).astype(np.float64); meta=json.loads((DATA/'metadata.json').read_text()); pp=np.load(DATA/'pseudobulk.npz',allow_pickle=True); ident,idlabels=make_ident(meta,pp['pathways_full'].astype(np.float64),panel); idmap={x:i for i,x in enumerate(idlabels)}; cid=np.asarray([idmap[x] for x in labels]); gem=gemall[source]; batches=np.unique(gem); bm={int(b):j for j,b in enumerate(batches)}; bo=np.zeros((len(gem),len(batches))); [bo.__setitem__((i,bm[int(b)]),1.) for i,b in enumerate(gem)]
 assert ident.shape[1]+len(batches)==878, ident.shape
 cref=(labels=='CONTROL')&((source%2)==0); target=Xgene-Xgene[cref].mean(0); rows=[]
 for split_seed in SEEDS:
  rng=np.random.default_rng(split_seed); ids=np.flatnonzero(idlabels!='CONTROL'); order=ids.copy(); rng.shuffle(order); ntr,nva=int(.7*len(order)),int(.15*len(order)); trlabs=set(idlabels[order[:ntr]].tolist())|{'CONTROL'}; val_labs=set(idlabels[order[ntr:ntr+nva]].tolist()); train_ids=np.flatnonzero(np.isin(idlabels,list(trlabs))); valmask=np.isin(labels,list(val_labs));
  gx=[]; gy=[]
  for ii in train_ids:
   for b in batches:
    m=(cid==ii)&(gem==b)
    if m.sum()>=2: gx.append(np.r_[ident[ii],bo[np.flatnonzero(m)[0]]]); gy.append(target[m].mean(0))
  gx=np.asarray(gx); gy=np.asarray(gy); q=np.c_[ident[cid],bo]; assert gx.shape[1]==878 and gy.shape[1]==512
  mu,sd=gx.mean(0),gx.std(0)+1e-3; gxz=(gx-mu)/sd; qz=(q-mu)/sd; constant=(gx.std(0)==0); gxz[:,constant]=0.; qz[:,constant]=0.; ym,ys=gy.mean(0),gy.std(0)+1e-3; gyz=(gy-ym)/ys; valy=target[valmask]; valx=qz[valmask]
  ridgegrid=ridge_predictions(gxz,gy,valx,ALPHAS); rbest=min(((metric(valy,p),a) for a,p in ridgegrid.items()),key=lambda x:x[0]['rmse']);
  rr={'split_seed':split_seed,'split_hash':hashlib.sha256(np.asarray(order,np.int64).tobytes()).hexdigest(),'n_train_group_rows':len(gx),'n_train_constant_features':int(constant.sum()),'validation_nonzero_train_constant_features':int(np.any(np.abs(q[valmask][:,constant]-mu[constant])>0,axis=0).sum()),'train_constant_indices':np.flatnonzero(constant).tolist(),'n_validation_cells':int(valmask.sum()),'input_dim':878,'output_dim':512,'ridge_alpha':rbest[1],'ridge_validation':rbest[0],'models':{}}
  for fam,skip in [('mlp',False),('mlp_linear_skip',True)]:
   for ts in TRAIN_SEEDS:
    pred,curve,npv,ep,best,state=train_model(gxz,gyz,valx,(valy-ym)/ys,ys,ts,skip); pred=pred*ys+ym; m=metric(valy,pred); key=f'{fam}_seed{ts}'; rr['models'][key]={'family':fam,'training_seed':ts,'validation':m,'best_epoch':ep,'best_raw_val_rmse':best,'parameter_count':npv}; (CURVES/f'{split_seed}_{fam}_{ts}.json').write_text(json.dumps(curve)); np.savez_compressed(CURVES/f'{split_seed}_{fam}_{ts}_validation_predictions.npz',prediction=pred,labels=labels[valmask],source_row=source[valmask]); torch.save({'state_dict':state,'family':fam,'training_seed':ts,'best_epoch':ep,'x_mean':mu,'x_sd':sd,'constant_feature_mask':constant,'y_mean':ym,'y_sd':ys},CURVES/f'{split_seed}_{fam}_{ts}.pt')
  rows.append(rr); print(split_seed,rr['ridge_validation'],{k:v['validation'] for k,v in rr['models'].items()},flush=True)
 out={'protocol':'validation_only_matched_nonlinear_878_to_512_corrected_raw_rmse','selection':'ridge alpha and MLP best epoch by validation cell-level RMSE only; no test metrics computed','feature_contract':'512 frozen panel genes + 256 pathway scores + 102 active component indicators + 8 observed gemgroup one-hot = 878','target_contract':'512 panel-gene control-relative response using CONTROL/source_row parity reference','split_seeds':SEEDS,'training_seeds':TRAIN_SEEDS,'architectures':['878-256-GELU-256-GELU-512','878-256-GELU-256-GELU-512 plus learned 878-to-512 linear skip'],'optimizer':'AdamW lr=1e-3 weight_decay=1e-4, full batch, max 300 epochs, patience 40','batches':batches.tolist(),'seeds':rows,'input_hashes':{'script':sha(__file__),'repaired_result':sha(OUT/'repaired_split_local_controls.json'),'metadata':sha(DATA/'metadata.json'),'pseudobulk':sha(DATA/'pseudobulk.npz'),'cell_panel':sha(DATA/'cell_panel.npz'),'raw_h5ad':sha(RAW)},'protocol_hash':sha(ROOT/'experiments/matched_nonlinear_corrected_protocol_20261007.md')}
 (OUT/'matched_nonlinear_validation_corrected.json').write_text(json.dumps(out,indent=2)); print(json.dumps({'path':str(OUT/'matched_nonlinear_validation_corrected.json'),'splits':len(rows)}))
if __name__=='__main__': main()
