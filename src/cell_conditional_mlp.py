"""Validation-only pathway-conditioned cell response MLP.

The script is deliberately usable offline on a GPU job.  It never reads
confirmation cells for model selection; the confirmation branch is enabled
only with --confirm after a validation improvement has been recorded.
"""
from pathlib import Path
import argparse, json, random
import numpy as np
import torch
from torch import nn
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed'; OUT=ROOT/'results'
def metrics(y,p):
    return {'rmse':float(np.sqrt(mean_squared_error(y,p))), 'pearson':float(pearsonr(y.ravel(),p.ravel()).statistic), 'spearman':float(spearmanr(y.ravel(),p.ravel()).statistic), 'n_cells':int(len(y))}
class Decoder(nn.Module):
    def __init__(self,n_in,n_out,hidden=256):
        super().__init__(); self.net=nn.Sequential(nn.Linear(n_in,hidden),nn.LayerNorm(hidden),nn.GELU(),nn.Dropout(.1),nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,n_out))
    def forward(self,x): return self.net(x)
def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=300); ap.add_argument('--seed',type=int,default=11); ap.add_argument('--confirm',action='store_true'); ap.add_argument('--device',default='cuda'); ap.add_argument('--max-cells',type=int,default=0); args=ap.parse_args(); set_seed(args.seed)
    device=torch.device(args.device if args.device=='cpu' or torch.cuda.is_available() else 'cpu')
    z=np.load(DATA/'cell_panel.npz',allow_pickle=True); X=z['X'].astype(np.float32); labs=np.asarray(z['labels'],dtype=str); meta=json.loads((DATA/'metadata.json').read_text()); P=np.load(DATA/'pseudobulk.npz',allow_pickle=True)['pathways_full'].astype(np.float32); vocab=np.asarray(meta['pathway_vocab']); genes=np.asarray(meta['genes']); gi={g:i for i,g in enumerate(vocab)}; id_labs=np.asarray(meta['labels']['norman'],dtype=str)
    if args.max_cells and args.max_cells < len(X):
        keep=[]
        per=max(1,args.max_cells//max(1,len(np.unique(labs))))
        for l in np.unique(labs): keep.extend(np.flatnonzero(labs==l)[:per].tolist())
        keep=np.asarray(keep[:args.max_cells],dtype=int); X=X[keep]; labs=labs[keep]
    F=np.zeros((len(id_labs),len(vocab)),np.float32)
    for i,l in enumerate(id_labs):
        if l!='CONTROL':
            for t in str(l).replace('/','+').replace('-','_').split('+'):
                if t in gi:F[i,gi[t]]=1
    Fg=F[:,[gi[g] for g in genes]]; Fp=F@P; A=np.c_[Fg,Fp]; ids=np.flatnonzero(id_labs!='CONTROL'); rng=np.random.default_rng(11); rng.shuffle(ids); ntr=int(.70*len(ids)); nva=int(.15*len(ids)); tr_ids=ids[:ntr]; va_ids=ids[ntr:ntr+nva]; te_ids=ids[ntr+nva:]
    train_set=set(id_labs[tr_ids].tolist()+['CONTROL']); val_set=set(id_labs[va_ids]); test_set=set(id_labs[te_ids]); id_to_i={l:i for i,l in enumerate(id_labs)}; cell_id=np.asarray([id_to_i[l] for l in labs]); cell_train=np.isin(labs,list(train_set)); cell_val=np.isin(labs,list(val_set)); cell_test=np.isin(labs,list(test_set))
    mu=A[tr_ids].mean(0); sd=A[tr_ids].std(0)+1e-3; As=(A-mu)/sd; ym=X[cell_train].mean(0); ys=X[cell_train].std(0)+1e-3; Y=(X-ym)/ys
    tx=torch.tensor(As[cell_id][cell_train],dtype=torch.float32,device=device); ty=torch.tensor(Y[cell_train],dtype=torch.float32,device=device); vx=torch.tensor(As[cell_id][cell_val],dtype=torch.float32,device=device); vy=torch.tensor(Y[cell_val],dtype=torch.float32,device=device); model=Decoder(tx.shape[1],X.shape[1]).to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-3); best=float('inf'); best_state=None; stall=0; rng=np.random.default_rng(args.seed); batch=1024; curve=[]
    for ep in range(args.epochs):
        model.train(); order=rng.permutation(len(tx)); losses=[]
        for start in range(0,len(order),batch):
            ii=torch.tensor(order[start:start+batch],dtype=torch.long,device=device); opt.zero_grad(); pred=model(tx[ii]); loss=((pred-ty[ii])**2).mean(); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        model.eval();
        with torch.no_grad(): vl=float(((model(vx)-vy)**2).mean().cpu())
        curve.append({'epoch':ep+1,'train_mse':float(np.mean(losses)),'val_mse':vl})
        if vl<best-1e-6: best=vl; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stall=0
        else: stall+=1
        if stall>=40: break
    if best_state:model.load_state_dict(best_state)
    model.eval();
    with torch.no_grad(): pred=model(torch.tensor(As[cell_id],dtype=torch.float32,device=device)).cpu().numpy()*ys+ym
    control_mean=X[labs=='CONTROL'].mean(0); target_rel=X-control_mean; pred_rel=pred-control_mean
    out={'seed':args.seed,'device':str(device),'epochs_ran':len(curve),'best_val_mse_scaled':best,'split':{'train_identities':sorted(train_set),'validation_identities':sorted(val_set),'test_identities':sorted(test_set)},'validation':metrics(target_rel[cell_val],pred_rel[cell_val]),'validation_absolute':metrics(X[cell_val],pred[cell_val]),'validation_identity_mean':metrics(np.vstack([target_rel[labs==l].mean(0) for l in sorted(val_set)]),np.vstack([pred_rel[labs==l].mean(0) for l in sorted(val_set)])),'train':metrics(target_rel[cell_train],pred_rel[cell_train]),'curve':curve}
    if args.confirm: out['confirmation']=metrics(target_rel[cell_test],pred_rel[cell_test]); out['confirmation_absolute']=metrics(X[cell_test],pred[cell_test])
    out_path=OUT / (f'cell_conditional_mlp_seed{args.seed}_'+('confirm' if args.confirm else 'validation')+'.json'); out_path.write_text(json.dumps(out,indent=2)); print(json.dumps({k:v for k,v in out.items() if k!='curve'},indent=2))
if __name__=='__main__': main()
