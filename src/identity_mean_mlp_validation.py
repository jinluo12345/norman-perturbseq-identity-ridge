"""Validation-only nonlinear decoder trained on identity means."""
from pathlib import Path
import json, random
import numpy as np
import torch
from torch import nn
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed'; OUT=ROOT/'results'
def metric(y,p): return {'rmse':float(np.sqrt(mean_squared_error(y,p))),'pearson':float(pearsonr(y.ravel(),p.ravel()).statistic),'spearman':float(spearmanr(y.ravel(),p.ravel()).statistic),'n_identities':int(len(y))}
class Net(nn.Module):
    def __init__(self,d,o): super().__init__(); self.net=nn.Sequential(nn.Linear(d,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,128),nn.GELU(),nn.Linear(128,o))
    def forward(self,x): return self.net(x)
def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=500); ap.add_argument('--seed',type=int,default=11); ap.add_argument('--device',default='cuda'); args=ap.parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device=torch.device(args.device if args.device=='cpu' or torch.cuda.is_available() else 'cpu'); z=np.load(DATA/'pseudobulk.npz',allow_pickle=True); m=json.loads((DATA/'metadata.json').read_text()); labs=np.asarray(m['labels']['norman'],str); y=z['y_norman'].astype(np.float32); ctrl=np.flatnonzero(labs=='CONTROL'); y-=y[ctrl].mean(0); P=z['pathways_full'].astype(np.float32); v=np.asarray(m['pathway_vocab']); g=np.asarray(m['genes']); gi={x:i for i,x in enumerate(v)}; F=np.zeros((len(labs),len(v)),np.float32)
    for i,l in enumerate(labs):
        if l!='CONTROL':
            for t in str(l).replace('/','+').replace('-','_').split('+'):
                if t in gi:F[i,gi[t]]=1
    X=np.c_[F[:,[gi[x] for x in g]],F@P]; ids=np.flatnonzero(labs!='CONTROL'); rng=np.random.default_rng(11);rng.shuffle(ids);ntr=int(.7*len(ids));nva=int(.15*len(ids));tr=np.r_[ids[:ntr],ctrl];va=ids[ntr:ntr+nva];te=ids[ntr+nva:];mu=X[tr].mean(0);sd=X[tr].std(0)+1e-3;Xs=(X-mu)/sd;ym=y[tr].mean(0);ys=y[tr].std(0)+1e-3;Ys=(y-ym)/ys; tx=torch.tensor(Xs[tr],dtype=torch.float32,device=device);ty=torch.tensor(Ys[tr],dtype=torch.float32,device=device);vx=torch.tensor(Xs[va],dtype=torch.float32,device=device);vy=torch.tensor(Ys[va],dtype=torch.float32,device=device); net=Net(tx.shape[1],y.shape[1]).to(device); opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-3);best=1e9;state=None;stall=0;curve=[]
    for ep in range(args.epochs):
        net.train();opt.zero_grad();loss=((net(tx)-ty)**2).mean();loss.backward();opt.step();net.eval();
        with torch.no_grad():vl=float(((net(vx)-vy)**2).mean().cpu())
        curve.append({'epoch':ep+1,'train_mse':float(loss.detach().cpu()),'val_mse':vl})
        if vl<best-1e-7:best=vl;state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};stall=0
        else:stall+=1
        if stall>=60:break
    if state:net.load_state_dict(state)
    net.eval();
    with torch.no_grad():pred=net(torch.tensor(Xs,dtype=torch.float32,device=device)).cpu().numpy()*ys+ym
    out={'device':str(device),'seed':args.seed,'epochs_ran':len(curve),'best_val_mse_scaled':best,'split':{'train':tr.tolist(),'validation':va.tolist(),'test':te.tolist()},'validation':metric(y[va],pred[va]),'train':metric(y[tr],pred[tr]),'curve':curve}
    (OUT/f'identity_mean_mlp_seed{args.seed}_validation.json').write_text(json.dumps(out,indent=2)); print(json.dumps({k:v for k,v in out.items() if k!='curve'},indent=2))
if __name__=='__main__':main()
