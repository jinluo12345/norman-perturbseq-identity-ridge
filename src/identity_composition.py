"""Identity-composition stratification for the fixed Norman confirmation split."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

ROOT=Path(__file__).resolve().parents[1]
META=json.loads((ROOT/'data/processed/metadata.json').read_text())
labels=np.asarray(META['labels']['norman'],dtype=str)
diag=json.loads((ROOT/'results/model_results.json').read_text())['runs'][0]
tr=np.asarray(diag['split']['train'],dtype=int); te=np.asarray(diag['split']['test'],dtype=int)
nd=json.loads((ROOT/'results/neat_diagnostics.json').read_text())['runs'][0]
y=np.asarray(nd['y_test']); pn=np.asarray(nd['pred_neat']); pr=np.asarray(nd['pred_ridge'])
assert len(te)==len(y)==len(pn)==len(pr)

def toks(label):
    if label in {'CONTROL','NAN','*'}: return []
    return [x for x in str(label).replace('/','+').replace('-','_').split('+') if x and x not in {'ONLY','MOD'}]

train_labels=set(labels[tr])-{'CONTROL'}
train_genes=set(g for l in train_labels for g in toks(l))
tv=np.asarray(META['pathway_vocab'],dtype=str)
P=np.load(ROOT/'data/processed/pseudobulk.npz',allow_pickle=True)['pathways_full'].astype(float)
gidx={g:i for i,g in enumerate(tv)}
def active(label):
    inds=[gidx[g] for g in toks(label) if g in gidx]
    return tuple(np.flatnonzero(P[inds].sum(axis=0)>0).tolist()) if inds else tuple()
train_path_sigs={active(l) for l in labels[tr] if l!='CONTROL'}
rows=[]
for j,idx in enumerate(te):
    lab=labels[idx]; ts=toks(lab); seen=[g in train_genes for g in ts]; all_seen=bool(ts) and all(seen); any_unseen=bool(ts) and not all_seen
    if len(ts)==1 and not all_seen: category='novel single gene'
    elif len(ts)>1 and any_unseen: category='combination with unseen component'
    elif len(ts)>1 and all_seen: category='combination with seen components'
    elif len(ts)==1 and all_seen: category='seen single gene'
    else: category='unresolved identity'
    sig=active(lab)
    rows.append({'identity':lab,'n_components':len(ts),'components':' + '.join(ts),'all_components_seen_in_train':all_seen,'any_component_unseen':any_unseen,'composition_category':category,'pathway_signature_seen_in_train':sig in train_path_sigs,'n_active_pathways':len(sig),'test_order':j})
df=pd.DataFrame(rows)
def metrics(a,b):
    if len(a)==0: return {'n':0,'rmse':None,'pearson':None,'spearman':None}
    aa=np.asarray(a); bb=np.asarray(b)
    return {'n':int(len(aa)),'rmse':float(np.sqrt(mean_squared_error(aa,bb))),'pearson':float(pearsonr(aa.ravel(),bb.ravel()).statistic) if np.std(bb)>1e-9 else None,'spearman':float(spearmanr(aa.ravel(),bb.ravel()).statistic) if np.std(bb)>1e-9 else None}
out={'split_seed':int(diag['split_seed']),'n_test':int(len(te)),'train_component_gene_count':int(len(train_genes)),'train_pathway_signature_count':int(len(train_path_sigs)),'identity_rows':rows,'groups':{}}
for col in ['composition_category','all_components_seen_in_train','pathway_signature_seen_in_train']:
    for key,g in df.groupby(col,dropna=False):
        ix=g.test_order.to_numpy(); out['groups'][f'{col}={key}']={'definition':str(key),'metrics_neat':metrics(y[ix],pn[ix]),'metrics_ridge':metrics(y[ix],pr[ix]),'identities':g.identity.tolist()}
df.to_csv(ROOT/'results/identity_composition_stratification.csv',index=False)
(ROOT/'results/identity_composition_stratification.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
