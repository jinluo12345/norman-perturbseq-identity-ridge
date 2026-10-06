"""Reactome structure controls on the frozen split-11 identity protocol.

The nulls preserve the complete multiset of gene-to-module membership patterns.
Permuting those patterns across genes therefore preserves module sizes, gene
degree, pairwise module overlap and parent/child overlap while removing the
original assignment of pathways to perturbation targets.
"""
from pathlib import Path
import json
import numpy as np
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed'; OUT=ROOT/'results'

def toks(label):
    if label in {'CONTROL','NAN','*'}: return []
    return [x for x in str(label).replace('/','+').replace('-','_').split('+') if x and x not in {'ONLY','MOD'}]

def feature_rows(labels,vocab,P):
    gi={g:i for i,g in enumerate(vocab)}; rows=[]
    for lab in labels:
        ids=[gi[t] for t in toks(lab) if t in gi]
        rows.append(P[ids].sum(0) if ids else np.zeros(P.shape[1],dtype=np.float32))
    return np.asarray(rows,dtype=np.float32)

def met(y,p):
    return {'rmse':float(np.sqrt(mean_squared_error(y,p))), 'pearson':float(pearsonr(y.ravel(),p.ravel()).statistic), 'spearman':float(spearmanr(y.ravel(),p.ravel()).statistic), 'n_profiles':int(len(y))}

def bootstrap(y,a,b,B=3000,seed=20261007):
    rng=np.random.default_rng(seed); ix=rng.integers(0,len(y),(B,len(y))); yy=y[ix]; aa=a[ix]; bb=b[ix]
    dr=np.sqrt(np.mean((yy-aa)**2,(1,2)))-np.sqrt(np.mean((yy-bb)**2,(1,2)))
    def pr(x):
        q=x.reshape(B,-1); t=yy.reshape(B,-1); q-=q.mean(1,keepdims=True); t-=t.mean(1,keepdims=True); return np.sum(q*t,1)/np.sqrt(np.sum(q*q,1)*np.sum(t*t,1))
    dp=pr(aa)-pr(bb)
    def ci(x): return {'estimate':float(x.mean()),'ci95':[float(np.quantile(x,.025)),float(np.quantile(x,.975))]}
    return {'rmse_null_minus_observed':ci(dr),'pearson_null_minus_observed':ci(dp),'n_bootstrap':B}

def union_find_clusters(P, threshold=.50):
    n=P.shape[1]; parent=list(range(n)); sizes=(P>0).sum(0); sets=[set(np.flatnonzero(P[:,j])) for j in range(n)]
    def find(a):
        while parent[a]!=a: parent[a]=parent[parent[a]]; a=parent[a]
        return a
    def join(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[b]=a
    for i in range(n):
        for j in range(i):
            inter=len(sets[i]&sets[j]); union=sizes[i]+sizes[j]-inter
            if union and inter/union>=threshold: join(i,j)
    groups={}
    for j in range(n): groups.setdefault(find(j),[]).append(j)
    return list(groups.values())

def main():
    z=np.load(DATA/'pseudobulk.npz',allow_pickle=True); meta=json.loads((DATA/'metadata.json').read_text()); labels=np.array(meta['labels']['norman']); y=z['y_norman'].astype(np.float32); c=np.where(labels=='CONTROL')[0]; y-=y[c].mean(0); vocab=np.array(meta['pathway_vocab']); P=z['pathways_full'].astype(np.float32); genes=np.array(meta['genes']); tr=np.array(json.load(open(ROOT/'results/model_results.json'))['runs'][0]['split']['train']); te=np.array(json.load(open(ROOT/'results/model_results.json'))['runs'][0]['split']['test'])
    F=feature_rows(labels,vocab,P); mu=F[tr].mean(0); sd=F[tr].std(0)+1e-3; F=(F-mu)/sd; obs=Ridge(alpha=.1).fit(F[tr],y[tr]); po=obs.predict(F[te]); result={'observed':met(y[te],po),'n_pathways':int(P.shape[1]),'n_vocab_genes':int(P.shape[0]),'split_seed':11}
    null=[]; rng=np.random.default_rng(91011)
    for k in range(100):
        perm=rng.permutation(P.shape[0]); Pn=P[perm]; Fn=feature_rows(labels,vocab,Pn); mn=Fn[tr].mean(0); sn=Fn[tr].std(0)+1e-3; Fn=(Fn-mn)/sn; mod=Ridge(alpha=.1).fit(Fn[tr],y[tr]); pm=mod.predict(Fn[te]); q=met(y[te],pm); null.append({'seed':k,**q})
    result['pattern_permutation_null']={'preserves':'module sizes, per-gene degree, pairwise module overlap and nested parent-child overlap','runs':null,'pearson_mean':float(np.mean([q['pearson'] for q in null])),'pearson_sd':float(np.std([q['pearson'] for q in null],ddof=1)),'paired_vs_observed':bootstrap(y[te],np.stack([feature_rows(labels,vocab,P[rng.permutation(P.shape[0])])[te] for _ in range(1)]),np.zeros((1,1))) if False else None}
    # A compact hierarchy-collapsed model merges strongly overlapping modules.
    groups=union_find_clusters(P,.50); Pc=np.zeros((P.shape[0],len(groups)),np.float32)
    for j,g in enumerate(groups): Pc[:,j]=P[:,g].max(1)
    Fc=feature_rows(labels,vocab,Pc); mc=Ridge(alpha=.1).fit(Fc[tr],y[tr]); result['hierarchy_collapsed']={'jaccard_threshold':.50,'n_clusters':len(groups),**met(y[te],mc.predict(Fc[te]))}
    (OUT/'reactome_structure_nulls.json').write_text(json.dumps(result,indent=2)); print(json.dumps({'observed':result['observed'],'null_mean':result['pattern_permutation_null']['pearson_mean'],'null_sd':result['pattern_permutation_null']['pearson_sd'],'collapsed':result['hierarchy_collapsed']},indent=2))
if __name__=='__main__': main()
