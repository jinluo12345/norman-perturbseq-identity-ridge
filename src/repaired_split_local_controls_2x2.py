"""Exact aggregation x metadata factorial built on the repaired protocol.

This companion keeps the frozen panel, source-row-parity control reference,
identity splits, alpha grid and identity-cluster bootstrap from
``repaired_split_local_controls.py``.  It adds the missing factorial cell:
identity-wide means replicated once per supported gemgroup, with the
gemgroup one-hot appended.  The replicated response is intentionally the
identity-wide mean, so the comparison separates aggregation from metadata
conditioning while retaining the same supported group rows as the candidate.
"""
from pathlib import Path
import hashlib, json, gc, os
import h5py
import numpy as np
from scipy import sparse
from scipy.stats import rankdata
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/NormanWeissman2019_filtered.h5ad"
DATA = ROOT / "data/processed"
OUT = ROOT / "results"
SEEDS = [int(x) for x in os.environ.get("REPAIRED_2X2_SEEDS", "11,22,33,44,55").split(",") if x.strip()]
ALPHAS = [0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
B = int(os.environ.get("REPAIRED_2X2_BOOTSTRAP_B", "1000"))

# Reuse audited implementations and constants without importing main().
import repaired_split_local_controls as base

def bootstrap_groups(y_groups, pred_groups, rng, B):
    yy = np.vstack(y_groups); ry = rankdata(yy.ravel(), method="average")
    ycuts = np.cumsum([x.size for x in y_groups])[:-1]; ryp = np.split(ry, ycuts)
    out = {}
    for name, groups in pred_groups.items():
        pp = np.vstack(groups); rp = rankdata(pp.ravel(), method="average")
        rpp = np.split(rp, np.cumsum([x.size for x in groups])[:-1]); arr=[]
        for g in range(len(groups)):
            yv,pv,yrr,prr=y_groups[g].ravel(),groups[g].ravel(),ryp[g],rpp[g]
            arr.append([len(yv),np.sum((yv-pv)**2),np.sum(yv),np.sum(pv),np.sum(yv*yv),np.sum(pv*pv),np.sum(yv*pv),np.sum(yrr),np.sum(prr),np.sum(yrr*yrr),np.sum(prr*prr),np.sum(yrr*prr)])
        a=np.asarray(arr,float); ix=rng.integers(0,len(a),size=(B,len(a))); s=a[ix].sum(1)
        n,sse,sy,sp,sy2,sp2,syp,ry1,rp1,ry2,rp2,ryp1=s.T
        pear=(syp-sy*sp/n)/np.sqrt(np.maximum((sy2-sy*sy/n)*(sp2-sp*sp/n),1e-30))
        spear=(ryp1-ry1*rp1/n)/np.sqrt(np.maximum((ry2-ry1*ry1/n)*(rp2-rp1*rp1/n),1e-30))
        out[name]=np.c_[np.sqrt(sse/n),pear,spear]
    return out

def main():
    shape,indptr,raw_genes,raw_labels,gem_all,lib=base.load_source()
    source_rows=base.sampled_rows(raw_labels)
    zold=np.load(DATA/"cell_panel.npz",allow_pickle=True); assert np.array_equal(source_rows,zold["source_row"])
    labels=np.asarray(zold["labels"],dtype=str); gem=gem_all[source_rows]
    meta=json.loads((DATA/"metadata.json").read_text()); pp=np.load(DATA/"pseudobulk.npz",allow_pickle=True)
    pathways=pp["pathways_full"].astype(np.float64); id_labels=np.asarray(meta["labels"]["norman"],dtype=str)
    ids=np.flatnonzero(id_labels!="CONTROL"); id_to_i={x:i for i,x in enumerate(id_labels)}; cell_id=np.asarray([id_to_i[x] for x in labels],int)
    orders=[]
    for seed in SEEDS:
        rng=np.random.default_rng(seed); order=ids.copy(); rng.shuffle(order); orders.append(order)
    fixed_meta=json.loads((ROOT/".tmp/metadata_fixed_degree_panel.json").read_text()); panel=np.asarray(fixed_meta["genes"][:512],dtype=object)
    batch_vals=np.unique(gem); bm={int(x):j for j,x in enumerate(batch_vals)}; bo=np.zeros((len(gem),len(batch_vals)),float)
    for i,b in enumerate(gem): bo[i,bm[int(b)]]=1
    X=base.extract_panel(indptr,raw_genes,source_rows,lib,panel).astype(np.float64); all_rows=[]
    for seed,order in zip(SEEDS,orders):
        ntr,nva=int(.70*len(order)),int(.15*len(order)); trlabs=set(id_labels[order[:ntr]].tolist())|{"CONTROL"}; val_labels=set(id_labels[order[ntr:ntr+nva]].tolist()); test_labels=sorted(set(id_labels[order[ntr+nva:]].tolist()))
        train_ids=np.flatnonzero(np.isin(id_labels,list(trlabs))); val_mask=np.isin(labels,list(val_labels)); test_mask=np.isin(labels,list(test_labels))
        control_ref_mask=(labels=="CONTROL")&((source_rows%2)==0); control_mean=X[control_ref_mask].mean(0); target=X-control_mean
        ident,_,_=base.feature_matrix(labels,panel,meta,pathways); grx,gry,gids,gcounts=[],[],[],{int(ii):0 for ii in train_ids}
        for ii in train_ids:
            for b in batch_vals:
                m=(cell_id==ii)&(gem==b)
                if m.sum()>=2: grx.append(ident[ii]); gry.append(target[m].mean(0)); gids.append((int(ii),int(b))); gcounts[int(ii)]+=1
        grx=np.asarray(grx); gry=np.asarray(gry); gbo=np.asarray([bo[np.flatnonzero((cell_id==ii)&(gem==b))[0]] for ii,b in gids])
        qid=ident[cell_id]; base_y=np.asarray([target[cell_id==ii].mean(0) for ii in train_ids]); base_x=ident[train_ids]
        rep=np.repeat(np.arange(len(train_ids)),[gcounts[int(ii)] for ii in train_ids]); wx=base_x[rep]; wy=base_y[rep]
        # The exact missing factorial cell: duplicate the identity-wide target
        # once per supported batch, then append that batch indicator.
        wide_batch_x=np.c_[np.asarray([ident[ii] for ii,b in gids]),gbo]
        wide_batch_y=np.asarray([target[cell_id==ii].mean(0) for ii,b in gids])
        designs={"identity_mean":(base_x,base_y,qid),"identity_weighted":(wx,wy,qid),"identity_wide_batch":(wide_batch_x,wide_batch_y,np.c_[qid,bo]),"group_masked":(grx,gry,qid),"candidate":(np.c_[grx,gbo],gry,np.c_[qid,bo])}
        alphas={}; val_scores={}; preds={}
        for name,(tx,ty,qx) in designs.items():
            mu,sd=tx.mean(0),tx.std(0)+1e-3; grid=base.ridge_predictions((tx-mu)/sd,ty,(qx-mu)/sd,ALPHAS); scores=[(base.metric(target[val_mask],grid[a][val_mask]),a) for a in ALPHAS]; best,a=min(scores,key=lambda x:x[0]["rmse"]); alphas[name]=float(a); val_scores[name]=best; preds[name]=grid[a][test_mask].copy(); del grid; gc.collect()
        tt=target[test_mask]; points={n:base.metric(tt,p) for n,p in preds.items()}; labs=labels[test_mask]; test_labels_arr=labs; yg=[tt[test_labels_arr==lab] for lab in test_labels]; pg={n:[preds[n][test_labels_arr==lab] for lab in test_labels] for n in designs}; boot=bootstrap_groups(yg,pg,np.random.default_rng(seed+100000),B)
        row={"seed":seed,"split_hash":hashlib.sha256(np.asarray(order,dtype=np.int64).tobytes()).hexdigest(),"panel_sha256":hashlib.sha256("\n".join(map(str,panel)).encode()).hexdigest(),"n_train_identities":len(train_ids)-1,"n_validation_identities":len(val_labels),"n_test_identities":len(test_labels),"n_test_cells":int(test_mask.sum()),"n_candidate_group_rows":len(gids),"n_identity_wide_batch_rows":len(gids),"control_reference":{"mode":"pre_registered_source_row_parity","rule":"CONTROL and source_row % 2 == 0","n_cells":int(control_ref_mask.sum()),"source_row_sha256":hashlib.sha256(np.asarray(source_rows[control_ref_mask],dtype=np.int64).tobytes()).hexdigest()},"alphas":alphas,"validation":val_scores,"test":points,"bootstrap":{n:{"B":B,"unit":"held-out identity cluster","ci95":np.quantile(boot[n],[.025,.975],axis=0).tolist()} for n in designs}}
        for n in designs:
            if n!="candidate":
                d=boot["candidate"]-boot[n]; row["bootstrap"][n]["delta_vs_candidate"]={"point":{m:points["candidate"][m]-points[n][m] for m in ["rmse","pearson","spearman"]},"ci95":np.quantile(d,[.025,.975],axis=0).tolist()}
        all_rows.append(row); print(seed,{n:points[n] for n in designs},flush=True); del target,ident,preds,boot; gc.collect()
    out={"protocol":"exact_aggregation_metadata_factorial_repaired","factorial_cells":["identity_mean_no_batch","identity_by_supported_batch_no_batch","identity_wide_batch","identity_by_supported_batch_with_batch"],"aggregation_controls":["identity_mean","identity_weighted","identity_wide_batch","group_masked","candidate"],"control_profile":{"mode":"pre_registered_source_row_parity","rule":"CONTROL and source_row % 2 == 0"},"bootstrap":{"B":B,"unit":"held-out identity cluster","spearman":"fixed rank transform then additive cluster resampling"},"seeds":all_rows,"n_cells":len(labels),"batches":batch_vals.tolist()}
    (OUT/"repaired_2x2_factorial.json").write_text(json.dumps(out,indent=2)); print(json.dumps({"path":str(OUT/"repaired_2x2_factorial.json"),"seeds":len(all_rows)},indent=2))

if __name__=="__main__": main()
