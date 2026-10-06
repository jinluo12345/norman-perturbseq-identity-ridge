"""Audit whether a gemgroup holdout is identifiable under the one-hot contract."""
from pathlib import Path
import json, h5py, numpy as np
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed'; RAW=ROOT/'data/raw/NormanWeissman2019_filtered.h5ad'
z=np.load(DATA/'cell_panel.npz',allow_pickle=True); labels=np.asarray(z['labels'],str); source=np.asarray(z['source_row'],int)
with h5py.File(RAW,'r') as f: gem=np.asarray(f['obs']['gemgroup'][:],int)[source]
rows=[]
for b in sorted(np.unique(gem)):
    mask=gem==b; ids=np.unique(labels[mask]); ids=ids[ids!='CONTROL']
    rows.append({'gemgroup':int(b),'n_cells':int(mask.sum()),'n_noncontrol_cells':int(np.sum(mask & (labels!='CONTROL'))),'n_identities':int(len(ids)),'identities':ids.tolist()})
pair=np.zeros((len(rows),len(rows)),int)
sets=[set(r['identities']) for r in rows]
for i in range(len(rows)):
    for j in range(len(rows)): pair[i,j]=len(sets[i]&sets[j])
out={'protocol':'gemgroup_availability_overlap_audit','n_batches':len(rows),'batches':rows,'pairwise_identity_intersections':pair.tolist(),'interpretation':'All gemgroups are observed assay-batch labels inside one Norman screen. A leave-one-gemgroup-out candidate cannot estimate the held-out one-hot coefficient without a predeclared fallback; this audit therefore supports within-screen known-gemgroup scope and does not claim external batch transfer.'}
(ROOT/'results/batch_availability_audit.json').write_text(json.dumps(out,indent=2)); print(json.dumps({'n_batches':len(rows),'identity_counts':[r['n_identities'] for r in rows],'pairwise_min_offdiag':int(np.min(pair+np.eye(len(rows),dtype=int)*10**6))},indent=2))
