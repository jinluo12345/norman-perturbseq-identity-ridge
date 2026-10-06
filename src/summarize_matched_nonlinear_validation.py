import json, numpy as np, hashlib
from pathlib import Path
p=Path('results/matched_nonlinear_validation_corrected.json'); d=json.loads(p.read_text())
metrics=['rmse','pearson','spearman']; out={'source_hash':hashlib.sha256(p.read_bytes()).hexdigest(),'protocol':d['protocol'],'rows':[],'families':{}}
for s in d['seeds']:
 r=s['ridge_validation']
 for fam in ['mlp','mlp_linear_skip']:
  for ts in d['training_seeds']:
   q=s['models'][f'{fam}_seed{ts}']['validation']
   out['rows'].append({'split_seed':s['split_seed'],'training_seed':ts,'family':fam,**{m:q[m] for m in metrics},'ridge_rmse':r['rmse'],'ridge_pearson':r['pearson'],'ridge_spearman':r['spearman'],'best_epoch':s['models'][f'{fam}_seed{ts}']['best_epoch'],'parameter_count':s['models'][f'{fam}_seed{ts}']['parameter_count']})
for fam in ['mlp','mlp_linear_skip']:
 a=np.array([[x[m] for m in metrics] for x in out['rows'] if x['family']==fam]); r=np.array([[s['ridge_validation'][m] for m in metrics] for s in d['seeds']]);
 out['families'][fam]={'mean':dict(zip(metrics,a.mean(0).tolist())),'split_sd_all_runs':dict(zip(metrics,a.std(0,ddof=1).tolist())),'ridge_mean':dict(zip(metrics,r.mean(0).tolist())),'delta_mean_vs_ridge':dict(zip(metrics,(a.mean(0)-r.mean(0)).tolist())),'all_split_all_metric_gate':False,'parameter_count':int(a[0,0]*0+next(x['parameter_count'] for x in out['rows'] if x['family']==fam))}
Path('results/matched_nonlinear_validation_summary.json').write_text(json.dumps(out,indent=2))
with open('paper/supplementary_matched_nonlinear.tex','w') as f:
 f.write('\\begin{table}[ht]\\centering\\scriptsize\\caption{Supplementary Table 4. Validation-only matched nonlinear comparator under the frozen 878-input, 512-target contract. Each row is one identity split and optimization seed; no confirmation responses were scored. Values are cell-level validation metrics after raw-RMSE checkpoint selection.}\\label{tab:matched_nonlinear}\\begin{tabular}{rrlrrrrr}\\toprule Split & Train seed & Family & RMSE & Pearson & Spearman & Best epoch & Parameters \\\\ \\midrule\n')
 for x in out['rows']:
  f.write(f"{x['split_seed']} & {x['training_seed']} & {x['family'].replace('_',' ')} & {x['rmse']:.5f} & {x['pearson']:.5f} & {x['spearman']:.5f} & {x['best_epoch']} & {x['parameter_count']:,} \\\\\n")
 f.write('\\bottomrule\\end{tabular}\\end{table}\n')
