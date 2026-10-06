from pathlib import Path
import json,numpy as np
import matplotlib as mpl;mpl.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'paper/figures_batch';D=json.loads((ROOT/'results/repaired_split_local_controls.json').read_text())
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'font.size':7,'axes.labelsize':7,'xtick.labelsize':6.2,'ytick.labelsize':6.2,'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.65,'axes.spines.top':False,'axes.spines.right':False})
COL={'base':'#377EB8','cand':'#E69F00','ink':'#263238'}
def main():
 seeds=[r['seed'] for r in D['seeds']];x=np.arange(len(seeds));fig,axs=plt.subplots(2,2,figsize=(7.2,5.15),gridspec_kw={'wspace':.28,'hspace':.58})
 for ax,metric,label in [(axs[0,0],'rmse','RMSE'),(axs[0,1],'pearson','Pearson'),(axs[1,0],'spearman','Spearman')]:
  b=np.array([r['test']['identity_mean'][metric] for r in D['seeds']]);c=np.array([r['test']['candidate'][metric] for r in D['seeds']]);ax.plot(x-.08,b,'o-',color=COL['base'],lw=1.1,ms=3.5,label='Identity-wide / no batch');ax.plot(x+.08,c,'o-',color=COL['cand'],lw=1.1,ms=3.5,label='Group + batch');ax.set_xticks(x,seeds);ax.set_xlabel('Confirmation seed');ax.set_ylabel(label);ax.set_title(label,loc='left',fontweight='bold');ax.grid(axis='y',color='#DCE3E7',lw=.45)
 axs[1,1].axhline(0,color=COL['ink'],lw=.6)
 metric_cols={'rmse':'#263238','pearson':COL['cand'],'spearman':'#4C78A8'}
 for j,m in enumerate(['rmse','pearson','spearman']):
  vals=np.array([r['test']['candidate'][m]-r['test']['identity_mean'][m] for r in D['seeds']])
  ci=np.array([r['bootstrap']['identity_mean']['delta_vs_candidate']['ci95'] for r in D['seeds']])
  # Stored deltas are candidate minus identity-wide; show the paired identity-cluster CI.
  lo,hi=ci[:,0,j],ci[:,1,j]; col=metric_cols[m]
  axs[1,1].errorbar(x,vals,yerr=[vals-lo,hi-vals],fmt='o-',lw=1.0,ms=3.2,capsize=2,color=col,label='Δ '+m)
 axs[1,1].set_xticks(x,seeds);axs[1,1].set_xlabel('Confirmation seed');axs[1,1].set_ylabel('Candidate − identity-wide');axs[1,1].set_title('Paired point differences',loc='left',fontweight='bold');axs[1,1].grid(axis='y',color='#DCE3E7',lw=.45);axs[1,1].legend(frameon=False,fontsize=5.0,loc='upper center',bbox_to_anchor=(0.5,-0.20),ncol=3);axs[0,0].legend(frameon=False,fontsize=5.5,loc='best')
 for ax,l in zip(axs.flat,'abcd'):ax.text(-.18,1.06,l,transform=ax.transAxes,fontsize=9,fontweight='bold',va='top',color=COL['ink'])
 for ext,dpi in [('pdf',None),('svg',None),('png',600)]:fig.savefig(OUT/f'Fig2_cell_batch_repaired_performance.{ext}',bbox_inches='tight',dpi=dpi)
 plt.close(fig)
if __name__=='__main__':main()
