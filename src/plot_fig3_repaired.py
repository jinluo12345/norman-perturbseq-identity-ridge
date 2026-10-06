"""Plot composition-stratified confirmation metrics from repaired frozen-control evidence."""
from pathlib import Path
import json,numpy as np
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'paper/figures_batch'; D=json.loads((ROOT/'results/repaired_stratified_analysis.json').read_text())
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'font.size':7,'axes.labelsize':7,'xtick.labelsize':6.2,'ytick.labelsize':6.2,'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.65,'axes.spines.top':False,'axes.spines.right':False})
C={'base':'#377EB8','cand':'#E69F00','ink':'#263238','grid':'#DCE3E7'}
def main():
    rows=D['rows']; seeds=[r['seed'] for r in rows]; strata=['single','combination']; labels=['Singleton','Combination']; fig,axs=plt.subplots(2,2,figsize=(7.2,4.75),gridspec_kw={'wspace':.28,'hspace':.36})
    for ax,l in zip(axs.flat,'abcd'): ax.text(-.20,1.15,l,transform=ax.transAxes,fontsize=9,fontweight='bold',va='top',color=C['ink'])
    for ax,key,title in [(axs[0,0],'rmse','RMSE (lower)'),(axs[0,1],'pearson','Pearson'),(axs[1,0],'spearman','Spearman')]:
        x=np.arange(2); w=.18
        for k,(name,col) in enumerate([('Identity-wide / no batch',C['base']),('Group + batch',C['cand'])]):
            means=[np.mean([r['strata'][st][('baseline' if k==0 else 'candidate')][key] for r in rows]) for st in strata]; sd=[np.std([r['strata'][st][('baseline' if k==0 else 'candidate')][key] for r in rows],ddof=1) for st in strata]
            ax.bar(x+(k-.5)*w,means,w,yerr=sd,capsize=2,color=col,edgecolor='white',label=name,error_kw={'lw':.7})
        ax.set_xticks(x,labels); ax.set_ylabel(title); ax.grid(axis='y',color=C['grid'],lw=.45)
    ax=axs[1,1]; mat=np.array([[r['strata'][st]['candidate']['rmse']-r['strata'][st]['baseline']['rmse'] for st in strata] for r in rows]); im=ax.imshow(mat,cmap='RdBu_r',vmin=-np.max(np.abs(mat)),vmax=np.max(np.abs(mat)),aspect='auto')
    ax.set_xticks([0,1],labels); ax.set_yticks(np.arange(len(seeds)),seeds); ax.set_xlabel('Held-out stratum'); ax.set_ylabel('Confirmation seed')
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]): ax.text(j,i,f'{mat[i,j]:+.4f}',ha='center',va='center',fontsize=6)
    fig.colorbar(im,ax=ax,shrink=.85,label='Group + batch − identity-wide RMSE')
    fig.legend([plt.Rectangle((0,0),1,1,facecolor=C['base']),plt.Rectangle((0,0),1,1,facecolor=C['cand'])],['Identity-wide / no batch','Group + batch'],loc='upper center',bbox_to_anchor=(.5,1.06),ncol=2,frameon=False,fontsize=6)
    for ext,dpi in [('pdf',None),('svg',None),('png',600)]: fig.savefig(OUT/f'Fig3_cell_batch_strata.{ext}',bbox_inches='tight',dpi=dpi)
    plt.close(fig)
if __name__=='__main__': main()
