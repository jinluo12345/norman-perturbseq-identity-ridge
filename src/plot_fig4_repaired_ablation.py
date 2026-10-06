"""Plot the repaired validation-only feature-block ablation.

All points are generated from the frozen-control JSON.  The lower-right panel
shows paired ablation-minus-full changes for each seed, while the first three
panels retain the validation metric scale and expose the seed-level spread.
"""
from pathlib import Path
import json, numpy as np
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'paper/figures_batch'; OUT.mkdir(exist_ok=True)
D=json.loads((ROOT/'results/repaired_ablation_frozen_control.json').read_text())
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
    'font.size':7,'axes.labelsize':7,'xtick.labelsize':6.2,'ytick.labelsize':6.2,
    'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.65,
    'axes.spines.top':False,'axes.spines.right':False})
C={'ink':'#263238','full':'#5AAE61','mild':'#377EB8','strong':'#D55E00','grid':'#DCE3E7'}
NAMES=['full','no_pathway','no_gene','no_component']
LABELS=['Full','−pathway','−panel genes','−components']
def main():
    fig,axs=plt.subplots(2,2,figsize=(7.2047,4.7244),gridspec_kw={'wspace':.30,'hspace':.34})
    for ax,label in zip(axs.flat,'abcd'):
        ax.text(-.22,1.15,label,transform=ax.transAxes,fontsize=9,fontweight='bold',va='top',color=C['ink'])
    seeds=sorted({int(r['seed']) for r in D['rows']}); x=np.arange(len(seeds))
    palette=[C['full'],C['mild'],C['mild'],C['strong']]
    for ax,key,title in [(axs[0,0],'rmse','Validation RMSE'),(axs[0,1],'pearson','Validation Pearson'),(axs[1,0],'spearman','Validation Spearman')]:
        for name,label,col in zip(NAMES,LABELS,palette):
            vals=np.array([next(r['validation'][key] for r in D['rows'] if r['seed']==s and r['variant']==name) for s in seeds])
            ax.plot(x,vals,'o-',ms=3,lw=.9,color=col,label=label)
        ax.set_xticks(x,[str(s) for s in seeds]); ax.set_xlabel('Validation split seed'); ax.set_ylabel(title)
        ax.grid(axis='y',color=C['grid'],lw=.45)
    # Colors and block names are defined in the caption; avoiding an inset
    # legend keeps the validation traces unobstructed at journal scale.
    # Paired changes relative to the full model; light lines retain all seeds.
    ax=axs[1,1]; full={s:{k:next(r['validation'][k] for r in D['rows'] if r['seed']==s and r['variant']=='full') for k in ['rmse','pearson','spearman']} for s in seeds}
    metrics=[('rmse','Δ RMSE'),('pearson','Δ Pearson'),('spearman','Δ Spearman')]; xx=np.arange(len(metrics)); offsets=np.linspace(-.18,.18,len(seeds))
    for name,label,col in zip(NAMES[1:],LABELS[1:],palette[1:]):
        for s,off in zip(seeds,offsets):
            vals=[next(r['validation'][k] for r in D['rows'] if r['seed']==s and r['variant']==name)-full[s][k] for k,_ in metrics]
            ax.plot(xx+off,vals,'o',ms=3,color=col,alpha=.9)
        mean=np.array([np.mean([next(r['validation'][k] for r in D['rows'] if r['seed']==s and r['variant']==name)-full[s][k] for s in seeds]) for k,_ in metrics])
        sd=np.array([np.std([next(r['validation'][k] for r in D['rows'] if r['seed']==s and r['variant']==name)-full[s][k] for s in seeds],ddof=1) for k,_ in metrics])
        ax.errorbar(xx,mean,yerr=sd,fmt='o-',ms=3.5,lw=1.0,capsize=2,color=col,label=label)
    ax.axhline(0,color=C['ink'],lw=.65); ax.set_xticks(xx,[t for _,t in metrics]); ax.set_ylabel('Ablation − full'); ax.grid(axis='y',color=C['grid'],lw=.45)
    for ext,dpi in [('pdf',None),('svg',None),('png',600)]:
        fig.savefig(OUT/f'Fig4_cell_batch_ablation_repaired.{ext}',bbox_inches='tight',dpi=dpi)
    plt.close(fig)
if __name__=='__main__': main()
