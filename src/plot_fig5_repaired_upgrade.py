"""Final Fig. 5 candidate using repaired control reference and factorial evidence."""
from pathlib import Path
import json, numpy as np
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'paper/figures_batch'; OUT.mkdir(exist_ok=True)
R=json.loads((ROOT/'results/repaired_split_local_controls.json').read_text())
F=(ROOT/'results/repaired_2x2_factorial.json')
F=json.loads(F.read_text()) if F.exists() else None
if F is not None and int(F.get('bootstrap',{}).get('B',0)) < 1000: F=None
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'font.size':6.8,'axes.labelsize':6.8,'xtick.labelsize':6.2,'ytick.labelsize':6.2,'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.65,'axes.spines.top':False,'axes.spines.right':False})
C={'ink':'#263238','id':'#2C5D8A','base':'#377EB8','cand':'#E69F00','green':'#5AAE61','line':'#91A0AA','pale':'#EEF4F7','orangepale':'#FFF3E5'}
def box(ax,x,y,w,h,fc,ec,t,fs=5.2):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.006,rounding_size=.012',facecolor=fc,edgecolor=ec,lw=.7,transform=ax.transAxes)); ax.text(x+w/2,y+h-.035,t,ha='center',va='top',fontsize=fs,fontweight='bold',color=C['ink'],transform=ax.transAxes)
def panel_a(ax):
    img_path=ROOT/'figures/ai_generated/Fig5_A_identity_cluster_concept_v4_upsampled.png'
    if not img_path.exists():
        raise FileNotFoundError(f"Required native image2 panel missing: {img_path}")
    img=plt.imread(img_path)
    ax.imshow(img, interpolation='lanczos')
    ax.axis('off')
    ax.text(.006,.994,'a',transform=ax.transAxes,fontsize=9,fontweight='bold',va='top',ha='left',color=C['ink'],bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=.4))

def panel_b(ax):
    ax.text(-.12,1.05,'b',transform=ax.transAxes,fontsize=9,fontweight='bold',va='top',color=C['ink']); ax.axis('off')
    seeds=[r['seed'] for r in R['seeds']]; y=np.arange(len(seeds)); names=['rmse','pearson','spearman']; labels=['Δ RMSE','Δ Pearson','Δ Spearman']
    for j,m in enumerate(names):
        iax=ax.inset_axes([.08, .69-j*.28, .87, .20])
        pts=np.array([r['bootstrap']['identity_mean']['delta_vs_candidate']['point'][m] for r in R['seeds']]); ci=np.array([r['bootstrap']['identity_mean']['delta_vs_candidate']['ci95'] for r in R['seeds']]); lo,hi=ci[:,0,j],ci[:,1,j]; xx=np.arange(len(y)); iax.errorbar(xx,pts,yerr=[pts-lo,hi-pts],fmt='o',ms=2.8,lw=.7,capsize=1.8,color=C['cand'],ecolor=C['cand'])
        iax.axhline(0,color=C['ink'],lw=.5); iax.set_ylabel(labels[j],fontsize=5.5); iax.grid(axis='y',color='#DCE3E7',lw=.35); iax.tick_params(labelsize=5)
        if j<2: iax.set_xticks(xx,[])
        else: iax.set_xticks(xx,[str(s) for s in seeds]); iax.set_xlabel('confirmation seed',fontsize=5.5)
def panel_c(ax):
    ax.text(-.12,1.05,'c',transform=ax.transAxes,fontsize=9,fontweight='bold',va='top',color=C['ink'])
    # Aggregation-matched controls are shown for all three paired estimands.
    # The repaired bundle stores identity-cluster percentile intervals for each
    # metric, so this panel makes the Spearman uncertainty visible instead of
    # implying that only RMSE was audited.
    metrics=[('rmse','Δ RMSE'),('pearson','Δ Pearson'),('spearman','Δ Spearman')]
    controls=[('identity_weighted',C['base'],'weighted identity'),('group_masked',C['id'],'group masked')]
    x=np.arange(len(metrics)); offsets=[-.11,.11]
    for (n,col,label),off in zip(controls,offsets):
        pts=np.array([np.mean([r['bootstrap'][n]['delta_vs_candidate']['point'][m] for r in R['seeds']]) for m,_ in metrics])
        lo=np.array([np.mean([r['bootstrap'][n]['delta_vs_candidate']['ci95'][0][j] for r in R['seeds']]) for j,(m,_) in enumerate(metrics)])
        hi=np.array([np.mean([r['bootstrap'][n]['delta_vs_candidate']['ci95'][1][j] for r in R['seeds']]) for j,(m,_) in enumerate(metrics)])
        ax.errorbar(x+off,pts,yerr=[pts-lo,hi-pts],fmt='o',ms=3.2,lw=.8,capsize=2,color=col,ecolor=col,label=label)
    ax.axhline(0,color=C['ink'],lw=.6);ax.set_xticks(x,[t for _,t in metrics]);ax.set_ylabel('Candidate − control');ax.grid(axis='y',color='#DCE3E7',lw=.4)
def panel_d(ax):
    ax.text(-.12,1.05,'d',transform=ax.transAxes,fontsize=9,fontweight='bold',va='top',color=C['ink'])
    if F:
        names=['identity_mean','identity_weighted','identity_wide_batch','group_masked','candidate']; labels=['wide/0','weighted','wide+batch','group/0','group+batch']; means=[]; cis=[]
        for n in names:
            vals=[]
            for r in F['seeds']:
                b=r['bootstrap'].get(n,{}); vals.append(r['test'][n]['rmse'])
            means.append(np.mean(vals));cis.append(np.std(vals,ddof=1))
        ax.errorbar(np.arange(len(names)),means,yerr=cis,fmt='o',color=C['cand'],ecolor=C['cand'],capsize=2,lw=.8);ax.set_xticks(np.arange(len(names)),labels,rotation=28,ha='right',rotation_mode='anchor',fontsize=4.6);ax.set_ylabel('RMSE (repaired control)');ax.grid(axis='y',color='#DCE3E7',lw=.4)
    else:
        ax.text(.5,.5,'factorial result pending',ha='center',va='center',transform=ax.transAxes,color=C['ink'])
        ax.set_axis_off()
def main():
    # Asymmetric journal layout: the native image2 concept panel spans three
    # columns so its labels remain readable at single-column manuscript scale;
    # quantitative panels retain independent axes and equal visual weight.
    fig=plt.figure(figsize=(7.2,5.4))
    gs=fig.add_gridspec(2,4,wspace=.50,hspace=.34)
    ax_a=fig.add_subplot(gs[0,:3]); ax_b=fig.add_subplot(gs[0,3])
    ax_c=fig.add_subplot(gs[1,:2]); ax_d=fig.add_subplot(gs[1,2:])
    panel_a(ax_a); panel_b(ax_b); panel_c(ax_c); panel_d(ax_d)
    for ext,dpi in [('pdf',None),('svg',None),('png',600)]: fig.savefig(OUT/f'Fig5_identity_cluster_uncertainty_repaired.{ext}',bbox_inches='tight',dpi=dpi)
    plt.close(fig)
if __name__=='__main__':main()
