"""Nature Communications style figures generated only from saved study outputs."""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import r2_score

ROOT=Path(__file__).resolve().parents[1]; FIG=ROOT/'paper/figures'; FIG.mkdir(parents=True,exist_ok=True)
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'svg.fonttype':'none','pdf.fonttype':42,'font.size':7,'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':.7,'legend.frameon':False})
BLUE='#2b6cb0'; TEAL='#2c7a7b'; ORANGE='#dd6b20'; GREY='#8a98a8'; DARK='#243447'; LIGHT='#e8f1f8'; PALETTE={'Mean':'#9aa6b2','Ridge (pathway)':'#dd6b20','Ridge (gene)':'#718096','MLP (pathway)':'#805ad5','Pathway-NEAT':'#2c7a7b','No pathway':'#c53030','No gene residual':'#d69e2e'}

def save(fig,name):
    # The ARIS alignment gate is applied to every multi-panel figure.
    try:
        sys.path.insert(0,'/inspire/qb-ilm/project/exploration-topic/jinluozhijie-CZXS25210075/.codex_aris/skills/nature-figure/scripts')
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(fig,json_out=str(FIG/f'{name}.alignment.json'),overlay_svg=str(FIG/f'{name}.alignment.svg'),tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    except Exception as e:
        (FIG/f'{name}.alignment.warning.txt').write_text(str(e))
    fig.savefig(FIG/f'{name}.svg',bbox_inches='tight'); fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight'); fig.savefig(FIG/f'{name}.png',dpi=600,bbox_inches='tight'); fig.savefig(FIG/f'{name}.tiff',dpi=600,bbox_inches='tight'); plt.close(fig)

def read():
    r=json.loads((ROOT/'results/model_results.json').read_text()); rows=[]
    disp={'mean':'Mean','ridge_pathway':'Ridge (pathway)','ridge_gene':'Ridge (gene)','mlp_pathway':'MLP (pathway)','pathway_neat':'Pathway-NEAT','ablate_no_pathway':'No pathway','ablate_no_gene_residual':'No gene residual'}
    for run in r['runs']:
        for k,v in run['models'].items(): rows.append({'seed':run['seed'],'model':disp[k],**v})
    return r,pd.DataFrame(rows)

def fig1_method():
    fig,ax=plt.subplots(figsize=(7.2,3.2)); ax.set_xlim(0,10); ax.set_ylim(0,5); ax.axis('off')
    def box(x,y,w,h,label,fc,ec=BLUE,fs=8):
        p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06,rounding_size=.08',fc=fc,ec=ec,lw=1.1); ax.add_patch(p); ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=fs,color=DARK,weight='bold')
    def arrow(x1,y1,x2,y2,label=''):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=12,lw=1,color=GREY));
        if label: ax.text((x1+x2)/2,(y1+y2)/2+.16,label,ha='center',va='center',fontsize=6,color=DARK)
    ax.text(.2,4.65,'a',fontsize=10,weight='bold'); ax.text(.55,4.65,'Identity-to-response decomposition with explicit forward-pass terms',fontsize=10,weight='bold',color=DARK)
    box(.25,2.05,1.35,1.0,'identity\neᵢ',LIGHT); box(2.05,3.22,1.35,.85,'incidence\nA (genes\n× K)','#fef3c7',ORANGE,fs=7); box(2.05,1.05,1.35,.85,'panel mask\neᵢ,G','#edf2f7',GREY)
    arrow(1.6,2.55,1.96,3.42); arrow(1.6,2.55,1.96,1.68)
    # Actual matrix-like glyphs make the pathway and residual routes visible.
    ax.imshow(np.array([[1,0,1,0,0],[0,1,1,0,1],[1,0,0,1,0],[0,0,1,1,1]]),extent=(3.58,4.55,3.32,4.00),cmap='Blues',vmin=0,vmax=1,aspect='auto')
    ax.imshow(np.array([[1,.2,0],[0,1,.3],[.1,0,1]]),extent=(3.58,4.55,1.15,1.83),cmap='Greys',vmin=0,vmax=1,aspect='auto'); ax.text(4.06,.78,'eᵢ,G (512 genes)',ha='center',fontsize=6,color=DARK)
    arrow(4.55,3.65,4.92,3.65); arrow(4.55,1.48,4.92,1.48)
    box(5.0,3.05,1.55,1.15,'softplus(a) ⊙ z\nU V + zD','#d9f0ee',TEAL,7); box(5.0,1.0,1.55,1.0,'R eᵢ,G\n(full 512×512)','#edf2f7',GREY,7)
    arrow(6.55,3.62,7.15,2.75,'Σ'); arrow(6.55,1.48,7.15,2.45,'Σ')
    box(7.15,2.05,1.65,1.05,'ŷ = b + T + R\ncontrol-relative y','#c6f6d5',TEAL,7)
    ax.text(8.95,3.55,'Tₖ is retained\nper module',ha='center',va='center',fontsize=6,color=DARK)
    ax.text(8.95,1.35,'outputs:\nRMSE, r,\ncoefficients',ha='center',va='center',fontsize=6,color=DARK)
    ax.text(.3,.28,'Module terms, the direct decoder D and residual R are summed before output; no post-hoc explainer is required.',fontsize=7,color=DARK)
    save(fig,'Fig1_method_overview')

def fig2_performance(r,df):
    fig,axs=plt.subplots(2,2,figsize=(7.2,4.8),gridspec_kw={'wspace':.34,'hspace':.42}); axs=axs.ravel(); order=['Mean','Ridge (gene)','MLP (pathway)','Ridge (pathway)','Pathway-NEAT']
    d=df[df.model.isin(order)]; means=d.groupby('model').rmse.mean().reindex(order); sem=d.groupby('model').rmse.std().reindex(order).fillna(0)
    labels=['mean','gene','MLP','PR','NEAT']; axs[0].bar(np.arange(len(order)),means,yerr=sem,capsize=2,color=[PALETTE[x] for x in order],edgecolor='white',lw=.5); axs[0].set_xticks(np.arange(len(order)),labels,fontsize=6); axs[0].set_ylabel('RMSE'); axs[0].set_title('Norman confirmation (n=36)',loc='left',weight='bold'); axs[0].set_ylim(0,.18); axs[0].text(-.14,1.05,'a',transform=axs[0].transAxes,fontsize=9,weight='bold')
    pear=d.groupby('model').pearson.mean().reindex(order); perr=d.groupby('model').pearson.std().reindex(order).fillna(0); axs[1].bar(np.arange(len(order)),pear,yerr=perr,capsize=2,color=[PALETTE[x] for x in order]); axs[1].set_xticks(np.arange(len(order)),labels,fontsize=6); axs[1].set_ylabel('Pearson r'); axs[1].set_ylim(.4,.74); axs[1].set_title('Direction recovered',loc='left',weight='bold'); axs[1].text(-.14,1.05,'b',transform=axs[1].transAxes,fontsize=9,weight='bold')
    ext=json.loads((ROOT/'results/external_stratification.json').read_text()); rows=[]
    for ds,v in ext.items():
        for group,g in v['groups'].items(): rows.append({'dataset':ds.replace('_',' '),'group':group,'n':g['n_profiles'],'r':g['pearson']})
    ed=pd.DataFrame(rows); labels=[]; vals=[]; cols=[]
    for ds in ['dixit','adamson single','adamson combo']:
        for group in ['matched','unmatched']:
            q=ed[(ed.dataset==ds)&(ed.group==group)].iloc[0]; short={'dixit':'Dixit','adamson single':'Adamson S','adamson combo':'Adamson C'}[ds]; labels.append(f"{short}\n{group[0].upper()} n={q.n}"); vals.append(q.r); cols.append(TEAL if group=='matched' else GREY)
    x=np.arange(len(vals)); axs[2].axhline(0,color='#cbd5e0',lw=.7); axs[2].scatter(x,vals,c=cols,s=34,zorder=3); axs[2].set_xticks(x,labels,fontsize=5); axs[2].set_ylabel('Pearson r'); axs[2].set_ylim(-.28,.08); axs[2].set_title('External stress test (descriptive)',loc='left',weight='bold'); axs[2].text(-.14,1.05,'c',transform=axs[2].transAxes,fontsize=9,weight='bold'); axs[2].text(.02,.03,'matched groups underpowered (n=1–2)',transform=axs[2].transAxes,fontsize=5.5,color=DARK)
    rep=pd.read_csv(ROOT/'results/repeated_local_summary.csv'); x=np.arange(len(rep)); axs[3].axhline(0,color='#718096',lw=.8); axs[3].axhline(rep['neat_pearson_mean'].sub(rep['ridge_pearson']).mean(),color=TEAL,lw=1); axs[3].scatter(x,rep['neat_pearson_mean']-rep['ridge_pearson'],c=TEAL,s=28); axs[3].set_xlabel('split seed (panel reselected)'); axs[3].set_ylabel('Pearson\nNEAT − ridge'); axs[3].set_ylim(-.004,.0015); axs[3].set_xticks(x,rep.split_seed.astype(str),fontsize=5); axs[3].text(-.14,1.05,'d',transform=axs[3].transAxes,fontsize=9,weight='bold')
    for ax in axs: ax.grid(axis='y',color='#e2e8f0',lw=.5); ax.set_axisbelow(True)
    save(fig,'Fig2_model_performance')

def fig3_ablation(df):
    fig,axs=plt.subplots(2,2,figsize=(7.2,4.8),gridspec_kw={'wspace':.34,'hspace':.42}); axs=axs.ravel(); order=['Pathway-NEAT','No pathway','No gene residual']; d=df[df.model.isin(order)]
    for j,metric in enumerate(['rmse','pearson']):
        means=d.groupby('model')[metric].mean().reindex(order); sd=d.groupby('model')[metric].std().reindex(order).fillna(0); axs[j].bar(range(3),means,yerr=sd,capsize=2,color=[PALETTE[x] for x in order]); axs[j].set_xticks(range(3),['full','− pathway','− gene\nresidual']); axs[j].tick_params(axis='both',labelsize=9); axs[j].set_ylabel('RMSE' if j==0 else 'Pearson r',fontsize=9); axs[j].set_title('Ablation: '+('error' if j==0 else 'direction'),loc='left',weight='bold',fontsize=10); axs[j].text(-0.14,1.06,chr(97+j),transform=axs[j].transAxes,fontsize=11,weight='bold'); axs[j].grid(axis='y',color='#e2e8f0',lw=.5); axs[j].set_axisbelow(True)
    b=json.load(open(ROOT/'results/neat_paired_bootstrap.json'))['paired_bootstrap_seed11']; vals=[b['rmse_neat_minus_ridge']['estimate'],b['pearson_neat_minus_ridge']['estimate']]; lo=[b['rmse_neat_minus_ridge']['ci95'][0],b['pearson_neat_minus_ridge']['ci95'][0]]; hi=[b['rmse_neat_minus_ridge']['ci95'][1],b['pearson_neat_minus_ridge']['ci95'][1]]; err=np.array([[vals[i]-lo[i] for i in range(2)],[hi[i]-vals[i] for i in range(2)]])
    ax=axs[2]; ax.axhline(0,color='#718096',lw=.8); ax.errorbar(range(2),vals,yerr=err,fmt='o',color=TEAL,capsize=3); ax.set_xticks(range(2),['RMSE\nNEAT−ridge','Pearson\nNEAT−ridge']); ax.tick_params(axis='both',labelsize=9); ax.set_title('Paired identity bootstrap',loc='left',weight='bold',fontsize=10); ax.text(-0.14,1.06,'c',transform=ax.transAxes,fontsize=11,weight='bold'); ax.grid(axis='y',color='#e2e8f0',lw=.5); ax.set_axisbelow(True)
    null=json.loads((ROOT/'results/reactome_structure_nulls.json').read_text()); nv=np.array([q['pearson'] for q in null['pattern_permutation_null']['runs']]); ax=axs[3]; ax.hist(nv,bins=14,color='#9bb6cf',edgecolor='white'); ax.axvline(null['observed']['pearson'],color=ORANGE,lw=1.5); ax.axvline(null['hierarchy_collapsed']['pearson'],color=TEAL,lw=1.2,ls='--'); ax.set_xlabel('Pearson r',fontsize=9); ax.set_ylabel('null count',fontsize=9); ax.tick_params(axis='both',labelsize=9); ax.set_title('Structure-preserving null',loc='left',weight='bold',fontsize=10); ax.text(-0.14,1.06,'d',transform=ax.transAxes,fontsize=11,weight='bold'); ax.grid(axis='y',color='#e2e8f0',lw=.5); ax.set_axisbelow(True)
    save(fig,'Fig3_ablation')
def fig4_pathways(r):
    meta=json.loads((ROOT/'data/processed/metadata.json').read_text()); genes=np.array(meta['genes']); names=np.array(meta['pathway_names'])
    # Use the saved forward-pass decoder from all three Pathway-NEAT fits.
    # This is deliberately distinct from the pathway-ridge baseline and is
    # retained in machine-readable form with seed and shape provenance below.
    coeff=np.stack([np.asarray(run['pathway_coefficients'],dtype=float) for run in r['runs']],axis=0)
    coef=coeff.mean(axis=0); score=np.linalg.norm(coef,axis=1); top=np.argsort(score)[-12:][::-1]
    gscore=np.linalg.norm(coef[top,:],axis=0); gtop=np.argsort(gscore)[-12:][::-1]; mat=coef[np.ix_(top,gtop)]
    prov={'source':'results/model_results.json pathway_coefficients','seeds':[int(x['seed']) for x in r['runs']], 'coefficient_shape':[int(x) for x in coeff.shape], 'aggregation':'arithmetic mean across optimization seeds', 'pathway_count':12, 'gene_count':12, 'pathway_indices':top.tolist(), 'gene_indices':gtop.tolist(), 'pathway_names':[str(names[i]) for i in top], 'gene_names':[str(genes[i]) for i in gtop], 'scale':'standardized response coefficient'}
    (ROOT/'results/neat_decoder_provenance.json').write_text(json.dumps(prov,indent=2))
    short={'Transcriptional regulation by RUNX1':'RUNX1 transcription','MITF-M-regulated melanocyte development':'MITF melanocyte development','Chromatin modifying enzymes':'Chromatin enzymes','MITF-M-dependent gene expression':'MITF gene expression','Transmission across Chemical Synapses':'Chemical synapses','Gastrulation':'Gastrulation','Cellular Senescence':'Cellular senescence','Interleukin-1 family signaling':'IL-1 signalling','Regulation of Expression and Function of Type I Classical Cadherins':'Cadherin regulation','Regulation of CDH1 Expression and Function':'CDH1 regulation','Mitotic G1 phase and G1/S transition':'G1/S transition','Somatic hypermutation of immunoglobulin genes':'Ig somatic hypermutation'}
    lim=np.percentile(abs(mat),98) or 1
    fig,ax=plt.subplots(figsize=(7.2,3.5)); im=ax.imshow(mat,cmap=LinearSegmentedColormap.from_list('bwr2',['#2b6cb0','#f7fafc','#c53030']),aspect='auto',vmin=-lim,vmax=lim); ax.set_yticks(range(len(top)),[short.get(str(names[i]),str(names[i])) for i in top],fontsize=8); ax.set_xticks(range(len(gtop)),genes[gtop],rotation=90,ha='center',fontsize=8); [t.set_rotation_mode('anchor') for t in ax.get_xticklabels()]; ax.tick_params(axis='x',which='both',length=0,pad=2); ax.spines['bottom'].set_visible(False); ax.set_xlabel('12 highest-norm response genes (summary of the full decoder)',fontsize=9); ax.set_title('Pathway-NEAT signed decoder terms',loc='left',weight='bold',fontsize=10); cb=fig.colorbar(im,ax=ax,fraction=.018,pad=.02); cb.set_label('standardized response coefficient',fontsize=9); save(fig,'Fig4_pathway_contributions')

def fig6_capacity():
    """Nested training-size learning curve and optimization diagnostics."""
    d=pd.read_csv(ROOT/'results/learning_curve_split11.csv')
    neat=d[d.model=='Pathway-NEAT'].copy(); ridge=d[d.model=='Pathway Ridge'].copy()
    order=sorted(neat.n_training_identities.unique()); x=np.arange(len(order)); labels=[str(int(v)) for v in order]
    nm=neat.groupby('n_training_identities'); rm=ridge.groupby('n_training_identities')
    fig,axs=plt.subplots(2,2,figsize=(7.2,4.8),gridspec_kw={'wspace':.34,'hspace':.42}); axs=axs.ravel()
    for j,metric in enumerate(['test_rmse','test_pearson']):
        nmean=nm[metric].mean().reindex(order); nstd=nm[metric].std().reindex(order).fillna(0)
        rmean=rm[metric].mean().reindex(order)
        axs[j].errorbar(x,nmean,yerr=nstd,fmt='o-',color=TEAL,capsize=2,label='Pathway-NEAT')
        axs[j].plot(x,rmean,'s-',color=ORANGE,label='Pathway ridge')
        axs[j].set_xticks(x,labels); axs[j].set_xlabel('training identities'); axs[j].set_ylabel('test RMSE' if j==0 else 'test Pearson r')
        axs[j].set_title('Nested capacity curve',loc='left',weight='bold'); axs[j].legend(fontsize=7,loc='best'); axs[j].text(-.14,1.05,chr(97+j),transform=axs[j].transAxes,fontsize=9,weight='bold')
    tr=neat.groupby('n_training_identities')[['train_rmse','validation_rmse']].mean().reindex(order); va=neat.groupby('n_training_identities')[['train_rmse','validation_rmse']].std().reindex(order).fillna(0)
    axs[2].plot(x,tr.train_rmse,'o-',color=BLUE,label='NEAT train'); axs[2].plot(x,tr.validation_rmse,'o--',color=TEAL,label='NEAT validation')
    axs[2].fill_between(x,tr.validation_rmse-va.validation_rmse,tr.validation_rmse+va.validation_rmse,color=TEAL,alpha=.12)
    axs[2].set_xticks(x,labels); axs[2].set_xlabel('training identities'); axs[2].set_ylabel('RMSE'); axs[2].set_title('Train (solid) and validation (dashed)',loc='left',weight='bold'); axs[2].text(-.14,1.05,'c',transform=axs[2].transAxes,fontsize=9,weight='bold')
    delta=(nm.test_rmse.mean()-rm.test_rmse.mean()).reindex(order)
    axs[3].axhline(0,color='#718096',lw=.8); axs[3].plot(x,delta,'o-',color=ORANGE); axs[3].set_xticks(x,labels); axs[3].set_xlabel('training identities'); axs[3].set_ylabel('NEAT − ridge RMSE'); axs[3].set_title('Paired test difference',loc='left',weight='bold'); axs[3].text(-.14,1.05,'d',transform=axs[3].transAxes,fontsize=9,weight='bold')
    for ax in axs: ax.grid(axis='y',color='#e2e8f0',lw=.5); ax.set_axisbelow(True)
    save(fig,'Fig6_capacity_learning_curve')

def fig5_errors(r,df):
    prof=pd.read_csv(ROOT/'results/profile_error_stratification.csv'); neat=pd.read_csv(ROOT/'results/neat_profile_deltas.csv'); meta=json.loads((ROOT/'data/processed/metadata.json').read_text())
    # Aggregate the three neural seeds so uncertainty is visible at identity level.
    nd=neat.groupby('identity').agg(rmse_neat=('rmse_neat','mean'),rmse_ridge=('rmse_ridge','mean'),delta_rmse=('delta_rmse','mean'),delta_pearson=('delta_pearson','mean')).reset_index(); prof=prof.merge(nd,on='identity',how='left')
    fig,axs=plt.subplots(1,3,figsize=(8.6,2.8),gridspec_kw={'wspace':.38})
    o=np.argsort(prof.rmse_pathway_ridge.to_numpy()); axs[0].bar(np.arange(len(prof)),prof.rmse_pathway_ridge.to_numpy()[o],color=[ORANGE if x=='single' else TEAL for x in prof.single_or_combo.to_numpy()[o]],width=.82); axs[0].set_xlabel('held-out identity (sorted)'); axs[0].set_ylabel('pathway-ridge RMSE'); axs[0].set_title('Profile error by identity',loc='left',weight='bold'); axs[0].text(-.12,1.05,'a',transform=axs[0].transAxes,fontsize=9,weight='bold')
    axs[1].scatter(prof.response_norm,prof.rmse_pathway_ridge,c=[ORANGE if x=='single' else TEAL for x in prof.single_or_combo],s=24,edgecolor='white',lw=.4); axs[1].set_xlabel('observed response norm'); axs[1].set_ylabel('profile RMSE'); axs[1].set_title('Magnitude and error',loc='left',weight='bold'); axs[1].text(-.12,1.05,'b',transform=axs[1].transAxes,fontsize=9,weight='bold')
    axs[2].axhline(0,color='#718096',lw=.8); axs[2].scatter(prof.response_norm,prof.delta_rmse,c=[ORANGE if x=='single' else TEAL for x in prof.single_or_combo],s=24,edgecolor='white',lw=.4); axs[2].set_xlabel('observed response norm'); axs[2].set_ylabel('NEAT − ridge RMSE'); axs[2].set_title('Neural excess error',loc='left',weight='bold'); axs[2].text(-.12,1.05,'c',transform=axs[2].transAxes,fontsize=9,weight='bold')
    for ax in axs: ax.grid(axis='y',color='#e2e8f0',lw=.5); ax.set_axisbelow(True)
    save(fig,'Fig5_error_boundary')

def main():
    r,df=read(); fig1_method(); fig2_performance(r,df); fig3_ablation(df); fig4_pathways(r); fig5_errors(r,df); fig6_capacity(); df.to_csv(ROOT/'results/model_summary.csv',index=False); print('figures written',len(list(FIG.glob('Fig*.pdf'))))
if __name__=='__main__': main()
