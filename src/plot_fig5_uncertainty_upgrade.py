"""Rebuild Fig. 5 as a dense conceptual + quantitative evidence figure.

Panel A is a deterministic vector reconstruction of the image2 construction
reference. Panels B-D use only frozen Norman result JSON files. The image2 PNG
is never embedded in the submission figure.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"paper/figures_batch"; OUT.mkdir(exist_ok=True)
BOOT=json.loads((ROOT/"results/cell_batch_ridge_bootstrap.json").read_text())
STRAT=json.loads((ROOT/"results/cell_batch_stratified_analysis.json").read_text())
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],"font.size":7.0,"axes.labelsize":7.0,"xtick.labelsize":6.4,"ytick.labelsize":6.4,"svg.fonttype":"none","pdf.fonttype":42,"axes.linewidth":0.65,"axes.spines.top":False,"axes.spines.right":False})
COL={"ink":"#263238","id":"#2C5D8A","base":"#377EB8","cand":"#E69F00","green":"#5AAE61","line":"#91A0AA","pale":"#EEF4F7","orangepale":"#FFF3E5"}

def box(ax,x,y,w,h,fc,ec,title,fontsize=6.5):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.008,rounding_size=0.015",facecolor=fc,edgecolor=ec,linewidth=.8,transform=ax.transAxes))
    ax.text(x+w/2,y+h-.045,title,ha="center",va="top",fontsize=fontsize,fontweight="bold",color=COL["ink"],transform=ax.transAxes)

def panel_a(ax):
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.text(.01,.98,"a",transform=ax.transAxes,fontsize=9,fontweight="bold",va="top",color=COL["ink"])
    box(ax,.04,.57,.35,.32,"#F0F6FA",COL["id"],"identity cluster",6.2)
    for j,label in enumerate(["A","B","C"]):
        x=.065+j*.112; box(ax,x,.63,.095,.16,"#F8FBFD",COL["id"],label,4.9)
        for k in range(3): ax.add_patch(Circle((x+.028+k*.028,.695+(k%2)*.035),.0065,color=COL["base"],transform=ax.transAxes))
    box(ax,.43,.57,.27,.32,"#F4F7F8",COL["line"],"split",6.0)
    for j,label in enumerate(["train","validation","held-out"]):
        x=.455+j*.078; ax.add_patch(FancyBboxPatch((x,.67),.065,.12,boxstyle="round,pad=.004,rounding_size=.008",facecolor="#DDEAF1" if j<2 else "#EEF0F1",edgecolor=COL["line"],linewidth=.5,transform=ax.transAxes)); ax.text(x+.032,.73,label,ha="center",va="center",fontsize=4.4,color=COL["ink"],transform=ax.transAxes)
    ax.add_patch(FancyArrowPatch((.39,.73),(.425,.73),arrowstyle="-|>",mutation_scale=8,color=COL["ink"],lw=.9,transform=ax.transAxes))
    box(ax,.74,.57,.22,.32,"#EFF8EE",COL["green"],"cell score",6.0)
    for k in range(4): ax.add_patch(Circle((.78+k*.035,.72+(k%2)*.05),.0065,color=COL["base"],transform=ax.transAxes))
    ax.text(.85,.64,"512-gene response",ha="center",fontsize=4.5,color=COL["ink"],transform=ax.transAxes)
    # lower row: paired cluster bootstrap and estimands
    box(ax,.04,.10,.38,.31,"#F5F9FB",COL["id"],"resample",5.3)
    ax.text(.07,.285,"identities with replacement",fontsize=4.5,color=COL["ink"],transform=ax.transAxes)
    for j in range(3):
        x=.18+j*.075
        ax.text(x,.305,f"I{j+1}",ha="center",fontsize=4.6,color=COL["ink"],transform=ax.transAxes)
        for k in range(3):
            ax.add_patch(Circle((x-.018+k*.018,.25),.006,color=COL["base"],transform=ax.transAxes))
            ax.add_patch(Circle((x-.018+k*.018,.19),.006,color=COL["cand"],transform=ax.transAxes))
        ax.plot([x-.02,x+.02],[.22,.22],color=COL["ink"],lw=.45,transform=ax.transAxes)
    ax.add_patch(FancyArrowPatch((.06,.22),(.14,.22),arrowstyle="-|>",mutation_scale=8,color=COL["ink"],lw=.8,transform=ax.transAxes))
    box(ax,.47,.10,.23,.31,"#F7F7F6",COL["line"],"Δ metrics",5.3)
    for j,t in enumerate(["Δ RMSE","Δ Pearson","Δ Spearman"]):
        ax.text(.585,.285-j*.065,t,ha="center",fontsize=4.5,color=COL["ink"],transform=ax.transAxes)
    ax.add_patch(FancyArrowPatch((.42,.25),(.46,.25),arrowstyle="-|>",mutation_scale=8,color=COL["ink"],lw=.8,transform=ax.transAxes))
    box(ax,.75,.10,.21,.31,COL["orangepale"],COL["cand"],"batches",5.3)
    for j in range(8):
        x=.77+j*.022; ax.text(x,.285,f"{j+1}",ha="center",fontsize=3.0,color=COL["ink"],transform=ax.transAxes)
        for k in range(2+(j%3)): ax.add_patch(Circle((x,.20-k*.035),.0045,color=COL["line"],transform=ax.transAxes))
    ax.add_patch(FancyArrowPatch((.85,.57),(.85,.43),arrowstyle="-|>",mutation_scale=8,color=COL["cand"],lw=1.0,linestyle="--",transform=ax.transAxes))
    ax.text(.925,.485,"metadata",fontsize=3.6,color=COL["cand"],rotation=90,va="center",transform=ax.transAxes)
    ax.text(.855,.125,"same Norman screen",ha="center",fontsize=4.0,color=COL["ink"],transform=ax.transAxes)

def panel_b(ax):
    ax.text(-.12,1.05,"b",transform=ax.transAxes,fontsize=9,fontweight="bold",va="top",color=COL["ink"])
    seeds=[r["seed"] for r in BOOT["seeds"]]; y=np.arange(len(seeds));
    metrics=[("RMSE",0,"Δ RMSE (batch − identity)"),("Pearson",1,"Δ Pearson (batch − identity)")]
    for j,(name,col,label) in enumerate(metrics):
        pts=np.array([r["paired_delta_candidate_minus_baseline"][name.lower()] for r in BOOT["seeds"]]); ci=np.array([r["identity_cluster_bootstrap"]["delta_ci95_rmse_pearson"] for r in BOOT["seeds"]]); lo,hi=ci[:,0,col],ci[:,1,col]
        xx=np.array([j]*len(y)) + (y-y.mean())*.018
        ax.errorbar(xx,pts,yerr=[pts-lo,hi-pts],fmt="o",ms=3.2,lw=.8,capsize=2,color=COL["cand"],ecolor=COL["cand"])
        for k in range(len(y)): ax.text(xx[k]+.035,pts[k],str(seeds[k]),fontsize=4.5,va="center",color=COL["ink"])
    ax.axhline(0,color=COL["ink"],lw=.65); ax.set_xlim(-.55,1.55); ax.set_xticks([0,1],["RMSE","Pearson"]); ax.set_ylabel("Candidate − baseline"); ax.set_title("identity-cluster 95% intervals",loc="left",fontsize=7.2,fontweight="bold"); ax.grid(axis="y",color="#DCE3E7",lw=.45)

def panel_c(ax):
    ax.text(-.12,1.05,"c",transform=ax.transAxes,fontsize=9,fontweight="bold",va="top",color=COL["ink"])
    seeds=[r["seed"] for r in BOOT["seeds"]]; x=np.arange(len(seeds));
    sp=np.array([r["paired_delta_candidate_minus_baseline"]["spearman"] for r in BOOT["seeds"]])
    ax.plot(x,sp,"o-",color=COL["cand"],lw=1.2,ms=4,label="Spearman point")
    for st,col in [("single",COL["base"]),("combination",COL["id"])]:
        vals=np.array([r["strata"][st]["candidate"]["rmse"]-r["strata"][st]["baseline"]["rmse"] for r in STRAT["rows"]]); ax.plot(x,vals,"o--",color=col,lw=.9,ms=3,label=f"Δ RMSE · {st}")
    ax.axhline(0,color=COL["ink"],lw=.65); ax.set_xticks(x,seeds); ax.set_xlabel("Confirmation seed"); ax.set_ylabel("Paired difference"); ax.set_title("rank and composition-stratified differences",loc="left",fontsize=7.2,fontweight="bold"); ax.grid(axis="y",color="#DCE3E7",lw=.45); ax.legend(frameon=False,fontsize=5.4,loc="lower right")

def panel_d(ax):
    ax.text(-.12,1.05,"d",transform=ax.transAxes,fontsize=9,fontweight="bold",va="top",color=COL["ink"])
    vals=[]
    for b in range(1,9): vals.append([r["batch"][str(b)]["candidate"]["rmse"]-r["batch"][str(b)]["baseline"]["rmse"] for r in STRAT["rows"]])
    bp=ax.boxplot(vals,positions=np.arange(1,9),widths=.58,patch_artist=True,showfliers=False,boxprops={"facecolor":"#FFF3E5","edgecolor":COL["cand"]},medianprops={"color":COL["ink"],"lw":1},whiskerprops={"color":COL["cand"],"lw":.8},capprops={"color":COL["cand"],"lw":.8})
    ax.axhline(0,color=COL["ink"],lw=.65); ax.set_xlabel("Observed gemgroup"); ax.set_ylabel("Δ RMSE"); ax.set_xticks(np.arange(1,9),[f"G{i}" for i in range(1,9)]); ax.set_title("same Norman screen · descriptive strata",loc="left",fontsize=7.2,fontweight="bold"); ax.grid(axis="y",color="#DCE3E7",lw=.45)

def main():
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.1),gridspec_kw={"wspace":.28,"hspace":.34})
    panel_a(axs[0,0]); panel_b(axs[0,1]); panel_c(axs[1,0]); panel_d(axs[1,1])
    fig.savefig(OUT/"Fig5_identity_cluster_uncertainty_upgrade.pdf",bbox_inches="tight"); fig.savefig(OUT/"Fig5_identity_cluster_uncertainty_upgrade.svg",bbox_inches="tight"); fig.savefig(OUT/"Fig5_identity_cluster_uncertainty_upgrade.png",dpi=600,bbox_inches="tight")
    plt.close(fig)
if __name__=="__main__": main()
