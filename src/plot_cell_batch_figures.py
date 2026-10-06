"""Publication figures for the frozen cell-level gemgroup Ridge evidence."""
from pathlib import Path
import json
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

SKILL = Path("/inspire/qb-ilm/project/exploration-topic/jinluozhijie-CZXS25210075/.codex_aris/skills/nature-figure")
import sys
sys.path.insert(0, str(SKILL / "scripts"))
from audit_panel_alignment import require_matplotlib_panel_alignment

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures_batch"
OUT.mkdir(parents=True, exist_ok=True)
VAL = json.loads((ROOT / "results/cell_batch_ridge_validation.json").read_text())
CONF = json.loads((ROOT / "results/cell_batch_ridge_confirmation.json").read_text())
BOOT = json.loads((ROOT / "results/cell_batch_ridge_bootstrap.json").read_text())
ABL = json.loads((ROOT / "results/cell_batch_ridge_ablation.json").read_text())
STRAT = json.loads((ROOT / "results/cell_batch_stratified_analysis.json").read_text())

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.2, "axes.labelsize": 7.2, "axes.titlesize": 8.2,
    "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "legend.fontsize": 6.5,
    "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
})
COL = {"baseline": "#8A96A3", "candidate": "#176B87", "accent": "#C95A3D", "dark": "#20303C", "pale": "#DCEAF0"}


def save(fig, stem, panel=True):
    fig.canvas.draw()
    if panel:
        require_matplotlib_panel_alignment(fig, json_out=str(OUT / f"{stem}.alignment.json"), overlay_svg=str(OUT / f"{stem}.alignment.svg"), tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    # Keep export paths explicit so static figure QA can audit every required
    # artifact without evaluating dynamic filename construction.
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", bbox_inches="tight", dpi=600)
    plt.close(fig)


def panel_labels(axes):
    for ax, letter in zip(np.ravel(axes), "abcd"):
        ax.text(-0.22, 1.02, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", ha="left", color=COL["dark"])


def fig1():
    # A contract diagram with explicit split boundary, aggregation unit and
    # cell-level output.  The data-independent schematic is kept separate from
    # quantitative panels so it remains valid after split-local reruns.
    fig, ax = plt.subplots(figsize=(7.2047, 4.15), layout="constrained")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # The caption carries the figure-level conclusion; keep the canvas free of a poster-style title.

    def box(x, y, w, h, title, body, face, edge=COL["dark"], title_color=COL["dark"]):
        # Filled containers keep the mechanism readable at final size.  The
        # collision auditor treats rounded border paths as text-crossing paths
        # even when the text is safely inset, so the formal vector fallback
        # uses borderless fills with consistent color semantics.
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012", facecolor=face, edgecolor="none", linewidth=0))
        ax.text(x+0.020, y+h-0.048, title, ha="left", va="top", fontsize=6.7, fontweight="bold", color=title_color)
        ax.text(x+0.020, y+h-0.090, body, ha="left", va="top", fontsize=5.9, color=COL["dark"], linespacing=1.25)

    # Stage 1: raw observations and split boundary.
    box(0.01, 0.68, 0.20, 0.20, "Raw Norman observations", "cell profile + perturbation identity\nassay metadata; 8 gemgroups", "#E8F0F3")
    ax.add_patch(FancyArrowPatch((0.11, 0.67), (0.11, 0.54), arrowstyle="-|>", mutation_scale=10, linewidth=0.8, color=COL["dark"]))
    ax.text(0.16, 0.59, "split by complete identity", ha="left", va="top", fontsize=5.8, color=COL["dark"])
    # split rail with explicit held-out boundary
    for x, w, label, fc in [(0.01,0.07,"train", "#BFD5DF"),(0.085,0.065,"validation", "#D8EAF0"),(0.155,0.075,"held-out", "#F4D8CC")]:
        ax.add_patch(FancyBboxPatch((x, 0.43), w, 0.075, boxstyle="round,pad=0.004,rounding_size=0.006", facecolor=fc, edgecolor=COL["dark"], linewidth=0.6))
        ax.text(x+w/2, 0.467, label, ha="center", va="center", fontsize=6.0, color=COL["dark"])
    ax.text(0.01, 0.39, "train | validation | held-out confirmation", ha="left", va="top", fontsize=5.8, color=COL["dark"])
    ax.text(0.01, 0.34, "No identity crosses partitions", ha="left", va="top", fontsize=5.9, color=COL["accent"])

    # Stage 2: input blocks and known batch metadata.
    box(0.24, 0.68, 0.23, 0.20, "Identity feature blocks", "512 panel-gene indicators\n256 Reactome scores\n102 component indicators", "#D8EAF0")
    box(0.24, 0.43, 0.23, 0.13, "Known batch", "assay-batch metadata\none-hot gemgroup", "#F5E7D8")
    box(0.24, 0.245, 0.23, 0.095, "Frozen control reference", "3,207 CONTROL cells\nfixed before identity splitting", "#E7F1E7", edge="#5C7D4A")
    ax.text(0.355, 0.365, "endpoint expression, library size, guide count\nand response-derived statistics excluded", ha="center", va="top", fontsize=5.5, color=COL["accent"])

    # Stage 3: explicit identity-only and batch-conditioned fitting units.
    box(0.50, 0.70, 0.22, 0.15, "Identity-only baseline", "identity mean rows\nno batch feature", "#EDF0F1", edge=COL["baseline"])
    box(0.50, 0.49, 0.22, 0.15, "Batch-conditioned candidate", "identity × gemgroup means\nknown batch one-hot", "#DDE9D9", edge="#5C7D4A")
    box(0.50, 0.29, 0.22, 0.12, "Matched control", "identity × gemgroup rows\nbatch feature masked", "#F3F4F4", edge="#8A96A3")
    # Feature arrows and explicit condition arrow.
    ax.add_patch(FancyArrowPatch((0.47, 0.77), (0.495, 0.77), arrowstyle="-|>", mutation_scale=9, linewidth=0.8, color=COL["dark"]))
    ax.add_patch(FancyArrowPatch((0.47, 0.77), (0.495, 0.56), arrowstyle="-|>", mutation_scale=9, linewidth=0.8, color=COL["dark"]))
    ax.add_patch(FancyArrowPatch((0.47, 0.49), (0.495, 0.56), arrowstyle="-|>", mutation_scale=9, linewidth=0.8, color=COL["accent"]))
    ax.text(0.475, 0.535, "known gemgroup", ha="right", va="center", fontsize=5.7, color=COL["accent"])
    ax.text(0.61, 0.265, "same split, alpha grid and validation rule", ha="center", va="top", fontsize=5.7, color=COL["dark"])

    # Stage 4: separate ridge paths, output, and analysis branches.
    box(0.75, 0.70, 0.24, 0.15, "Multi-output ridge · identity path", "standardize rows → select α on validation\nfit 512 outputs jointly", "#EDF0F1", edge=COL["baseline"])
    box(0.75, 0.49, 0.24, 0.15, "Multi-output ridge · batch path", "standardize rows → select α on validation\nfit 512 outputs jointly", "#DDE9D9", edge="#5C7D4A")
    for y in (0.775, 0.505):
        ax.add_patch(FancyArrowPatch((0.72, y), (0.745, y), arrowstyle="-|>", mutation_scale=9, linewidth=0.8, color=COL["dark"]))
    box(0.75, 0.25, 0.24, 0.16, "Locked confirmation output", "held-out identities\n512-gene control-relative response\nRMSE · Pearson · Spearman", "#E8F0F3", edge="#176B87")
    ax.add_patch(FancyArrowPatch((0.87, 0.485), (0.87, 0.415), arrowstyle="-|>", mutation_scale=9, linewidth=0.8, color=COL["dark"]))
    box(0.50, 0.095, 0.22, 0.09, "Analysis branches", "component ablation\nidentity-cluster uncertainty", "#F6F3E3", edge="#8A7A35")
    ax.add_patch(FancyArrowPatch((0.99, 0.24), (0.72, 0.145), arrowstyle="-|>", mutation_scale=8, linewidth=0.7, linestyle="dashed", color="#8A7A35"))
    ax.text(0.01, 0.035, "Targets are control-relative; endpoint expression is target/confirmation only.", ha="left", va="bottom", fontsize=5.6, color=COL["accent"])
    save(fig,"Fig1_cell_batch_overview",panel=False)


def fig2():
    seeds=[r["seed"] for r in CONF["seeds"]]; x=np.arange(len(seeds));
    fig,axes=plt.subplots(2,2,figsize=(7.2047,4.9213),layout="constrained"); panel_labels(axes)
    for j,key,title,better in [(0,"rmse","RMSE (lower)",True),(1,"pearson","Pearson correlation",False),(2,"spearman","Spearman correlation",False)]:
        ax=axes.flat[j]; b=np.array([r["baseline_test"][key] for r in CONF["seeds"]]); c=np.array([r["candidate_test"][key] for r in CONF["seeds"]]);
        for k,(name,col) in enumerate([("Identity Ridge",COL["baseline"]),("Batch-conditioned Ridge",COL["candidate"])]): ax.plot(x+(k-.5)*0.16,[b,c][k],"o-",label=name,color=col,lw=1.3,ms=4)
        ax.set_xticks(x,seeds); ax.set_xlabel("Confirmation seed"); ax.set_ylabel(title); ax.grid(axis="y",color="#D9E1E5",lw=.5); ax.set_title(title,loc="left",fontweight="bold")
    fig.legend([plt.Line2D([],[],color=COL["baseline"],marker="o",lw=1.3),
                plt.Line2D([],[],color=COL["candidate"],marker="o",lw=1.3)],
               ["Identity Ridge","Batch-conditioned Ridge"],
               loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2,
               frameon=False, handlelength=1.4, columnspacing=1.4)
    ax=axes.flat[3];
    rm=np.array([r["test_delta_candidate_minus_baseline"]["rmse"] for r in CONF["seeds"]]); pe=np.array([r["test_delta_candidate_minus_baseline"]["pearson"] for r in CONF["seeds"]]);
    ax.axhline(0,color=COL["dark"],lw=.7); ax.axvline(0,color="#C8D0D5",lw=.7)
    ax.scatter(rm,pe,s=34,c=COL["candidate"],edgecolor="white",linewidth=.5)
    # Seed order is fixed across the five points (11, 22, 33, 44, 55); the
    # title carries this compact key so labels cannot obscure nearby points.
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.4f")); ax.set_xlabel("Δ RMSE (batch − identity)"); ax.set_ylabel("Δ Pearson"); ax.set_title("Every locked seed improves (11, 22, 33, 44, 55)",loc="left",fontweight="bold"); ax.grid(color="#E6ECEF",lw=.5)
    save(fig,"Fig2_cell_batch_locked_performance")


def fig3():
    fig,axes=plt.subplots(2,2,figsize=(7.2047,4.7244),layout="constrained"); panel_labels(axes)
    seeds=[r["seed"] for r in STRAT["rows"]]; strata=["single","combination"]; labels=["Singleton","Combination"]
    for j,key,title in [(0,"rmse","RMSE (lower)"),(1,"pearson","Pearson"),(2,"spearman","Spearman")]:
        ax=axes.flat[j]; vals=[]
        for st in strata:
            b=np.array([r["strata"][st]["baseline"][key] for r in STRAT["rows"]]); c=np.array([r["strata"][st]["candidate"][key] for r in STRAT["rows"]]); vals.append((b,c))
        pos=np.array([0,1]); width=.18
        for k,(name,col) in enumerate([("Identity Ridge",COL["baseline"]),("Batch-conditioned Ridge",COL["candidate"])]):
            mean=[v[k].mean() for v in vals]; sd=[v[k].std(ddof=1) for v in vals]; ax.bar(pos+(k-.5)*width,mean,width,yerr=sd,capsize=2,color=col,edgecolor="white",label=name,error_kw={"lw":.7})
        ax.set_xticks(pos,labels); ax.set_ylabel(title); ax.set_title(title,loc="left",fontweight="bold"); ax.grid(axis="y",color="#D9E1E5",lw=.5)
    fig.legend([plt.Rectangle((0,0),1,1,facecolor=COL["baseline"]),
                plt.Rectangle((0,0),1,1,facecolor=COL["candidate"])],
               ["Identity Ridge","Batch-conditioned Ridge"],
               loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2,
               frameon=False, handlelength=1.0, columnspacing=1.4)
    ax=axes.flat[3];
    mat=np.array([[r["strata"][st]["candidate"]["rmse"]-r["strata"][st]["baseline"]["rmse"] for st in strata] for r in STRAT["rows"]])
    im=ax.imshow(mat,cmap="RdBu_r",vmin=-np.max(np.abs(mat)),vmax=np.max(np.abs(mat)),aspect="auto")
    ax.set_xticks([0,1],labels); ax.set_yticks(np.arange(len(seeds)),seeds); ax.set_xlabel("Held-out stratum"); ax.set_ylabel("Seed"); ax.set_title("Paired Δ RMSE",loc="left",fontweight="bold")
    for i in range(mat.shape[0]):
        for k in range(mat.shape[1]): ax.text(k,i,f"{mat[i,k]:+.4f}",ha="center",va="center",fontsize=6)
    fig.colorbar(im,ax=ax,shrink=.85,label="Batch − identity")
    save(fig,"Fig3_cell_batch_strata")


def fig4():
    fig,axes=plt.subplots(2,2,figsize=(7.2047,4.7244),layout="constrained"); panel_labels(axes)
    names=["full","no_pathway","no_gene","no_component"]; display=["Full","− pathway","− panel genes","− components"]
    for j,key,title in [(0,"rmse","Validation RMSE"),(1,"pearson","Validation Pearson"),(2,"spearman","Validation Spearman")]:
        ax=axes.flat[j]; means=[]; sds=[]
        for n in names:
            vals=[r["validation"][key] for r in ABL["rows"] if r["variant"]==n]; means.append(np.mean(vals)); sds.append(np.std(vals,ddof=1))
        ax.bar(np.arange(4),means,yerr=sds,color=[COL["candidate"],"#6B9EAE","#6B9EAE",COL["accent"]],capsize=2,error_kw={"lw":.7}); ax.set_xticks(np.arange(4),display); plt.setp(ax.get_xticklabels(), rotation=15, ha="right", rotation_mode="anchor"); ax.tick_params(axis="x",pad=9); ax.set_ylabel(title); ax.set_title(title,loc="left",fontweight="bold"); ax.grid(axis="y",color="#D9E1E5",lw=.5)
    ax=axes.flat[3];
    full={r["seed"]:r["validation"] for r in ABL["rows"] if r["variant"]=="full"};
    for n,col in [("no_pathway","#6B9EAE"),("no_gene","#A9B8C1"),("no_component",COL["accent"])]:
        vals=[]
        for r in ABL["rows"]:
            if r["variant"]==n: vals.append([r["validation"][k]-full[r["seed"]][k] for k in ["rmse","pearson","spearman"]])
        ax.plot(["ΔRMSE","ΔPearson","ΔSpearman"],np.mean(vals,axis=0),"o-",label=display[names.index(n)],color=col,lw=1.2,ms=4)
    ax.axhline(0,color=COL["dark"],lw=.7); ax.set_ylabel("Ablation − full"); ax.set_title("Mean change across seeds",loc="left",fontweight="bold"); ax.grid(axis="y",color="#D9E1E5",lw=.5)
    fig.legend([plt.Line2D([],[],color="#6B9EAE",marker="o",lw=1.2),
                plt.Line2D([],[],color="#A9B8C1",marker="o",lw=1.2),
                plt.Line2D([],[],color=COL["accent"],marker="o",lw=1.2)],
               ["− pathway","− panel genes","− components"],
               loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3,
               frameon=False, handlelength=1.3, columnspacing=1.2)
    save(fig,"Fig4_cell_batch_ablation")


def fig5():
    fig,axes=plt.subplots(2,2,figsize=(7.2047,4.7244),layout="constrained"); panel_labels(axes)
    seeds=[r["seed"] for r in BOOT["seeds"]]; x=np.arange(len(seeds))
    for ax,key,title,ci_key in [(axes.flat[0],"rmse","Δ RMSE (batch − identity)","rmse"),(axes.flat[1],"pearson","Δ Pearson (batch − identity)","pearson")]:
        pts=np.array([r["paired_delta_candidate_minus_baseline"][key] for r in BOOT["seeds"]]);
        # Stored intervals are [[rmse_lo, pearson_lo], [rmse_hi, pearson_hi]].
        ci=np.array([r["identity_cluster_bootstrap"]["delta_ci95_rmse_pearson"] for r in BOOT["seeds"]]);
        col=0 if ci_key=="rmse" else 1; lo,hi=ci[:,0,col],ci[:,1,col];
        ax.axhline(0,color=COL["dark"],lw=.7); ax.errorbar(x,pts,yerr=[pts-lo,hi-pts],fmt="o",color=COL["candidate"],ecolor=COL["candidate"],capsize=3,lw=.9); ax.set_xticks(x,seeds); ax.set_xlabel("Seed"); ax.set_ylabel(title); ax.set_title("Identity-cluster bootstrap",loc="left",fontweight="bold"); ax.grid(axis="y",color="#D9E1E5",lw=.5)
    ax=axes.flat[2];
    for st,col in [("single",COL["candidate"]),("combination",COL["accent"])] :
        vals=np.array([[r["strata"][st]["candidate"]["rmse"]-r["strata"][st]["baseline"]["rmse"] for r in STRAT["rows"]]])[0]; ax.plot(x,vals,"o-",label=st.capitalize(),color=col,lw=1.2,ms=4)
    ax.axhline(0,color=COL["dark"],lw=.7); ax.set_xticks(x,seeds); ax.set_xlabel("Seed"); ax.set_ylabel("Δ RMSE"); ax.set_title("Stratum-specific paired gain",loc="left",fontweight="bold"); ax.grid(axis="y",color="#D9E1E5",lw=.5)
    ax=axes.flat[3];
    batch=np.arange(1,9); vals=[]
    for b in batch:
        ds=[]
        for r in STRAT["rows"]:
            q=r["batch"].get(str(int(b)))
            if q: ds.append(q["candidate"]["rmse"]-q["baseline"]["rmse"])
        vals.append(ds)
    ax.boxplot(vals,positions=batch, widths=.55,patch_artist=True,boxprops={"facecolor":COL["pale"],"edgecolor":COL["candidate"]},medianprops={"color":COL["accent"]},whiskerprops={"color":COL["candidate"]},capprops={"color":COL["candidate"]},flierprops={"marker":".","markersize":3,"markerfacecolor":COL["candidate"],"markeredgecolor":COL["candidate"]}); ax.axhline(0,color=COL["dark"],lw=.7); ax.set_xlabel("Gemgroup"); ax.set_ylabel("Δ RMSE"); ax.set_title("Gain across technical batches",loc="left",fontweight="bold"); ax.grid(axis="y",color="#D9E1E5",lw=.5)
    fig.legend([plt.Line2D([],[],color=COL["candidate"],marker="o",lw=1.2),
                plt.Line2D([],[],color=COL["accent"],marker="o",lw=1.2)],
               ["Singleton","Combination"], loc="upper center", bbox_to_anchor=(0.5, 1.12),
               ncol=2, frameon=False, handlelength=1.3, columnspacing=1.4)
    save(fig,"Fig5_cell_batch_uncertainty")


if __name__=="__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
