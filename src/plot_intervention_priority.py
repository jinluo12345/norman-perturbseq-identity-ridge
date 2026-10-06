from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
D = json.loads((ROOT / 'results/intervention_priority_simulation.json').read_text())
R = D['seeds']; ks = [5, 10, 20]; x = np.arange(len(ks))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,
                     'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
                     'axes.linewidth':0.7,'savefig.dpi':300})
blue, orange, teal, grey = '#2c5d8a', '#d9822b', '#238b8d', '#66727a'
fig, ax = plt.subplots(2, 2, figsize=(7.1, 5.0), constrained_layout=True)

for r in R:
    ax[0,0].plot(x, [r['top_k'][str(k)]['topk_overlap'] for k in ks], 'o-', color=blue, alpha=.28, lw=.8, ms=3)
mean = [np.mean([r['top_k'][str(k)]['topk_overlap'] for r in R]) for k in ks]
sd = [np.std([r['top_k'][str(k)]['topk_overlap'] for r in R], ddof=1) for k in ks]
ax[0,0].errorbar(x, mean, yerr=sd, color=blue, marker='o', lw=1.8, ms=4, capsize=2, label='observed mean +/- split SD')
null = np.asarray(ks, dtype=float) / 26.0
ax[0,0].plot(x, null, '--', color=grey, lw=1.1, label='random null k/26')
ax[0,0].set(xlabel='Number of assays selected (k)', ylabel='Retrospective top-k overlap', ylim=(0,1.05), xticks=x, xticklabels=ks)
ax[0,0].legend(frameon=False, loc='lower right', handlelength=2.2, borderpad=.2)
ax[0,0].text(-.14,1.04,'a', transform=ax[0,0].transAxes, weight='bold', fontsize=11)

for r in R:
    ax[0,1].plot(x, [r['top_k'][str(k)]['ndcg'] for k in ks], 'o-', color=orange, alpha=.28, lw=.8, ms=3)
mean = [np.mean([r['top_k'][str(k)]['ndcg'] for r in R]) for k in ks]
sd = [np.std([r['top_k'][str(k)]['ndcg'] for r in R], ddof=1) for k in ks]
ax[0,1].errorbar(x, mean, yerr=sd, color=orange, marker='o', lw=1.8, ms=4, capsize=2)
ax[0,1].set(xlabel='Number of assays selected (k)', ylabel='NDCG', ylim=(.75,1.02), xticks=x, xticklabels=ks)
ax[0,1].text(-.14,1.04,'b', transform=ax[0,1].transAxes, weight='bold', fontsize=11)

for r in R:
    ax[1,0].plot([0,1], [r['pathway_profile_spearman'], r['pathway_profile_cosine_mean']], 'o-', color=teal, alpha=.3, lw=.8, ms=3)
mean = [np.mean([r['pathway_profile_spearman'] for r in R]), np.mean([r['pathway_profile_cosine_mean'] for r in R])]
sd = [np.std([r['pathway_profile_spearman'] for r in R], ddof=1), np.std([r['pathway_profile_cosine_mean'] for r in R], ddof=1)]
ax[1,0].errorbar([0,1], mean, yerr=sd, color=teal, marker='o', lw=1.8, ms=4, capsize=2)
ax[1,0].set(xlabel='Pathway-level agreement', ylabel='Pathway agreement', ylim=(.75,1.0), xticks=[0,1], xticklabels=['Spearman','Cosine'])
ax[1,0].text(-.14,1.04,'c', transform=ax[1,0].transAxes, weight='bold', fontsize=11)

for j,k in enumerate(ks):
    vals = D['stability'][str(k)]['pairwise_jaccard']
    ax[1,1].scatter(np.full(len(vals), j), vals, color=grey, s=13, alpha=.65, edgecolor='none')
    ax[1,1].plot(j, np.mean(vals), 'o', color=blue, ms=5)
ax[1,1].set(xlabel='Priority list size (k)', ylabel='Pairwise Jaccard across splits', ylim=(0, .55), xticks=x, xticklabels=ks)
ax[1,1].text(-.14,1.04,'d', transform=ax[1,1].transAxes, weight='bold', fontsize=11)
for a in ax.ravel():
    a.grid(axis='y', color='#dce3e7', lw=.45); a.set_axisbelow(True)
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
fig.savefig(ROOT/'paper/figures_batch/Fig6_intervention_priority.pdf', bbox_inches='tight', pad_inches=.04)
fig.savefig(ROOT/'paper/figures_batch/Fig6_intervention_priority.svg', bbox_inches='tight', pad_inches=.04)
print('wrote Fig6_intervention_priority')
