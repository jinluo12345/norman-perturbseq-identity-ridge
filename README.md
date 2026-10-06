# Auditable Norman–Weissman Perturb-seq response prediction

This release contains the exact processed inputs, numerical result objects, analysis scripts, figure assets, manuscript source, and native image2 provenance used for the manuscript **Prioritizing gene and combination assays in the Norman–Weissman K562 Perturb-seq screen**.

The scientific estimand is deliberately bounded: identity-held-out prediction of a 512-gene control-relative response in the Norman–Weissman K562 screen, conditional on observed gemgroup metadata. The retrospective top-*k* analysis is a validation-only ranking simulation. The release does not claim causal pathway activation, unseen-batch transfer, clinical benefit, or prospective assay success.

## Source data

The primary public source is the Norman–Weissman Perturb-seq study, GEO accession **GSE133344** ([GEO record](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344); study DOI [10.1038/s41586-019-1431-3](https://doi.org/10.1038/s41586-019-1431-3)). Raw h5ad files are intentionally not duplicated here. `data/processed/input_checksums.json` records the source-file checksums used to produce the released processed panel. Reactome pathway annotations are identified in `data/processed/metadata.json` and `provenance/panel_manifest.json`.

## Contents

- `data/processed/cell_panel.npz`: frozen 59,879-cell, 512-gene response panel with source row indices and parsed identities.
- `data/processed/pathway_scores_control_weighted.npz`: frozen identity-level Reactome score matrix.
- `results/`: final numerical bundles used by the manuscript and quantitative figures.
- `src/`: analysis and plotting code. All paths are relative to the release root.
- `figures/`: final quantitative figure exports and native image2 concept panels.
- `image2/`: prompt files and native generation records for non-quantitative panels.
- `manuscript/`: manuscript source, bibliography, and the built PDF.
- `provenance/`: environment lock, panel manifest, release audits, and reconstruction helper.

## Reconstruction

Use Python 3.12 with the pinned versions in `provenance/environment_lock.txt`. From the release root, the final quantitative panels can be regenerated with:

```bash
python src/plot_fig2_repaired_performance.py
python src/plot_fig3_repaired.py
python src/plot_fig4_repaired_ablation.py
python src/plot_fig5_repaired_upgrade.py
python src/plot_intervention_priority.py
```

The scripts read only the released `results/` objects and the native image2 panel in `figures/`. Rebuilding the complete processed panel from raw h5ad requires the public GSE133344 source and the preprocessing contract documented in `manuscript/main.tex`; the released processed arrays are the frozen inputs used for the reported analysis.

The manuscript is compiled from `manuscript/main.tex` after placing the released figures under the relative paths expected by the source. The release audits record the PDF hash and figure/source checksums.

The versioned archive URL for this release is https://github.com/jinluo12345/norman-perturbseq-identity-ridge/tree/v1.0.0. The repository commit history and `provenance/release_manifest.sha256` provide the immutable version and file-integrity record.

## License and citation

Code is released under the MIT License; derived arrays, figures and manuscript text are released under CC BY 4.0. Please cite the manuscript and the Norman–Weissman source study when reusing this release.
