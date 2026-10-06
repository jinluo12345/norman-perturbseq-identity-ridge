# Reproducibility metadata

This directory records the public release metadata for the Norman–Weissman
Perturb-seq analysis. The versioned archive is
https://github.com/jinluo12345/norman-perturbseq-identity-ridge/tree/v1.1.9.

The source experiment is identified by `../data/processed/input_checksums.json`
and GEO accession GSE133344. `panel_manifest.json` records the 512 output genes,
256 Reactome pathways and processed dimensions. `release_manifest.sha256`
records hashes for every public release file. `environment_lock.txt` records
observed package versions and is an environment report, not an installation
command.

The manuscript is under `../paper/`; final quantitative figures are generated
by the five plotting scripts listed in the top-level README. Native image2
records and prompts are under `../image2/`. Raw h5ad files are referenced by
accession and source checksums; frozen processed arrays are included in the
release.
