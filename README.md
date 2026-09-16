# uv_vae

Unsupervised clustering of single-read SNVs from ppmSeq feature-map parquets:

**parquet → VAE (16-D latent) → parametric UMAP (2-D) → HDBSCAN → SBS96 → SigProfiler `uv_only`**

This repository holds the code to **apply the trained models** to new samples and to
**retrain every stage** from the 95-sample cohort. It was developed during the PURE Summer
Program (Sabancı University) and run on the lab GPU node, miletus.

---

## Trained models are not in this repository

The model files are too large for git and are kept by the lab. On miletus they are in
`~/uv_vae_deployment/models/`. Every runner expects them at the same relative paths inside
this checkout:

| Path | What it is | Needed for |
|---|---|---|
| `models/vae/model.pt` | VAE checkpoint (feature and normalisation reports embedded) | inference; start of any UMAP/HDBSCAN refit |
| `models/umap/13_BEST_25M_nn15_md0.1_umap.pt` | parametric UMAP encoder | inference |
| `models/hdbscan/hdbscan_model.pkl` | cohort HDBSCAN model — **needs cuML to load** | inference |
| `models/hdbscan/fit_indices.npy` | the 1 M rows the HDBSCAN model was fit on | inference |
| `models/coords/umap_coords_2d.npy` | cohort UMAP coordinates | inference; HDBSCAN refit |
| `models/coords/context.parquet` | CHROM/POS/REF/ALT and trinucleotide context per cohort row | inference; HDBSCAN refit |
| `models/coords/vae_latent_16d.npy` | cohort VAE latent (9.4 GB) | only for refitting UMAP on the shipped VAE |

`fit_indices.npy` must stay paired with `hdbscan_model.pkl`, and every array in
`models/coords/` is row-aligned with the others. Copy them together.

---

## Layout

```
uv_vae/                   core package (pyproject.toml, uv.lock, ml_features.json)
  uv_vae/                 VAE model, data loading, multi-parquet GPU trainer, inference
  scripts/                preflights, clustering/SigProfiler helpers, tmux_lib.sh
  tests/
Early_Stopping_Tests/     training CLI and tmux_train_multi.sh (trained the cohort VAE)
umap_hdbscan_sweep/       dedup, VAE encode, UMAP fit/apply, HDBSCAN fit, per-sample inference
docs/                     design records: multi-parquet loading, sampling strategy, tmux runners
models/                   not tracked — place the trained models here (see above)
runs/, results/           not tracked — created by the runners
```

Keep this layout: scripts locate the package by walking up to the folder that contains
`uv_vae/`.

---

## Environment

A CUDA GPU and cuML are required for inference, because the HDBSCAN model is a cuML object.
On miletus:

```bash
micromamba activate uv_vae
```

Setting up elsewhere is covered in [`RUNNING_INFERENCE.md` §1](RUNNING_INFERENCE.md): cuML,
CuPy and RMM come from the `rapidsai` conda channel; everything else from
`uv_vae/pyproject.toml`.

---

## Run inference with the trained models

From the repository root, after placing `models/`:

```bash
tmux new-session -d -s inference 'bash umap_hdbscan_sweep/tmux_per_parquet_inference.sh'
```

The runner defaults to the files in `models/` and writes to
`results/per_parquet_inference_cuml/`. Point it at other samples with `PARQUET_GLOB=...`.
Details, outputs and checks: [`RUNNING_INFERENCE.md`](RUNNING_INFERENCE.md). Reading the
per-row coordinates and labels back: [`ACCESSING_ASSIGNMENTS.md`](ACCESSING_ASSIGNMENTS.md).

---

## Retrain

```bash
STATS_ONLY=1 bash Early_Stopping_Tests/scripts/tmux_train_multi.sh     # preflight, once
tmux new-session -d -s train_multi 'bash Early_Stopping_Tests/scripts/tmux_train_multi.sh'
```

That retrains the VAE with the shipped configuration. The full chain (dedup → VAE encode →
UMAP fit → UMAP apply → HDBSCAN fit) is in [`REBUILDING_MODELS.md`](REBUILDING_MODELS.md).
Retrained models are written to `runs/`, never over `models/`. Retraining any stage
invalidates every stage after it.

---

## The shipped models

| Stage | Configuration |
|---|---|
| VAE | 95 parquets, 5.08 B rows after `st = 'MIXED' AND et = 'MIXED' AND FILT = 1`; latent 16, hidden 256,128; batch 1,048,576; lr 1e-3; β 0.005; dropout 0.1/0.1; best epoch 30 of 38; 16/16 active units |
| UMAP | parametric encoder fit on 25 M rows; n_neighbors 15, min_dist 0.1; applied to 157.5 M deduplicated loci |
| HDBSCAN | cuML; 1 M fit rows; min_cluster_size 2500, min_samples 15, eom; 175 clusters, 7.36 % noise |
| Signatures | SigProfilerAssignment against the lab `uv_only` reference (GRCh38) |

Every parameter and reported metric: [`pipeline_parameters.md`](pipeline_parameters.md).

---

The original package README is kept at [`uv_vae/README.md`](uv_vae/README.md); its example
paths refer to an earlier setup.
