# UV_VAE Deployment — File Manifest

Everything required to reproduce or re-run the full pipeline:
**parquet → VAE → latent → parametric UMAP → HDBSCAN → SigProfiler (`uv_only`)**

Parameters for every stage are in [`pipeline_parameters.md`](pipeline_parameters.md).

---

## Layout

```
UV_VAE_Deployment/
├── RUNNING_INFERENCE.md            apply the checkpoints to new data  ← start here
├── REBUILDING_MODELS.md            retrain VAE / UMAP / HDBSCAN from scratch
├── ACCESSING_ASSIGNMENTS.md        how to read coordinates + labels back per row
├── pipeline_parameters.md          parameter master sheet (all stages)
├── DEPLOYMENT_MANIFEST.md          this file — what every file is and why
├── uv_vae/                         installable core package
├── Early_Stopping_Tests/           training entry point
├── umap_hdbscan_sweep/             projection + clustering + signature stages, and
│                                   the full parameter-sweep code behind every choice
├── models/                         trained models and coordinates
├── results/                        cohort signature assignments
├── plots/                          cohort figures
└── docs/                           design decision records
```

**Which document you want:**

| Goal | Read |
|---|---|
| Run the pipeline on new parquets | [`RUNNING_INFERENCE.md`](RUNNING_INFERENCE.md) |
| Retrain any stage from scratch | [`REBUILDING_MODELS.md`](REBUILDING_MODELS.md) |
| Query UMAP coordinates + cluster labels per row | [`ACCESSING_ASSIGNMENTS.md`](ACCESSING_ASSIGNMENTS.md) |
| Look up a hyperparameter or a reported metric | [`pipeline_parameters.md`](pipeline_parameters.md) |
| Find out what a particular file is | this file |
| Understand why a design choice was made | [`docs/`](docs/) |

The directory nesting matters. Every script locates the core package by walking
up its own path until it finds a directory named `uv_vae/`. Keeping
`umap_hdbscan_sweep/` and `Early_Stopping_Tests/` as siblings of `uv_vae/` is
what makes `import uv_vae.inference` resolve. Moving a script to a different
depth breaks the import silently.

---

## 1. `uv_vae/` — core package

The only code meant to be reused; everything else imports from it.

| File | Why it is included |
|---|---|
| `pyproject.toml`, `uv.lock`, `.python-version` | Pins the exact dependency set. `uv sync` reproduces the environment; torch is pinned to the CUDA-12.8 index on Linux. |
| `ml_features.json` | **Feature specification** — the 11 categorical + 29 numeric columns, their types, and the categorical value→code maps. Passed explicitly as `--feature-spec-path` at every stage. Without it the model cannot be applied to new data. |
| `uv_vae/features.py` | Parses `ml_features.json` into `FeatureSpec` objects. |
| `uv_vae/data.py` | All DuckDB access: row filtering, reservoir sampling, Arrow batch streaming, row counts. |
| `uv_vae/preprocess.py` | Numeric standardisation and categorical encoding. Holds the NULL→mean→0.0 imputation and the mask tensor that flags imputed values. Must match between training and inference or the latent space is invalid. |
| `uv_vae/model.py` | `TabularVAE` + `VAEConfig` — the architecture the checkpoint deserialises into. |
| `uv_vae/training.py` | Stock training loop, `seed_everything()`, loss computation. Source of truth for the maths; the other trainers reuse it rather than forking it. |
| `uv_vae/early_stopping.py` | In-RAM trainer adding per-dimension KL, posterior variance, and active-unit count (Burda et al. 2016). |
| `uv_vae/streaming.py` | Streaming trainer for flat memory on the full dataset. AMP, LR warmup, windowed shuffle. |
| `uv_vae/multi_parquet.py` | The 95-file interleaved reader — proportional per-file batch allocation and shuffled row-group order. This is how the cohort model was trained. |
| `uv_vae/multi_streaming.py` | Trainer that drives the interleaved reader. |
| `uv_vae/splitting.py` | Content-hash train/val predicates. The positional split cannot survive interleaved reading; this keeps every read at a locus on one side. |
| `uv_vae/stats_cache.py` | Per-file row counts and normalisation statistics in one scan, cached on disk. Preflight for the cohort run. |
| `uv_vae/inference.py` | `LatentInference.from_checkpoint()` — encodes new data through the trained model. Used by every downstream stage. |
| `uv_vae/gpu_budget.py`, `uv_vae/gpu_decode.py` | VRAM budgeting (torch vs. RMM pool split) and the optional cuDF/cuPy GPU encode path. |
| `uv_vae/convergence.py`, `uv_vae/latent_metrics.py`, `uv_vae/evaluation.py` | Latent-space diagnostics: Procrustes, CKA, trustworthiness, active units, collapse detection. |
| `uv_vae/pipeline_vae.py`, `uv_vae/cli.py`, `uv_vae/train_cli.py` | Orchestration and argparse entry points. |
| `scripts/gpu_preflight.py` | Probes the GPU and writes the budget plan before a long run. |
| `scripts/multi_parquet_stats.py` | The statistics preflight — run once before the first cohort run; its cache is reused. |
| `scripts/combine_parquets.py` | Merges per-sample parquets when a single input is wanted. |
| `scripts/run_variant_cluster_pipeline.py` | Clustering pipeline against an existing checkpoint. |
| `scripts/run_full_pipeline.sh` | End-to-end SLURM job: combine → train → cluster → SigProfiler. |
| `scripts/export_clustered_vcf.py`, `scripts/export_vcf.sh` | Exports cluster-labelled variants back to VCF. |
| `scripts/tmux_lib.sh` | Shared helpers for the tmux runners (miletus has no SLURM). |

## 2. `Early_Stopping_Tests/` — training entry point

| File | Why it is included |
|---|---|
| `Python Files/train_with_early_stopping.py` | **The unified training CLI.** `--parquet-paths` (globs) selects the interleaved trainer used for the cohort model; otherwise `--parquet-path` plus `--streaming` / `--use-all` / `--sample-rows` picks one of the other three modes. |
| `scripts/tmux_train_multi.sh` | The runner that produced the final cohort model. Also runs the statistics preflight with `STATS_ONLY=1`. |
| `scripts/tmux_train_only.sh`, `scripts/run_train_only*.sh` | Single-parquet training variants. |

## 3. `umap_hdbscan_sweep/` — projection, clustering, signatures

54 modules, 15 runners, and the test suite. The first 22 below are the
production pipeline (the transitive import closure of the stages); the rest are
the parameter sweeps, scaling studies, and plotting code that justify every
parameter in `pipeline_parameters.md` — included so the choices can be audited
and re-run, not just read.

**Pipeline stages, in run order:**

| File | Why it is included |
|---|---|
| `stage0_dedup.py` | Deduplicates reads to one row per locus before embedding. |
| `stage1_embed.py` | **VAE encode** — runs the checkpoint over the cohort and writes `latent.npy` (16-D µ) plus `context.parquet` (CHROM/POS/REF/ALT/trinucleotide context) that the signature stage needs. |
| `parametric_umap.py` | The parametric UMAP encoder — a neural network trained to approximate the UMAP embedding function so new rows project in one forward pass. |
| `stage2_sweep.py`, `parametric_sweep.py` | Fits the encoder across the size × parameter grid and scores each cell. Produced the selected 25M / nn15 / md0.1 configuration. |
| `stage3_apply_full.py`, `apply_parametric_full.py` | **Applies the fitted encoder to the full cohort** (157.5 M rows → 2-D coordinates). |
| `hdbscan_param_sweep.py` | **The cohort clustering stage.** Fits HDBSCAN on 1 M UMAP coordinates per grid cell, labels the whole cohort, saves the model + `fit_indices.npy`, builds the per-cluster SBS96 matrix, and runs SigProfilerAssignment against `uv_only`. The shipped model is its `fit1000000_mcs2500_ms15_eom` cell (mcs 2500, ms 15, ε 0.0). |
| `fast_predict.py` | **The RBC (Random Ball Cover) approximate-predict backend.** Precomputes a ball-cover index over the fit points so labelling new rows is sub-linear instead of O(n · fit_size). This is what makes per-sample labelling take seconds instead of minutes. |
| `per_parquet_inference.py` | **Per-sample inference.** For each parquet: VAE encode → parametric UMAP → HDBSCAN label via `fast_predict` → SigProfiler → plots. Streams in 5 M-row batches so peak memory is bounded by batch size, not sample size. Dynamically imports `run_variant_cluster_pipeline` inside its SigProfiler subprocess worker. |
| `export_assignments.py` | Collects the per-sample assignments into a manifest, re-checking every source fingerprint; `--attach` materialises enriched parquets. |
| `build_enriched_views.py` | Builds a DuckDB database of views joining each source to its assignments — the merged view without the copy. See [`ACCESSING_ASSIGNMENTS.md`](ACCESSING_ASSIGNMENTS.md). |
| `verify_saved_models.py` | Loads a saved cell from disk and proves it still labels correctly: pickle loads, `fit_indices` match, held rows reproduce. |
| `attach_row_ids.py` | Backfills `file_row_number` into the per-sample results so coordinates and labels join back to source parquet rows. |
| `verify_source_fingerprint.py` | Checks size / mtime / footer SHA-256 before any join is trusted — detects silent source-file drift. |
| `sweep_core.py` | Shared plumbing: paths, GPU budget wiring, checkpoint loading. Imported by nearly every stage. |
| `aumap.py` | UMAP embedding comparison used by the parametric encoder during fitting. |
| `umap_metrics.py`, `clustering_metrics.py`, `cluster_quality.py`, `cross_size_ari.py`, `rescore_dbcv.py` | Quality scoring: trustworthiness, continuity, RNX, KNN overlap, ARI, DBCV. Imported by the stages above; needed for the runs to complete. |
| `assignment_metrics.py`, `spectrum_metrics.py` | SBS96 spectrum construction and SigProfiler assignment scoring, including the vectorised 625-entry trinucleotide lookup table. |
| `signature_db/uv_only_SBS_GRCh38.tsv` | **The `uv_only` signature reference.** The lab-specific database every SigProfiler call is run against — not full COSMIC v3.5. |

**Runners:**

| File | Why it is included |
|---|---|
| `tmux_per_parquet_inference.sh` | Runs the 95-sample inference sweep. Also the clearest documentation of the runtime inputs — every path the pipeline needs appears as an overridable environment variable at the top. |
| `tmux_final_models.sh` | Fits and saves the selected cell (or the full grid with `FULL_GRID=1`) for both backends. |
| `tmux_attach_row_ids.sh` | Runs the row-id backfill. |
| `run_stage2_16gb.sh`, `run_parametric_sweep.sh`, `save_final_models.sh` | UMAP fitting sweep and model export. |

**Sweep and analysis code (35 modules):**

| Group | Modules | Why it is included |
|---|---|---|
| UMAP selection | `parameter_sweep`, `size_convergence`, `seed_stability`, `stability_sweep`, `global_structure`, `compare_cells` | The size × parameter grid and the seed-stability replicates that selected 25M / nn15 / md0.1. |
| HDBSCAN selection | `hdbscan_scaling_sweep`, `clustering_regime_sweep`, `aggregate_clustering_regime`, `condensed_tree_report`, `cohort_cluster_report` | The fit-size scaling study (500 K → 50 M) and the regime comparison that selected mcs 2500 / ms 1 / ε 0.05. The 50 M cell is where the GPU ran out of memory. |
| Cluster analysis | `cluster_profiles`, `cluster_quality`, `feature_discrimination`, `clustering_core` | Per-cluster feature profiling and quality scoring. |
| Row recovery | `recover_source_columns`, `label_dedup_source`, `verify_enriched` | Joining results back to source parquet rows and verifying the join. |
| Plotting (11) | `plot_*` | Every figure in the analysis: feature atlas, cluster dominance, sample enrichment, cosine similarity, sweep comparison. |
| Other | `build_stratified_embed`, `make_test_dataset`, `recover_source_columns` | Stratified embedding construction, held-out test set, and recovery of source identity by fingerprint join when it has been lost. |
| `tests/` | 12 files | Unit tests for the sweep and assignment code. |

> Shell scripts have already been converted to LF line endings. If they are edited
> on Windows again, re-run `sed -i 's/\r$//' <script>.sh` before submitting.

## 4. `models/` — trained models and coordinates

| File | Size | Why it is included |
|---|---|---|
| `vae/model.pt` | 622 KB | **The trained VAE.** 5.08 B rows, 38 epochs, best epoch 30, all 16 latent units active. Every downstream stage starts here. |
| `vae/run_20260802T192814Z/*.json` | ~180 KB | Run artifacts: `feature_report` (which features survived), `preprocess_report` (the normalisation statistics — required to encode new data identically), `training_report`, `diagnostics_report` (per-epoch KL and active units), `summary`. |
| `vae/sampling_plan.json` | 40 KB | Per-file batch allocation for the 95-parquet interleave. Documents how the training set was assembled. |
| `vae/gpu_preflight.json` | 2 KB | GPU budget plan for the training run. |
| `vae/embed_summary.json` | 2 KB | Provenance for the cohort embedding — row counts and checkpoint hash. |
| `umap/13_BEST_25M_nn15_md0.1_umap.pt` | 408 KB | **The parametric UMAP encoder.** Fit on 25 M rows; applies to any new row in one forward pass. |
| `umap/final_models_README.md` | 4 KB | Describes the eight candidate encoders and why this one was selected. |
| `hdbscan/hdbscan_model.pkl` | 164 MB | **The fitted cohort HDBSCAN model** — sweep cell `fit1000000_mcs2500_ms15_eom`, cuML, 1 M fit rows, **175 clusters**. Reused for all 95 samples via `fast_predict` so per-sample labels stay comparable to the cohort run. **Requires cuML to unpickle** (`cuml.cluster.hdbscan.HDBSCAN`); a CPU-only machine can read every other file here but cannot load this one. |
| `hdbscan/fit_indices.npy` | 8.0 MB | **The 1 M row indices the model was fit on**, into the 157.5 M cohort. `fast_predict` indexes the fit set positionally, so pairing the model with the wrong index file silently labels every read against the wrong neighbours. |
| `hdbscan/metrics.json` | 12 KB | Fit metrics: cluster count, noise fraction, timings, per-cluster statistics, and the backend that ran. |
| `hdbscan/cluster_persistence.npy` | 1.5 KB | Per-cluster persistence, shape `(175,)`. cuML returns a degenerate 1.0 for every cluster — the real distribution is only available from the CPU refit. |
| `coords/vae_latent_16d.npy` | 9.4 GB | **VAE latent coordinates** — the 16-D µ vector for all 157.5 M deduplicated rows. The VAE's actual output. |
| `coords/umap_coords_2d.npy` | 1.2 GB | **UMAP projection coordinates** — the 2-D embedding of the same rows. What HDBSCAN clusters and what every plot renders. |
| `coords/cohort_labels.npy` | 601 MB | HDBSCAN cluster label per row (−1 = noise), 175 clusters, 7.3553 % noise. Aligns index-for-index with the two coordinate arrays. |
| `coords/cohort_probabilities.npy` | 601 MB | Cluster membership probability per row, float32. Fit rows come from `clusterer.probabilities_`, the other 156.5 M from `fast_predict`. |
| `coords/context.parquet` | 563 MB | CHROM, POS, REF, ALT and trinucleotide context per row. Required to turn cluster labels into SBS96 spectra — the coordinates alone cannot produce a mutational signature. |

All five arrays in `coords/` are **row-aligned by position**: row *i* of
`vae_latent_16d.npy`, `umap_coords_2d.npy`, `cohort_labels.npy`,
`cohort_probabilities.npy`, and `context.parquet` describe the same read. That
alignment is only valid for this particular stage-0 dedup output — re-running
dedup produces a different row order and invalidates the join.

## 5. `results/` — signature assignments

| Path | Why it is included |
|---|---|
| `cohort/sigprofilerassignment_uv_only_grch38_v3.5/` | **The cohort result.** Per-cluster signature activities, decomposed mutation-type probabilities, and solution statistics for all **175 clusters**, assigned against `uv_only`. `input/cluster_sbs96_matrix.tsv` is the 96 × 175 spectrum matrix that was fed in, covering 145,916,931 mutations. |
| `cohort/cluster_labels.parquet` | Cluster label per row in parquet form, for the feature-atlas join. Generated from `cohort_labels.npy`; the sweep itself writes only the `.npy`. |

The SBS96 matrix covers every deduplicated row **except noise**: 157,501,580 rows
minus 7.3553 % noise leaves the 145.9 M mutations above. Noise gets no cluster
column, so the totals will never equal the full row count.

Per-sample results are **not** in this folder — the 95 samples produce ~104 GB
(43 GB of it `row_assignments.parquet`). They are distributed separately. See
[`ACCESSING_ASSIGNMENTS.md`](ACCESSING_ASSIGNMENTS.md) for how to read the
per-sample coordinates and labels back without copying them.

### Provenance

Everything in `models/` and `results/` comes from one sweep cell —
`sweep_both_backends/cuml/cells/fit1000000_mcs2500_ms15_eom` — and is mutually
consistent at 175 clusters / 7.3553 % noise. Verified with
`verify_saved_models.py`: the pickle loads from disk, `labels_` matches
`cohort_labels.npy` at `fit_indices`, and 200,000 held rows re-label exactly as
the sweep recorded.

Note also that cuML's HDBSCAN is **not bit-reproducible across runs**: two fits
on identical input, same seed and same parameters, hold the cluster count stable
but move the noise boundary by roughly 0.3 percentage points. Refitting to
"reproduce" a shipped model will not return the shipped labels — use the saved
`hdbscan_model.pkl` and `fit_indices.npy` instead.

## 5b. Which code is load-bearing

Of the Python in this folder, **19 files are required to run the pipeline**; the rest is the
sweep and analysis code kept for provenance — the evidence behind each parameter choice.

| Location | Required files |
|---|---|
| `umap_hdbscan_sweep/` | `per_parquet_inference`, `fast_predict`, `parametric_umap`, `aumap`, `sweep_core`, `export_assignments`, `build_enriched_views`, `verify_saved_models`, `verify_source_fingerprint`, `cross_size_ari` |
| `uv_vae/uv_vae/` | `__init__`, `features`, `data`, `model`, `preprocess`, `inference`, `training`, `gpu_budget` |
| `uv_vae/scripts/` | `run_variant_cluster_pipeline` |

Plus one data file: `umap_hdbscan_sweep/signature_db/uv_only_SBS_GRCh38.tsv`, the `uv_only`
reference SigProfiler assigns against.

Three of those are invisible to a dependency scanner and must not be pruned:

- **`run_variant_cluster_pipeline.py`** is imported *dynamically inside the SigProfiler
  subprocess worker* (`import run_variant_cluster_pipeline as rvcp`). Nothing references it
  statically. Without it a run proceeds normally until the first SigProfiler step, then every
  worker fails.
- **`gpu_budget.py`** is imported lazily inside `sweep_core.apply_gpu_budget()`, not at module
  level.
- **`__init__.py`** is the package marker; `import uv_vae` fails without it.



---

## 6. `docs/` — decision records

| File | Why it is included |
|---|---|
| `MULTI_PARQUET_LOADING.md` | What the interleaved loader does, what was measured, what is still unverified. |
| `SAMPLING_STRATEGY.md` | Why proportional interleaving was chosen and the six alternatives were rejected. |
| `TMUX_RUNNERS.md` | How to run each stage on miletus (no SLURM there). |

---

## Deliberately excluded


- **`cohort_probabilities.npy`** (601 MB) — HDBSCAN membership strengths, one float per row. The hard labels in `cohort_labels.npy` carry the result; this is only needed to threshold membership differently.
- **`sbs96_index.npy`** (151 MB) — a cache mapping each row to its SBS96 channel. Rebuilt automatically from `context.parquet` on first use.
