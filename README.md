# uv-vae

## Usage

`[run.sh](/home/carlos/Clone/uv_vae/scripts/run.sh)` is a fully expanded example command that sets every environment variable and then runs `[run_train_then_cluster.sh](/home/carlos/Clone/uv_vae/scripts/run_train_then_cluster.sh)`.

Run it as-is:

```bash
cd /home/carlos/Clone/uv_vae
bash scripts/run.sh
```

Run the pipeline script directly. The clustering row filter must be passed explicitly:

```bash
cd /home/carlos/Clone/uv_vae
bash scripts/run_train_then_cluster.sh --cluster-row-filter "st = 'MIXED' AND et = 'MIXED' AND FILT = 1"
```

Override selected environment variables inline:

```bash
cd /home/carlos/Clone/uv_vae
RUN_ROOT=/home/carlos/Clone/uv_vae/artifacts/my_run \
TRAIN_SAMPLE_ROWS=500000 \
CLUSTER_USE_ALL=1 \
DUCKDB_MEMORY_LIMIT=12GB \
bash scripts/run_train_then_cluster.sh --cluster-row-filter "st = 'MIXED' AND et = 'MIXED' AND FILT = 1"
```

## Required Script Argument

- `--cluster-row-filter`
  Required for `bash scripts/run_train_then_cluster.sh`.
  DuckDB SQL filter passed to the clustering pipeline. It is intentionally explicit and does not inherit the training filter default.

## Environment Variables / Shell Variables

Paths:

- `UV_PATH`
  Default: resolved from `command -v uv`
  Path to the `uv` executable used by `run.sh` and `run_train_then_cluster.sh`.
- `RUN_ROOT`
  Default: `artifacts/train_then_cluster_<utc timestamp>`
  Root directory for the combined training+clustering run. The script creates the training and clustering outputs under this path unless you override them separately.
- `PARQUET_PATH`
  Default: `/home/carlos/Clone/uv_vae/data/ddbR9-b2-ppm0029.featuremap.parquet`
  Input parquet used by both the training pipeline and the clustering pipeline.
- `FEATURE_SPEC_PATH`
  Default: `/home/carlos/Clone/uv_vae/ml_features.json`
  JSON feature specification consumed by the training pipeline when building tensors and the model input schema.
- `TRAIN_OUTPUT_DIR`
  Default: `${RUN_ROOT}/training`
  Output directory passed to the training pipeline. The resulting `run_*` directory with `model.pt` is created here.
- `CLUSTER_OUTPUT_ROOT`
  Default: `${RUN_ROOT}/clustering`
  Output root passed to the clustering pipeline. UMAP, HDBSCAN, SigProfiler, and plot artifacts are written here.
- `MODEL_PATH`
  Default: unset
  Optional existing `model.pt` checkpoint. When set, `run_train_then_cluster.sh` skips training and starts from clustering.
- `CLUSTER_ROW_FILTER`
  Example default in `scripts/run.sh`: `st = 'MIXED' AND et = 'MIXED' AND FILT = 1`
  Shell variable used by `scripts/run.sh` when it calls `bash scripts/run_train_then_cluster.sh --cluster-row-filter "$CLUSTER_ROW_FILTER"`.
  This is not read implicitly by `run_train_then_cluster.sh`; the script only uses the explicit CLI argument.

Reproducibility:

- `PIPELINE_SEED`
  Example default in `scripts/run.sh`: `42`
  Convenience seed used by `scripts/run.sh` to set both `TRAIN_SEED` and `CLUSTER_SEED`.
- `PYTHONHASHSEED`
  Default: `${TRAIN_SEED}` in `run_train_then_cluster.sh` when not already set.
  Python hash seed exported before Python subprocesses start.
- `CUBLAS_WORKSPACE_CONFIG`
  Default: `:4096:8` in `run_train_then_cluster.sh` when not already set.
  CUDA workspace setting used by PyTorch deterministic mode.
- `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`
  Default: `1` in `run_train_then_cluster.sh` when not already set.
  CPU numerical-library thread counts pinned for stricter reproducibility.

Training:

- `TRAIN_ROW_FILTER`
  Default: `st = 'MIXED' AND et = 'MIXED' AND FILT = 1`
  DuckDB SQL filter applied before training-row counting and sampling.
- `TRAIN_USE_ALL`
  Default: `0`
  If `1`, training uses all filtered rows.
- `TRAIN_SAMPLE_ROWS`
  Default: `1000000`
  Used only when `TRAIN_USE_ALL=0`.
  Target number of filtered rows sampled for VAE training.
- `TRAIN_EPOCHS`
  Default: `10`
  Number of optimization epochs for the VAE.
- `TRAIN_BATCH_SIZE`
  Default: `4096`
  PyTorch batch size used during training and validation.
- `TRAIN_LATENT_DIM`
  Default: `16`
  Latent dimensionality of the VAE bottleneck.
- `TRAIN_HIDDEN_DIMS`
  Default: `256,128`
  Comma-separated hidden layer widths for the encoder/decoder MLP.
- `TRAIN_LEARNING_RATE`
  Default: `1e-3`
  AdamW learning rate.
- `TRAIN_KL_WEIGHT`
  Default: `0.05`
  Weight applied to the VAE KL-divergence term in the total loss.
- `TRAIN_FRACTION`
  Default: `0.9`
  Fraction of sampled rows assigned to the training split; the remainder goes to validation.
- `TRAIN_SEED`
  Default: `42`
  Random seed used for row sampling, train/validation splitting, and PyTorch/NumPy seeding.
- `TRAIN_THREADS`
  Default: `12`
  DuckDB thread count used while scanning/counting the training parquet.

Clustering:

- `CLUSTER_SEED`
  Default: `${TRAIN_SEED}`
  Random seed passed to the clustering pipeline for deduplicated sampling, UMAP-fit sampling, UMAP initialization, and plot sampling.
- `CLUSTER_USE_ALL`
  Default: `0`
  If `1`, clustering uses the full deduplicated population.
- `CLUSTER_SAMPLE_ROWS`
  Default: `1000000`
  Used only when `CLUSTER_USE_ALL=0`.
  Number of deduplicated variants sampled before embedding and clustering.
- `CLUSTER_THREADS`
  Default: `12`
  DuckDB thread count for deduplication, aggregation queries, and parquet reads in the clustering pipeline. Seeded sampling steps force DuckDB to one thread for reproducibility.
- `DEVICE`
  Default: `auto`
  Inference device for latent embedding generation. `auto` selects CUDA when available, otherwise CPU.
- `EMBED_BATCH_SIZE`
  Default: `4096`
  Model batch size used when encoding deduplicated rows into latent vectors.
- `SCAN_BATCH_ROWS`
  Default: `100000`
  Number of parquet rows streamed per batch in embedding and analysis-transform stages.
- `UMAP_FIT_ROWS`
  Default: `100000`
  Size of the latent subset used to fit the 2D UMAP model.
- `PLOT_ROWS`
  Default: `100000`
  Size of the sampled subset used for the global UMAP visualization panels.
- `UMAP_N_NEIGHBORS`
  Default: `30`
  `n_neighbors` parameter passed to UMAP.
- `UMAP_MIN_DIST`
  Default: `0.05`
  `min_dist` parameter passed to UMAP.
- `UMAP_METRIC`
  Default: `euclidean`
  Distance metric used by UMAP in latent space.
- `HDBSCAN_MIN_CLUSTER_SIZE`
  Default: `250`
  Minimum cluster size for HDBSCAN on the UMAP coordinates.
- `HDBSCAN_MIN_SAMPLES`
  Default: `25`
  `min_samples` parameter for HDBSCAN density estimation.
- `SIGPROFILER_CPU`
  Default: `12`
  CPU worker count passed to SigProfilerAssignment.
- `COSMIC_VERSION`
  Default: `3.5`
  COSMIC signature version used when building the UV-only signature reference for SigProfilerAssignment.
- `GENOME_BUILD`
  Default: `GRCh38`
  Reference genome build passed to SigProfilerAssignment.
- `DUCKDB_MEMORY_LIMIT`
  Default: `4GB`
  DuckDB memory cap for the clustering pipeline so large dedup/sample queries spill to disk instead of exhausting RAM.
- `COLOR_COLUMNS`
  Default: `BCSQ,RAW_VAF,DP,SMQ_BEFORE,SMQ_AFTER,EDIST,MAPQ,SNVQ`
  Comma-separated numeric metadata columns rendered as global UMAP color panels.
