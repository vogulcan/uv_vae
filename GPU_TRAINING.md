# GPU-Specific Training

Training the full 95-sample cohort requires a GPU with at least 16 GB of VRAM dedicated to PyTorch.
The runner handles environment activation, GPU budgeting, and tmux session management automatically.

## Quick start (miletus)

```bash
# 1. Verify all 95 files parse and print per-sample interleave weights (cheap, reuse cache)
STATS_ONLY=1 bash Early_Stopping_Tests/scripts/tmux_train_multi.sh

# 2. Run training with the final configuration used for the cohort model
EPOCH_CEILING=40 EPOCH_SHARDS=20 PATIENCE=8 BATCH_SIZE=1048576 \
DECODE_WORKERS=8 KL_WEIGHT=0.005 INPUT_DROPOUT=0.1 HIDDEN_DROPOUT=0.1 \
bash Early_Stopping_Tests/scripts/tmux_train_multi.sh
```

## Key environment variables

| Variable | Default | Effect |
|---|---|---|
| `PARQUET_GLOB` | auto-detected | Glob for input parquet files; miletus defaults to `/data/lab/ppmseq_parquets/*.featuremap.parquet` |
| `BATCH_SIZE` | 32768 | Rows per forward pass; final cohort run used **1,048,576** |
| `DECODE_WORKERS` | 1 | Parallel CPU workers decoding parquet batches; set to **8** for the 2.4x speed gain |
| `EPOCH_SHARDS` | 20 | Splits one full pass over 5B rows into N mini-epochs; early stopping checks between shards |
| `EPOCH_CEILING` | 40 | Hard cap on epochs; cohort run stopped at epoch 30 of 38 run |
| `PATIENCE` | 8 | Epochs without val ELBO improvement (and no active-unit change) before stopping |
| `KL_WEIGHT` | 0.05 | Beta in beta-VAE loss; final run used **0.005** to avoid posterior collapse |
| `UV_VAE_GPU_MEM_GB` | auto | Torch VRAM budget in GB; auto-detected, cohort run capped at 16 GB |
| `UV_VAE_ENABLE_CUML` | 0 | Set to 1 to use cuML/cuDF GPU acceleration for HDBSCAN and UMAP (requires RAPIDS) |
| `STATS_ONLY` | 0 | Set to 1 to run only the statistics pre-flight and stop before training |
| `SEED` | 42 | Controls model init, row shuffling, and train/val split |

## GPU decode (not recommended)

Setting `UV_VAE_GPU_DECODE=1` routes parquet decoding through cuDF on the GPU.
In practice this is slower than CPU decode -- the GPU was idle waiting for the CPU in CPU mode,
and flipping to GPU decode made the CPU wait for the GPU decode instead. The 8-worker CPU decode
path (`DECODE_WORKERS=8`) is 2.4x faster than the 1-worker baseline and was used for the final
cohort run.

## AMP and VRAM budgeting

AMP (Automatic Mixed Precision) is always on: forward pass and loss run in float16, gradients in float32.
A GPU preflight check runs before training and rejects configs that exceed the VRAM budget.
To skip it (e.g. if you have verified the config manually): `SKIP_PREFLIGHT=1`.

```bash
# Run preflight only without training
python uv_vae/scripts/gpu_preflight.py \
    --batch-size 1048576 \
    --budget-gb 16 \
    --feature-spec-path uv_vae/ml_features.json
```

---

# Inference-Only Passes

Use this when you have a trained VAE checkpoint and want to encode new parquet data without retraining.
All three model files (VAE, UMAP encoder, HDBSCAN model) must be from the same pipeline run to be consistent.

## Full per-sample inference (VAE -> UMAP -> HDBSCAN -> SigProfiler)

```bash
python umap_hdbscan_sweep/per_parquet_inference.py \
    --parquet-glob '/data/lab/ppmseq_parquets/*.parquet' \
    --checkpoint  <path-to>/run_20260802T192814Z/model.pt \
    --umap-model  <path-to>/13_BEST_25M_nn15_md0.1_umap.pt \
    --feature-spec uv_vae/ml_features.json \
    --coords      <path-to>/umap_coords_2d.npy \
    --context     <path-to>/context.parquet \
    --output-dir  results/per_parquet_inference
```

Each sample gets its own subdirectory under `--output-dir` containing cluster labels, SigProfiler
assignments, and four plots (UMAP coloured by cluster, substitution, SigProfiler cosine, and
per-sample coverage).

## VAE encode only (latent mu vectors)

Use `LatentInference.from_checkpoint` directly when you only need the 16-D latent vectors:

```python
from uv_vae.inference import LatentInference

inf = LatentInference.from_checkpoint(
    checkpoint_path="<path-to>/model.pt",
    feature_spec_path="uv_vae/ml_features.json",
    device="cuda",          # or "cpu"
)

# Encode one parquet file (streams in batches, returns concatenated mu)
mu = inf.encode_parquet(
    parquet_path="<path-to>/sample.parquet",
    row_filter="st = 'MIXED' AND et = 'MIXED' AND FILT = 1",
    batch_size=5_000_000,
)
# mu.shape == (n_rows, 16), dtype float32
```

## Checkpoint path layout

A run directory produced by any of the three trainers always contains:

```
run_YYYYMMDDTHHMMSSZ/
├── model.pt                 # VAE weights (load with LatentInference.from_checkpoint)
├── feature_report.json      # which features were active and their stats
├── preprocess_report.json   # standardisation means/stds per numeric feature
├── training_report.json     # loss curves, early-stopping diagnostics
└── summary.json             # single-line summary of the run
```

Point `--checkpoint` or `LatentInference.from_checkpoint` at `model.pt`. The loader reads the
sibling `feature_report.json` and `preprocess_report.json` automatically to reconstruct the exact
preprocessing that was applied during training.
