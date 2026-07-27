# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Tabular VAE pipeline for genomic variant analysis (PURE Summer Program, Sabanci University). Trains a VAE on parquet feature maps of genomic variants, encodes to latent space, projects with UMAP, clusters with HDBSCAN, and runs SigProfiler mutational signature assignment.

## Commands

```bash
# Run the full pipeline (training + clustering)
cd /path/to/uv_vae
bash scripts/run.sh

# Run pipeline with custom settings
TRAIN_SAMPLE_ROWS=5000000 TRAIN_EPOCHS=10 \
bash scripts/run_train_then_cluster.sh --cluster-row-filter "st = 'MIXED' AND et = 'MIXED' AND FILT = 1"

# Run subsample sweep (stability testing)
python VAE_Stability_Testing/scripts/vae_subsample_sweep.py \
    --parquet-path /path/to/featuremap.parquet \
    --test-set-path /path/to/test_set.parquet \
    --output-dir output/ \
    --max-sample-rows 5000000 \
    --subsample-fractions "1.0,0.5,0.25,0.1" \
    --epochs 10 --seed 42 --data-seed 42

# Run tests
uv run pytest tests/
```

## Architecture

**Pipeline flow**: Parquet (DuckDB SQL filter) → feature extraction → VAE training → latent embedding → UMAP 2D → HDBSCAN clustering → SigProfiler signature assignment.

**Core package** (`uv_vae/`):
- `model.py` — `TabularVAE` (nn.Module) with `VAEConfig`; categorical embeddings + numeric features, encoder/decoder MLP
- `training.py` — training loop
- `train_cli.py` — training CLI entry point
- `data.py` — DuckDB-backed data loading from parquet
- `preprocess.py` — `transform_frame()` for numeric standardization and categorical encoding
- `features.py` — feature spec handling (`ml_features.json`)
- `inference.py` — `LatentInference.from_checkpoint()` for encoding data through a trained model
- `evaluation.py` — Procrustes, CKA, trustworthiness, latent collapse diagnostics
- `pipeline_vae.py` — end-to-end pipeline orchestration
- `cli.py` — argparse CLI (`evaluate-subsamples` command)

**Configuration**: Pipeline configured via environment variables (see README.md for full list). Key ones: `PARQUET_PATH`, `TRAIN_SAMPLE_ROWS`, `TRAIN_EPOCHS`, `TRAIN_SEED`, `TRAIN_LATENT_DIM` (default 16), `TRAIN_HIDDEN_DIMS` (default 256,128), `TRAIN_KL_WEIGHT` (default 0.05).

## HPC environment

- GPU node: `miletus.sabanciuniv.edu` — RTX PRO 5000 Blackwell (48 GB), no SLURM, tmux runners
- Env on miletus: **micromamba `uv_vae`** (override with `MAMBA_ENV`)
- Older CPU cluster: `tosun.sabanciuniv.edu`, user `patrickgao765`; conda env `patrickg`
- SLURM: account=adelab, partition=genomics, qos=adelab
- Parquet files: `/cta/users/patrickgao765/parquet_files/`
- Test set: `/cta/users/patrickgao765/uv_vae/test_set.parquet`
- Always set `export TQDM_DISABLE=1` in SLURM scripts (non-interactive)
- Always set `export UV_VAE_ROOT="$HOME/uv_vae"` before running sweep scripts
- Defer `import matplotlib.pyplot` inside visualization functions (matplotlib can crash on HPC if style files are missing)

## VAE stability testing

Located in `VAE_Stability_Testing/` (sibling directory to repo, at `C:\Users\Owner\Documents\PURE Files\VAE_Stability_Testing\`).

**Key scripts** in `VAE_Stability_Testing/scripts/`:
- `vae_subsample_sweep.py` — main sweep runner; `--data-seed` controls DuckDB row sampling separately from `--seed` (training seed)
- `seed_sweep_vs_52M.py` — computes Procrustes/CKA for seed sweep checkpoints against the 52M full-data reference
- `vs_52M_reference.py` — general-purpose comparison of any checkpoints against 52M reference
- `compare_sweeps.py` / `compare_sweeps_vs_52M.py` — cross-sweep comparison scripts

**Key findings** (from sweep analysis):
- Stability threshold at ~5M rows (seed CV% drops to 0.19%)
- Optimal training: 10-20 epochs (more epochs hurt trustworthiness)
- Latent geometry does NOT converge monotonically with more data — zigzag pattern across all sweep sources
- Only 5 data seeds tested; more seeds at key row counts would strengthen confidence
