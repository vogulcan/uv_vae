#!/usr/bin/env bash
# Per-parquet inference — miletus runner
# For each of 95 parquet files: VAE encode -> parametric UMAP -> HDBSCAN -> SigProfiler (uv_only) -> 4 plots
# SigProfiler + plots run in parallel across --n-workers processes.
#
# Usage:
#   tmux new-session -d -s per_parquet 'bash umap_hdbscan_sweep/tmux_per_parquet_inference.sh'
#   CLUSTER_BACKEND=cpu tmux ... (same, against the CPU model instead)
set -euo pipefail
export TQDM_DISABLE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/hdbscan:${PYTHONPATH:-}"

# ── detect cluster and activate environment ─────────────────────────────────────
if command -v micromamba &>/dev/null; then
    eval "$(micromamba shell hook -s bash)"
    micromamba activate "${MAMBA_ENV:-uv_vae}"
elif command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-patrickg}"
fi

# ── default paths (miletus) ─────────────────────────────────────────────────────
PARQUET_GLOB="${PARQUET_GLOB:-/data/lab/ppmseq_parquets/*.parquet}"
CHECKPOINT="${CHECKPOINT:-$HOME/pure-internship/uv_vae/runs/train_multi_20260802T192756Z/training/run_20260802T192814Z/model.pt}"
FEATURE_SPEC="${FEATURE_SPEC:-$HOME/pure-internship/uv_vae/ml_features.json}"
UMAP_MODEL="${UMAP_MODEL:-$HOME/pure-internship/umap_hdbscan_sweep/umap/results/final_models/13_BEST_25M_nn15_md0.1_umap.pt}"
COORDS="${COORDS:-$HOME/pure-internship/umap_hdbscan_sweep/hdbscan/results/hdbscan_scaling/coords.npy}"
CONTEXT="${CONTEXT:-$HOME/pure-internship/uv_vae/runs/train_multi_20260802T192756Z/stage1_embed/context.parquet}"

# Reuse the cohort HDBSCAN rather than refitting here, so per-sample labels are directly
# comparable to the cohort run and to each other. Which model that is has changed:
#
#   was  low_noise_hdbscan/hdbscan_model.pkl  -- mcs=2500 ms=1 eps=0.05, 170 clusters
#   now  final_models/$CLUSTER_BACKEND/.../fit1000000_mcs2500_ms15_eom
#
# The low_noise model is a DIFFERENT CELL from the one the sweep selected, so the 95-sample
# run built on it labelled every read against a clustering that no reported metric describes.
# CLUSTER_BACKEND picks which implementation's model to label with; it must match whichever
# backend the reported cluster count came from (cuML = 175 clusters, cpu = 181), because the
# two produce different partitions and cluster ids are not comparable across them.
CLUSTER_BACKEND="${CLUSTER_BACKEND:-cuml}"
CELL="${CELL:-fit1000000_mcs2500_ms15_eom}"
MODEL_DIR="${MODEL_DIR:-$HOME/pure-internship/umap_hdbscan_sweep/hdbscan/results/final_models/$CLUSTER_BACKEND/cells/$CELL}"

OUTPUT_DIR="${OUTPUT_DIR:-$HOME/pure-internship/umap_hdbscan_sweep/per_parquet_inference_$CLUSTER_BACKEND}"

# Set HDBSCAN_MODEL="" to make this script fit its own CPU model with MCS/MS/EPSILON below.
HDBSCAN_MODEL="${HDBSCAN_MODEL:-$MODEL_DIR/hdbscan_model.pkl}"
FIT_INDICES="${FIT_INDICES:-$MODEL_DIR/fit_indices.npy}"

MCS="${MCS:-2500}"
MS="${MS:-15}"         # selected cell; ignored when HDBSCAN_MODEL is set
EPSILON="${EPSILON:-0.0}"
FIT_ROWS="${FIT_ROWS:-1000000}"
SEED="${SEED:-42}"
GENOME_BUILD="${GENOME_BUILD:-GRCh38}"
COSMIC_VERSION="${COSMIC_VERSION:-3.5}"
# 4 workers × 4 CPUs each = 16 total, matches the Blackwell's CPU allocation
N_WORKERS="${N_WORKERS:-4}"
SIGPROFILER_CPU="${SIGPROFILER_CPU:-4}"
DEVICE="${DEVICE:-auto}"
GPU_BUDGET_GB="${GPU_BUDGET_GB:-44}"              # 48 GB card; RMM takes 0.9 of this
PREDICT_BATCH_ROWS="${PREDICT_BATCH_ROWS:-5000000}"  # ~1.2 GB peak on the GPU

# ── log setup ───────────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"
LOG="$OUTPUT_DIR/run_$(date -u +%Y%m%dT%H%M%SZ).log"

echo "===== per_parquet_inference  $(date -u) =====" | tee -a "$LOG"
echo "  parquet_glob:   $PARQUET_GLOB"               | tee -a "$LOG"
echo "  checkpoint:     $CHECKPOINT"                 | tee -a "$LOG"
echo "  umap_model:     $UMAP_MODEL"                 | tee -a "$LOG"
echo "  output_dir:     $OUTPUT_DIR"                 | tee -a "$LOG"
echo "  cluster_backend:$CLUSTER_BACKEND  cell=$CELL" | tee -a "$LOG"
echo "  mcs=$MCS  ms=$MS  eps=$EPSILON"              | tee -a "$LOG"
echo "  n_workers=$N_WORKERS  sigprofiler_cpu=$SIGPROFILER_CPU"  | tee -a "$LOG"

sed -i 's/\r$//' "$SCRIPT_DIR/per_parquet_inference.py" 2>/dev/null || true

SKIP_DONE_FLAG=""
if [ "${SKIP_DONE:-0}" = "1" ]; then
    SKIP_DONE_FLAG="--skip-done"
fi

MODEL_ARGS=""
if [ -n "$HDBSCAN_MODEL" ]; then
    if [ ! -f "$HDBSCAN_MODEL" ]; then
        echo "ERROR: HDBSCAN_MODEL not found: $HDBSCAN_MODEL" | tee -a "$LOG"
        echo "  Run tmux_final_models.sh first (it fits and saves the selected cell for" | tee -a "$LOG"
        echo "  both backends), or set HDBSCAN_MODEL=\"\" to fit a CPU model here." | tee -a "$LOG"
        exit 1
    fi
    MODEL_ARGS="--hdbscan-model $HDBSCAN_MODEL"
    if [ -f "$FIT_INDICES" ]; then
        MODEL_ARGS="$MODEL_ARGS --fit-indices $FIT_INDICES"
    else
        # Not fatal -- the python re-derives them from (seed, fit_rows) and aborts if the
        # length disagrees with the model's own labels_. Worth saying out loud, because the
        # derivation only stays correct while this script and the sweep agree on both.
        echo "  WARNING: no fit_indices.npy beside the model; indices will be re-derived" | tee -a "$LOG"
        echo "           from seed=$SEED fit_rows=$FIT_ROWS" | tee -a "$LOG"
    fi
    echo "  hdbscan_model:  $HDBSCAN_MODEL" | tee -a "$LOG"
fi

python "$SCRIPT_DIR/per_parquet_inference.py" \
    --parquet-glob    "$PARQUET_GLOB" \
    --checkpoint      "$CHECKPOINT" \
    --feature-spec    "$FEATURE_SPEC" \
    --umap-model      "$UMAP_MODEL" \
    --coords          "$COORDS" \
    --context         "$CONTEXT" \
    --output-dir      "$OUTPUT_DIR" \
    --mcs             "$MCS" \
    --ms              "$MS" \
    --epsilon         "$EPSILON" \
    --fit-rows        "$FIT_ROWS" \
    --seed            "$SEED" \
    --genome-build    "$GENOME_BUILD" \
    --cosmic-version  "$COSMIC_VERSION" \
    --n-workers       "$N_WORKERS" \
    --sigprofiler-cpu "$SIGPROFILER_CPU" \
    --device          "$DEVICE" \
    --gpu-budget-gb   "$GPU_BUDGET_GB" \
    --predict-batch-rows "$PREDICT_BATCH_ROWS" \
    $MODEL_ARGS \
    $SKIP_DONE_FLAG \
    2>&1 | tee -a "$LOG"

echo "===== DONE  $(date -u) =====" | tee -a "$LOG"
