#!/usr/bin/env bash
# Fit and SAVE the final HDBSCAN model for the selected cell, once per backend.
#
# The sweep that chose fit1000000_mcs2500_ms15_eom ran before hdbscan_param_sweep.py saved
# hdbscan_model.pkl / fit_indices.npy, so the selected model exists only as metrics and
# labels -- there is no object to hand to per_parquet_inference. This refits it and writes
# the full artefact set for BOTH implementations, into separate directories that name the
# backend, because the two do not agree (measured: all 24 cells of the sweep grid differ,
# cuML from 7.3% below to 22.6% above the CPU cluster count) and a directory that does not
# say which one it holds is how a cuML run ended up published as a CPU result.
#
#   bash umap_hdbscan_sweep/tmux_final_models.sh                    # selected cell, both backends
#   BACKENDS=cuml bash umap_hdbscan_sweep/tmux_final_models.sh      # one backend only
#   FULL_GRID=1 bash umap_hdbscan_sweep/tmux_final_models.sh        # all 24 cells, both backends
#   DRY_RUN=1 bash umap_hdbscan_sweep/tmux_final_models.sh          # grid + projected cost only
#
# Run under tmux -- the cuML pass alone is ~20 min for one cell with scoring, and the CPU
# pass is slower per cell at 5M fit rows:
#   tmux new-session -d -s final_models 'bash umap_hdbscan_sweep/tmux_final_models.sh'
set -euo pipefail
export TQDM_DISABLE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
SCRIPT="$SCRIPT_DIR/hdbscan_param_sweep.py"

# cluster_quality.py and the k-DBCV package sit beside the sweep script, not on the default
# path: python puts only the script's OWN directory on sys.path, and this repo has kept a
# second copy under hdbscan/ since the results reorganisation. Both are added because which
# one a given checkout has is not knowable from here -- and when neither is found the DBCV /
# persistence / connectivity block is skipped, which is exactly the geometry this run exists
# to produce.
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/hdbscan:${PYTHONPATH:-}"

if command -v micromamba &>/dev/null; then
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate "${MAMBA_ENV:-uv_vae}"
elif command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-patrickg}"
fi

COORDS="${COORDS:-$HOME/pure-internship/umap_hdbscan_sweep/hdbscan/results/hdbscan_scaling/coords.npy}"
CONTEXT="${CONTEXT:-$HOME/pure-internship/uv_vae/runs/train_multi_20260802T192756Z/stage1_embed/context.parquet}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/pure-internship/umap_hdbscan_sweep/hdbscan/results/final_models}"

# The selected cell. Overridable, but these five values ARE the published configuration --
# changing one makes the output a different model than pipeline_parameters.md section 3
# describes.
FIT_SIZES="${FIT_SIZES:-1000000}"
MIN_CLUSTER_SIZES="${MIN_CLUSTER_SIZES:-2500}"
MIN_SAMPLES="${MIN_SAMPLES:-15}"
METHODS="${METHODS:-eom}"
EPSILONS="${EPSILONS:-0.0}"

if [ "${FULL_GRID:-0}" = "1" ]; then
    FIT_SIZES="${FIT_SIZES_FULL:-500000,1000000,5000000}"
    MIN_CLUSTER_SIZES="${MIN_CLUSTER_SIZES_FULL:-250,500,1000,2500}"
    MIN_SAMPLES="${MIN_SAMPLES_FULL:-5,15}"
fi

BACKENDS="${BACKENDS:-cuml,cpu}"
SEED="${SEED:-42}"
GPU_TOTAL_GB="${GPU_TOTAL_GB:-40}"
THREADS="${THREADS:-16}"
SIGPROFILER_CPU="${SIGPROFILER_CPU:-16}"
DBCV_PER_CLUSTER="${DBCV_PER_CLUSTER:-400}"
# Pinned per backend below rather than left on 'auto'. k-DBCV and hdbscan's validity_index
# disagree by more than the backends do, so letting each pass silently pick a different
# scorer would make the cuML-vs-CPU DBCV column meaningless.
DBCV_BACKEND="${DBCV_BACKEND:-hdbscan}"

if [ ! -f "$COORDS" ]; then
    echo "ERROR: coords not found: $COORDS" >&2
    exit 1
fi

EXTRA=()
[ "${DRY_RUN:-0}" = "1" ] && EXTRA+=(--dry-run)
# Default is a fresh fit: --resume defaults to True inside the script, which silently skips
# any cell that already has a metrics.json and would turn a re-run into a no-op.
[ "${RESUME:-0}" = "1" ] || EXTRA+=(--no-resume)

if [ -f "$CONTEXT" ]; then
    EXTRA+=(--context "$CONTEXT" --sigprofiler-cpu "$SIGPROFILER_CPU")
else
    echo "WARNING: no context.parquet at $CONTEXT -- models will be saved but NOT scored" >&2
    echo "         (no SigProfiler uv_only assignment for either backend)" >&2
fi

echo "===== final HDBSCAN models  $(date -u) ====="
echo "  coords        : $COORDS"
echo "  output root   : $OUTPUT_ROOT"
echo "  backends      : $BACKENDS"
echo "  cells         : fit=$FIT_SIZES  mcs=$MIN_CLUSTER_SIZES  ms=$MIN_SAMPLES"
echo "                  method=$METHODS  eps=$EPSILONS"
echo "  dbcv          : $DBCV_BACKEND backend, $DBCV_PER_CLUSTER pts/cluster"
echo
nvidia-smi --query-gpu=memory.used,memory.total --format=csv || true
echo

sed -i 's/\r$//' "$SCRIPT" 2>/dev/null || true

for BACKEND in ${BACKENDS//,/ }; do
    OUTPUT_DIR="$OUTPUT_ROOT/$BACKEND"
    mkdir -p "$OUTPUT_DIR"
    LOG="$OUTPUT_DIR/final_models_$(date -u +%Y%m%dT%H%M%SZ).log"

    echo "───────────────────────────────────────────────────────────────"
    echo "[$(date +%H:%M:%S)] backend=$BACKEND  ->  $OUTPUT_DIR"
    echo "───────────────────────────────────────────────────────────────"

    python "$SCRIPT" \
        --coords "$COORDS" \
        --output-dir "$OUTPUT_DIR" \
        --fit-sizes "$FIT_SIZES" \
        --min-cluster-sizes "$MIN_CLUSTER_SIZES" \
        --min-samples "$MIN_SAMPLES" \
        --methods "$METHODS" \
        --epsilons "$EPSILONS" \
        --cluster-backend "$BACKEND" \
        --dbcv-backend "$DBCV_BACKEND" \
        --dbcv-per-cluster "$DBCV_PER_CLUSTER" \
        --gpu-budget-gb "$GPU_TOTAL_GB" \
        --threads "$THREADS" \
        --seed "$SEED" \
        "${EXTRA[@]}" 2>&1 | tee -a "$LOG"

    echo "[$(date +%H:%M:%S)] backend=$BACKEND done"
    echo
done

echo "===== DONE  $(date -u) ====="
echo
echo "Artefacts per cell:"
for BACKEND in ${BACKENDS//,/ }; do
    for CELL_DIR in "$OUTPUT_ROOT/$BACKEND"/cells/*/; do
        [ -d "$CELL_DIR" ] || continue
        echo "  $BACKEND  $(basename "$CELL_DIR")"
        ls -1 "$CELL_DIR" | sed 's/^/      /'
    done
done
