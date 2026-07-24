#!/bin/bash
#SBATCH --job-name=vae-kl-sweep
#SBATCH --account=adelab
#SBATCH --partition=genomics
#SBATCH --qos=adelab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/cta/users/patrickgao765/uv_vae/kl_sweep_%j.log
#SBATCH --error=/cta/users/patrickgao765/uv_vae/kl_sweep_%j.err
#
# Uncomment if the genomics partition exposes GPUs to the adelab account.
# cuML-accelerated UMAP/HDBSCAN in the clustering stage needs a GPU.
##SBATCH --gres=gpu:1
#
# For each KL weight in KL_WEIGHTS:
#   1. Train VAE with early stopping (streaming, all filtered rows)
#   2. Run UMAP → HDBSCAN → UV signatures (optional: set CLUSTER=0 to skip)
# Then compare all checkpoints on a fixed reference sample with compare_kl_sweep.py.
#
# Defaults cover the range from minimal KL (reconstruction-dominated) to
# standard VAE (β=1). The current pipeline baseline β=0.05 is included.
#
#   sed -i 's/\r$//' scripts/run_kl_sweep.sh    # if edited on Windows
#   sbatch scripts/run_kl_sweep.sh
#
# Narrow to a range of interest:
#   KL_WEIGHTS="0.01,0.05,0.1" CLUSTER=0 sbatch scripts/run_kl_sweep.sh

set -euo pipefail

###############################################################################
# EDIT THIS — source parquet (glob across all 95 samples, or a single file
# for a fast test trial)
###############################################################################
PARQUET="${PARQUET:-/cta/users/patrickgao765/parquet_files/wt0-12-ppm0050.featuremap.parquet}"
###############################################################################

UV_VAE_DIR="${UV_VAE_DIR:-$HOME/uv_vae}"
RUN_ROOT="${RUN_ROOT:-/cta/users/patrickgao765/uv_vae/kl_sweep_${SLURM_JOB_ID:-manual}}"
ROW_FILTER="${ROW_FILTER:-st = 'MIXED' AND et = 'MIXED' AND FILT = 1}"
THREADS="${SLURM_CPUS_PER_TASK:-32}"
SEED="${SEED:-42}"

# KL weights to sweep — log-spaced from reconstruction-dominated to full-VAE.
# Current pipeline baseline is 0.05. β>1 is the disentanglement (β-VAE) regime.
KL_WEIGHTS="${KL_WEIGHTS:-0.001,0.005,0.01,0.05,0.1,0.5,1.0}"

# VAE training hyperparameters — match the main pipeline.
EPOCH_CEILING="${EPOCH_CEILING:-100}"
PATIENCE="${PATIENCE:-8}"
MIN_DELTA="${MIN_DELTA:-0.001}"
AU_THRESHOLD="${AU_THRESHOLD:-0.01}"
INPUT_DROPOUT="${INPUT_DROPOUT:-0.1}"
HIDDEN_DROPOUT="${HIDDEN_DROPOUT:-0.4}"
BATCH_SIZE="${BATCH_SIZE:-32768}"
LATENT_DIM="${LATENT_DIM:-16}"
HIDDEN_DIMS="${HIDDEN_DIMS:-256,128}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.8}"

# Set CLUSTER=1 to run the full UMAP → HDBSCAN → SigProfiler pipeline after
# each VAE train. CLUSTER=0 trains only — much faster, still gives you latent
# fidelity comparison in the final compare_kl_sweep.py step.
CLUSTER="${CLUSTER:-0}"

# Clustering pipeline settings (only used when CLUSTER=1).
DUCKDB_MEM="${DUCKDB_MEM:-8GB}"
UMAP_FIT_ROWS="${UMAP_FIT_ROWS:-100000}"
HDBSCAN_MIN_CLUSTER_SIZE="${HDBSCAN_MIN_CLUSTER_SIZE:-250}"
HDBSCAN_MIN_SAMPLES="${HDBSCAN_MIN_SAMPLES:-25}"

# compare_kl_sweep.py settings.
REFERENCE_KL="${REFERENCE_KL:-0.05}"   # model all others are compared against
REFERENCE_ROWS="${REFERENCE_ROWS:-4000}"

mkdir -p "$RUN_ROOT"

eval "$(conda shell.bash hook)"
conda activate patrickg
cd "$UV_VAE_DIR"

export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

###############################################################################
# Timing helpers (same convention as run_train_only.sh / run_full_pipeline.sh)
###############################################################################
declare -a STAGE_NAMES=()
declare -a STAGE_SECONDS=()
STAGE_START=0

fmt_seconds() {
    local s=$1
    printf '%dh %02dm %02ds' $((s / 3600)) $(((s % 3600) / 60)) $((s % 60))
}

begin_stage() {
    echo ""
    echo "[$(date '+%F %T')] ===== BEGIN: $1 ====="
    STAGE_START=$SECONDS
}

end_stage() {
    local duration=$((SECONDS - STAGE_START))
    STAGE_NAMES+=("$1")
    STAGE_SECONDS+=("$duration")
    echo "[$(date '+%F %T')] ===== END:   $1  ($(fmt_seconds "$duration")) ====="
}

print_summary() {
    echo ""
    echo "=========================== TIMING SUMMARY ==========================="
    if [ ${#STAGE_NAMES[@]} -gt 0 ]; then
        local i
        for i in "${!STAGE_NAMES[@]}"; do
            printf '  %-44s %8ss   %s\n' \
                "${STAGE_NAMES[$i]}" "${STAGE_SECONDS[$i]}" "$(fmt_seconds "${STAGE_SECONDS[$i]}")"
        done
    fi
    printf '  %-44s %8ss   %s\n' "TOTAL (job wall clock)" "$SECONDS" "$(fmt_seconds "$SECONDS")"
    echo ""
    if [ -f "$RUN_ROOT/kl_sweep_comparison.json" ]; then
        python - "$RUN_ROOT/kl_sweep_comparison.json" <<'PY' || true
import json, sys
data = json.load(open(sys.argv[1]))
ref = data.get("reference_kl", "?")
print(f"  Reference model  : β={ref}")
print(f"  {'β':>8}  {'epochs':>6}  {'early?':>6}  {'AU':>7}  {'procrustes':>10}  {'trust':>7}")
print(f"  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*10}  {'-'*7}")
for r in data.get("results", []):
    print(
        f"  {r['kl_weight']:>8g}"
        f"  {r['epochs_run']:>6d}"
        f"  {'yes' if r['stopped_early'] else 'no':>6}"
        f"  {r['final_active_units']:>3d}/{r['latent_dim']:<3d}"
        f"  {r['procrustes_distance']:>10.4f}"
        f"  {r['trustworthiness']:>7.4f}"
    )
PY
    fi
    echo "  Comparison JSON : $RUN_ROOT/kl_sweep_comparison.json"
    echo "  Plot            : $RUN_ROOT/kl_sweep_comparison.png"
    echo "  Artifacts       : $RUN_ROOT"
    echo "  sacct -j ${SLURM_JOB_ID:-<jobid>} --format=JobID,State,Elapsed,MaxRSS"
    echo "======================================================================"
}
trap print_summary EXIT

echo "Run root   : $RUN_ROOT"
echo "uv_vae dir : $UV_VAE_DIR"
echo "Parquet    : $PARQUET"
echo "Row filter : $ROW_FILTER"
echo "KL weights : $KL_WEIGHTS"
echo "Cluster    : $CLUSTER"
echo "Threads    : $THREADS"

###############################################################################
# Main sweep loop
###############################################################################
IFS=',' read -ra KL_ARRAY <<< "$KL_WEIGHTS"

for kl in "${KL_ARRAY[@]}"; do
    kl="${kl// /}"   # strip any spaces
    kl_dir="$RUN_ROOT/kl_${kl}"
    mkdir -p "$kl_dir"

    # --- Train ---------------------------------------------------------------
    begin_stage "train β=${kl}"
    python scripts/train_with_early_stopping.py \
        --parquet-path "$PARQUET" \
        --feature-spec-path "$UV_VAE_DIR/ml_features.json" \
        --output-dir "$kl_dir/training" \
        --row-filter "$ROW_FILTER" \
        --streaming \
        --epochs "$EPOCH_CEILING" \
        --patience "$PATIENCE" \
        --min-delta "$MIN_DELTA" \
        --active-unit-threshold "$AU_THRESHOLD" \
        --input-dropout "$INPUT_DROPOUT" \
        --hidden-dropout "$HIDDEN_DROPOUT" \
        --batch-size "$BATCH_SIZE" \
        --latent-dim "$LATENT_DIM" \
        --hidden-dims "$HIDDEN_DIMS" \
        --learning-rate "$LEARNING_RATE" \
        --kl-weight "$kl" \
        --train-fraction "$TRAIN_FRACTION" \
        --seed "$SEED" \
        --threads "$THREADS" \
        | tee "$kl_dir/train_result.json"
    end_stage "train β=${kl}"

    # --- Cluster (optional) --------------------------------------------------
    if [ "$CLUSTER" = "1" ]; then
        CHECKPOINT="$(python -c "
import json, sys
print(json.load(open(sys.argv[1]))['checkpoint_path'])
" "$kl_dir/train_result.json")"

        begin_stage "cluster β=${kl}"
        python scripts/run_variant_cluster_pipeline.py \
            --checkpoint-path "$CHECKPOINT" \
            --parquet-path "$PARQUET" \
            --row-filter "$ROW_FILTER" \
            --output-root "$kl_dir/clustering" \
            --use-all \
            --umap-fit-rows "$UMAP_FIT_ROWS" \
            --hdbscan-min-cluster-size "$HDBSCAN_MIN_CLUSTER_SIZE" \
            --hdbscan-min-samples "$HDBSCAN_MIN_SAMPLES" \
            --seed "$SEED" \
            --threads "$THREADS" \
            --sigprofiler-cpu "$THREADS" \
            --duckdb-memory-limit "$DUCKDB_MEM" \
            > "$kl_dir/cluster_result.json"
        end_stage "cluster β=${kl}"
    fi
done

###############################################################################
# Compare all checkpoints
###############################################################################
begin_stage "compare all β values"
python scripts/compare_kl_sweep.py \
    --sweep-dir "$RUN_ROOT" \
    --parquet-path "$PARQUET" \
    --feature-spec-path "$UV_VAE_DIR/ml_features.json" \
    --row-filter "$ROW_FILTER" \
    --reference-kl "$REFERENCE_KL" \
    --reference-rows "$REFERENCE_ROWS" \
    --seed "$SEED" \
    --threads "$THREADS"
end_stage "compare all β values"
