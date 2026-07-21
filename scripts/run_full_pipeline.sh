#!/bin/bash
#SBATCH --job-name=vae-full-pipeline
#SBATCH --account=adelab
#SBATCH --partition=genomics
#SBATCH --qos=adelab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=/cta/users/patrickgao765/uv_vae/full_pipeline_%j.log
#SBATCH --error=/cta/users/patrickgao765/uv_vae/full_pipeline_%j.err
#
# Uncomment the next line if the genomics partition exposes GPUs to the adelab account.
# Training and cuML-accelerated UMAP/HDBSCAN both auto-detect CUDA and fall back to CPU,
# so the job runs either way -- but a CPU-only timing run will over-estimate the cost of
# the eventual 95-sample run if you would normally train on a GPU.
##SBATCH --gres=gpu:1
#
# End-to-end timing run: combine 2 sample parquets -> train the VAE with early stopping
# -> full clustering pipeline (embed -> UMAP -> HDBSCAN -> UV signatures).
#
# The point of this job is to produce a per-stage wall-clock breakdown and a row count so
# the 95-sample run can be extrapolated. Every stage is timed independently.
#
#   sed -i 's/\r$//' scripts/run_full_pipeline.sh   # if edited on Windows
#   sbatch scripts/run_full_pipeline.sh

set -euo pipefail

###############################################################################
# EDIT THESE -- the two sample parquets to merge
###############################################################################
SAMPLE_1="/cta/users/patrickgao765/parquet_files/CHANGE_ME_sample_1.featuremap.parquet"
SAMPLE_2="/cta/users/patrickgao765/parquet_files/CHANGE_ME_sample_2.featuremap.parquet"
###############################################################################

UV_VAE_DIR="${UV_VAE_DIR:-$HOME/uv_vae}"
RUN_ROOT="${RUN_ROOT:-/cta/users/patrickgao765/uv_vae/full_pipeline_${SLURM_JOB_ID:-manual}}"
ROW_FILTER="${ROW_FILTER:-st = 'MIXED' AND et = 'MIXED' AND FILT = 1}"
THREADS="${SLURM_CPUS_PER_TASK:-32}"
SEED="${SEED:-42}"
DUCKDB_MEM="8GB"

# Early stopping: EPOCH_CEILING is a safety cap, not a target. PATIENCE is what should
# actually end training -- it stops once the validation ELBO has stagnated AND the active
# unit count has held steady for this many consecutive epochs.
EPOCH_CEILING="${EPOCH_CEILING:-100}"
PATIENCE="${PATIENCE:-2}"
MIN_DELTA="${MIN_DELTA:-0.001}"
AU_THRESHOLD="${AU_THRESHOLD:-0.01}"
INPUT_DROPOUT="${INPUT_DROPOUT:-0.1}"
HIDDEN_DROPOUT="${HIDDEN_DROPOUT:-0.4}"
TEST_PARQUET="${TEST_PARQUET:-}"
CONVERGENCE_ROWS="${CONVERGENCE_ROWS:-5000}"

COMBINED="$RUN_ROOT/combined.featuremap.parquet"

mkdir -p "$RUN_ROOT"

eval "$(conda shell.bash hook)"
conda activate patrickg
cd "$UV_VAE_DIR"

# Determinism knobs that do not cost throughput. BLAS thread counts are deliberately NOT
# pinned to 1 here: this is a timing run, and pinning them would distort the estimate.
export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

###############################################################################
# Timing helpers
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
            printf '  %-34s %8ss   %s\n' \
                "${STAGE_NAMES[$i]}" "${STAGE_SECONDS[$i]}" "$(fmt_seconds "${STAGE_SECONDS[$i]}")"
        done
    fi
    printf '  %-34s %8ss   %s\n' "TOTAL (job wall clock)" "$SECONDS" "$(fmt_seconds "$SECONDS")"
    echo ""
    if [ -f "$RUN_ROOT/train_result.json" ]; then
        python - "$RUN_ROOT/train_result.json" "$SECONDS" <<'PY' || true
import json, sys
result = json.load(open(sys.argv[1]))
total = int(sys.argv[2])
rows = int(result.get("eligible_rows", 0))
early = result.get("early_stopping", {}) or {}
print("  Rows used (post-filter): {:,}".format(rows))
print("  Epochs run             : {} of {} (stopped_early={})".format(
    early.get("epochs_run", "?"), early.get("epochs_requested", "?"),
    early.get("stopped_early", "?")))
print("  Best epoch             : {}  (val_total_loss={})".format(
    early.get("best_epoch", "?"), early.get("best_val_total_loss", "?")))
print("  Final active units     : {}".format(early.get("final_active_units", "?")))
if rows:
    print("")
    print("  EXTRAPOLATION BASIS")
    print("    {:.1f} s per 1M rows (whole job, this sample count)".format(total / (rows / 1e6)))
    print("    Scale by rows, not by sample count, and re-check epochs_run:")
    print("    early stopping may need a different number of epochs at 95 samples.")
PY
    fi
    echo "  Artifacts: $RUN_ROOT"
    echo "  sacct -j ${SLURM_JOB_ID:-<jobid>} --format=JobID,State,Elapsed,MaxRSS"
    echo "======================================================================"
}
trap print_summary EXIT

echo "Run root   : $RUN_ROOT"
echo "uv_vae dir : $UV_VAE_DIR"
echo "Row filter : $ROW_FILTER"
echo "Threads    : $THREADS"

###############################################################################
# Stage 1 -- combine the two sample parquets into one training input
###############################################################################
begin_stage "1. combine parquets"
python scripts/combine_parquets.py \
    --inputs "$SAMPLE_1" "$SAMPLE_2" \
    --output "$COMBINED" \
    --threads "$THREADS" \
    --memory-limit "$DUCKDB_MEM" \
    --temp-directory "$RUN_ROOT/duckdb_tmp"
end_stage "1. combine parquets"

###############################################################################
# Stage 2 -- train the VAE on the FULL merged pool, ended by early stopping
###############################################################################
begin_stage "2. train VAE (early stopping)"
python scripts/train_with_early_stopping.py \
    --parquet-path "$COMBINED" \
    --feature-spec-path "$UV_VAE_DIR/ml_features.json" \
    --output-dir "$RUN_ROOT/training" \
    --row-filter "$ROW_FILTER" \
    --streaming \
    --epochs "$EPOCH_CEILING" \
    --patience "$PATIENCE" \
    --min-delta "$MIN_DELTA" \
    --active-unit-threshold "$AU_THRESHOLD" \
    --input-dropout "$INPUT_DROPOUT" \
    --hidden-dropout "$HIDDEN_DROPOUT" \
    --batch-size 32768 \
    --latent-dim 16 \
    --hidden-dims 256,128 \
    --learning-rate 1e-3 \
    --kl-weight 0.05 \
    --train-fraction 0.8 \
    --seed "$SEED" \
    --threads "$THREADS" \
    ${TEST_PARQUET:+--test-parquet-path "$TEST_PARQUET" --convergence-rows "$CONVERGENCE_ROWS"} \
    | tee "$RUN_ROOT/train_result.json"
end_stage "2. train VAE (early stopping)"

CHECKPOINT="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['checkpoint_path'])" "$RUN_ROOT/train_result.json")"
echo "Checkpoint: $CHECKPOINT"

###############################################################################
# Stage 3 -- full clustering pipeline on the same merged parquet
###############################################################################
begin_stage "3. cluster pipeline (embed/UMAP/HDBSCAN/SigProfiler)"
python scripts/run_variant_cluster_pipeline.py \
    --checkpoint-path "$CHECKPOINT" \
    --parquet-path "$COMBINED" \
    --row-filter "$ROW_FILTER" \
    --output-root "$RUN_ROOT/clustering" \
    --use-all \
    --seed "$SEED" \
    --threads "$THREADS" \
    --sigprofiler-cpu "$THREADS" \
    --duckdb-memory-limit "$DUCKDB_MEM" \
    > "$RUN_ROOT/clustering_summary.json"
end_stage "3. cluster pipeline (embed/UMAP/HDBSCAN/SigProfiler)"

echo ""
echo "Pipeline finished. Key outputs:"
echo "  training run dir     : $RUN_ROOT/training"
echo "  diagnostics per epoch: <run dir>/diagnostics_report.json"
echo "  clustering outputs   : $RUN_ROOT/clustering"
echo "  cluster summary      : $RUN_ROOT/clustering_summary.json"
