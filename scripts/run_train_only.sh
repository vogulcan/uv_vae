#!/bin/bash
#SBATCH --job-name=vae-train-earlystop
#SBATCH --account=adelab
#SBATCH --partition=genomics
#SBATCH --qos=adelab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/cta/users/patrickgao765/uv_vae/train_earlystop_%j.log
#SBATCH --error=/cta/users/patrickgao765/uv_vae/train_earlystop_%j.err
#
# Uncomment if the genomics partition exposes GPUs to the adelab account.
##SBATCH --gres=gpu:1
#
# Training-only variant of run_full_pipeline.sh: combine (or reuse) the merged parquet and
# train with early stopping, then stop. Use this to tune PATIENCE / MIN_DELTA without
# paying for the clustering stage every time.
#
# Reuse an already-merged parquet to skip the combine step entirely:
#   COMBINED=/path/to/combined.featuremap.parquet sbatch scripts/run_train_only.sh
#
#   sed -i 's/\r$//' scripts/run_train_only.sh   # if edited on Windows
#   sbatch scripts/run_train_only.sh

set -euo pipefail

###############################################################################
# EDIT THESE -- only needed when COMBINED is not supplied
###############################################################################
SAMPLE_1="/cta/users/patrickgao765/parquet_files/CHANGE_ME_sample_1.featuremap.parquet"
SAMPLE_2="/cta/users/patrickgao765/parquet_files/CHANGE_ME_sample_2.featuremap.parquet"
###############################################################################

UV_VAE_DIR="${UV_VAE_DIR:-$HOME/uv_vae}"
RUN_ROOT="${RUN_ROOT:-/cta/users/patrickgao765/uv_vae/train_earlystop_${SLURM_JOB_ID:-manual}}"
ROW_FILTER="${ROW_FILTER:-st = 'MIXED' AND et = 'MIXED' AND FILT = 1}"
THREADS="${SLURM_CPUS_PER_TASK:-32}"
SEED="${SEED:-42}"
DUCKDB_MEM="8GB"

EPOCH_CEILING="${EPOCH_CEILING:-100}"
PATIENCE="${PATIENCE:-2}"
MIN_DELTA="${MIN_DELTA:-0.001}"
AU_THRESHOLD="${AU_THRESHOLD:-0.01}"
INPUT_DROPOUT="${INPUT_DROPOUT:-0.1}"
HIDDEN_DROPOUT="${HIDDEN_DROPOUT:-0.4}"
TEST_PARQUET="${TEST_PARQUET:-}"
CONVERGENCE_ROWS="${CONVERGENCE_ROWS:-5000}"

COMBINED="${COMBINED:-$RUN_ROOT/combined.featuremap.parquet}"

mkdir -p "$RUN_ROOT"

eval "$(conda shell.bash hook)"
conda activate patrickg
cd "$UV_VAE_DIR"

export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

fmt_seconds() {
    local s=$1
    printf '%dh %02dm %02ds' $((s / 3600)) $(((s % 3600) / 60)) $((s % 60))
}

combine_seconds=0
train_seconds=0

print_summary() {
    echo ""
    echo "=========================== TIMING SUMMARY ==========================="
    printf '  %-24s %8ss   %s\n' "combine" "$combine_seconds" "$(fmt_seconds "$combine_seconds")"
    printf '  %-24s %8ss   %s\n' "train"   "$train_seconds"   "$(fmt_seconds "$train_seconds")"
    printf '  %-24s %8ss   %s\n' "TOTAL"   "$SECONDS"         "$(fmt_seconds "$SECONDS")"
    if [ -f "$RUN_ROOT/train_result.json" ]; then
        python - "$RUN_ROOT/train_result.json" <<'PY' || true
import json, sys
early = (json.load(open(sys.argv[1])).get("early_stopping") or {})
print("  epochs_run      : {} of {}".format(
    early.get("epochs_run", "?"), early.get("epochs_requested", "?")))
print("  stopped_early   : {}".format(early.get("stopped_early", "?")))
print("  best_epoch      : {}".format(early.get("best_epoch", "?")))
print("  final AU count  : {}".format(early.get("final_active_units", "?")))
print("  stop_reason     : {}".format(early.get("stop_reason")))
PY
    fi
    echo "  Artifacts: $RUN_ROOT"
    echo "======================================================================"
}
trap print_summary EXIT

if [ -f "$COMBINED" ]; then
    echo "[$(date '+%F %T')] Reusing existing merged parquet: $COMBINED"
else
    echo "[$(date '+%F %T')] ===== BEGIN: combine parquets ====="
    stage_start=$SECONDS
    python scripts/combine_parquets.py \
        --inputs "$SAMPLE_1" "$SAMPLE_2" \
        --output "$COMBINED" \
        --threads "$THREADS" \
        --memory-limit "$DUCKDB_MEM" \
        --temp-directory "$RUN_ROOT/duckdb_tmp"
    combine_seconds=$((SECONDS - stage_start))
    echo "[$(date '+%F %T')] ===== END: combine ($(fmt_seconds "$combine_seconds")) ====="
fi

echo "[$(date '+%F %T')] ===== BEGIN: train VAE (early stopping) ====="
stage_start=$SECONDS
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
train_seconds=$((SECONDS - stage_start))
echo "[$(date '+%F %T')] ===== END: train ($(fmt_seconds "$train_seconds")) ====="

echo ""
echo "Per-epoch ELBO / per-dim KL / active units: <run dir>/diagnostics_report.json"
