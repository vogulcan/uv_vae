#!/bin/bash
#SBATCH --job-name=vae-feature-select
#SBATCH --account=adelab
#SBATCH --partition=genomics
#SBATCH --qos=adelab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/cta/users/patrickgao765/uv_vae/feature_select_%j.log
#SBATCH --error=/cta/users/patrickgao765/uv_vae/feature_select_%j.err
#
# Uncomment if the genomics partition exposes GPUs to the adelab account.
# Feature selection retrains ~10 small VAEs, so a GPU shortens it a lot.
##SBATCH --gres=gpu:1
#
# Rank the ml_features.json columns by five feature-selection methods, retrain a
# VAE on the top subset of each (with early stopping), and measure how far each
# reduced latent space is from the all-feature baseline using the project's own
# metrics (Procrustes, CKA, trustworthiness, jaccard-kNN, active units) plus
# optional HDBSCAN cluster agreement (ARI/NMI). See scripts/feature_selection_test.py.
#
#   sed -i 's/\r$//' scripts/run_feature_selection.sh   # if edited on Windows
#   sbatch scripts/run_feature_selection.sh
#
# Override any knob from the environment, e.g.:
#   PARQUET=/path/other.parquet SAMPLE_ROWS=500000 sbatch scripts/run_feature_selection.sh

set -euo pipefail

UV_VAE_DIR="${UV_VAE_DIR:-$HOME/uv_vae}"
PARQUET="${PARQUET:-/cta/users/patrickgao765/parquet_files/wt0-12-ppm0050.featuremap.parquet}"
RUN_ROOT="${RUN_ROOT:-/cta/users/patrickgao765/uv_vae/feature_select_${SLURM_JOB_ID:-manual}}"
ROW_FILTER="${ROW_FILTER:-st = 'MIXED' AND et = 'MIXED' AND FILT = 1}"
THREADS="${SLURM_CPUS_PER_TASK:-32}"
SEED="${SEED:-42}"

# Training rows per VAE. Small for fast iteration; the 2.5M-5M sweet spot is the
# lowest-variance-per-compute point, NOT full-data quality. For the final call use
# ALL_ROWS=1 to train on every filtered row (best absolute fidelity).
SAMPLE_ROWS="${SAMPLE_ROWS:-200000}"      # ignored when ALL_ROWS=1
ALL_ROWS="${ALL_ROWS:-0}"                 # 1 => use every filtered row, no subsampling
REFERENCE_ROWS="${REFERENCE_ROWS:-4000}"  # fixed rows every model encodes for the metrics

# The five requested selection methods (all scikit-learn except permutation,
# which ranks against the trained VAE's own latent space).
METHODS="${METHODS:-permutation,pca,ica,nmf,feature_clustering}"
PRIMARY_METHOD="${PRIMARY_METHOD:-permutation}"   # method swept across KEEP_SIZES for the elbow curve
KEEP_SIZES="${KEEP_SIZES:-8,12,16,20,24}"         # subset sizes for the primary elbow sweep
EVAL_KEEP="${EVAL_KEEP:-16}"                       # shared size where every method is compared head-to-head

# VAE hyperparameters + early stopping (match the training pipeline).
EPOCH_CEILING="${EPOCH_CEILING:-40}"
PATIENCE="${PATIENCE:-6}"
MIN_DELTA="${MIN_DELTA:-0.001}"
AU_THRESHOLD="${AU_THRESHOLD:-0.01}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
LATENT_DIM="${LATENT_DIM:-16}"
HIDDEN_DIMS="${HIDDEN_DIMS:-256,128}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
KL_WEIGHT="${KL_WEIGHT:-0.05}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.9}"

# Structural-method (PCA / ICA / NMF / feature clustering) knobs.
PCA_VARIANCE="${PCA_VARIANCE:-0.9}"
N_COMPONENTS="${N_COMPONENTS:-12}"
STRUCTURAL_ROWS="${STRUCTURAL_ROWS:-20000}"

# Downstream clustering agreement: none | hdbscan | umap-hdbscan (UMAP is optional).
CLUSTERING="${CLUSTERING:-hdbscan}"

# Optionally pin biologically essential columns so no method can drop them:
#   FORCE_KEEP="REF,ALT,X_PREV1,X_NEXT1"
FORCE_KEEP="${FORCE_KEEP:-}"

mkdir -p "$RUN_ROOT"

eval "$(conda shell.bash hook)"
conda activate patrickg
cd "$UV_VAE_DIR"

export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

###############################################################################
# Timing + summary helpers (same convention as run_train_only.sh)
###############################################################################
fmt_seconds() {
    local s=$1
    printf '%dh %02dm %02ds' $((s / 3600)) $(((s % 3600) / 60)) $((s % 60))
}

select_seconds=0

print_summary() {
    echo ""
    echo "=========================== TIMING SUMMARY ==========================="
    printf '  %-24s %8ss   %s\n' "feature selection" "$select_seconds" "$(fmt_seconds "$select_seconds")"
    printf '  %-24s %8ss   %s\n' "TOTAL"             "$SECONDS"        "$(fmt_seconds "$SECONDS")"
    if [ -f "$RUN_ROOT/feature_selection_results.json" ]; then
        python - "$RUN_ROOT/feature_selection_results.json" <<'PY' || true
import json, sys
data = json.load(open(sys.argv[1]))
base = data.get("baseline") or {}
rec = data.get("recommendation")
print("  baseline features : {} (input_dim {})".format(
    base.get("n_features", "?"), base.get("input_dim", "?")))
if rec:
    print("  recommended set   : {} ({}/{} features, input_dim {})".format(
        rec.get("label", "?"), rec.get("n_features", "?"),
        base.get("n_features", "?"), rec.get("input_dim", "?")))
    print("  fidelity          : CKA={:.3f}  trust={:.3f}  procrustes={:.4f}".format(
        rec.get("cka_similarity", float("nan")),
        rec.get("trustworthiness", float("nan")),
        rec.get("procrustes_distance", float("nan"))))
    print("  keep              : {}".format(", ".join(rec.get("features", []))))
else:
    print("  recommended set   : none met the CKA/trust thresholds (see log above)")
PY
    fi
    echo "  Ranking + evaluation JSON : $RUN_ROOT/feature_selection_results.json"
    echo "  Plot                      : $RUN_ROOT/feature_selection.png"
    echo "  Artifacts                 : $RUN_ROOT"
    echo "  sacct -j ${SLURM_JOB_ID:-<jobid>} --format=JobID,State,Elapsed,MaxRSS"
    echo "======================================================================"
}
trap print_summary EXIT

echo "Run root   : $RUN_ROOT"
echo "uv_vae dir : $UV_VAE_DIR"
echo "Parquet    : $PARQUET"
echo "Row filter : $ROW_FILTER"
echo "Methods    : $METHODS  (primary: $PRIMARY_METHOD)"
echo "Sizes      : keep=$KEEP_SIZES  eval-keep=$EVAL_KEEP"
if [ "$ALL_ROWS" = "1" ]; then
    echo "Rows       : ALL filtered rows  reference=$REFERENCE_ROWS"
else
    echo "Rows       : train=$SAMPLE_ROWS  reference=$REFERENCE_ROWS"
fi
echo "Clustering : $CLUSTERING"
echo "Threads    : $THREADS"

echo ""
echo "[$(date '+%F %T')] ===== BEGIN: rank features + retrain reduced VAEs ====="
stage_start=$SECONDS
python scripts/feature_selection_test.py \
    --parquet-path "$PARQUET" \
    --feature-spec-path "$UV_VAE_DIR/ml_features.json" \
    --output-dir "$RUN_ROOT" \
    --row-filter "$ROW_FILTER" \
    --sample-rows "$SAMPLE_ROWS" \
    $([ "$ALL_ROWS" = "1" ] && echo --all-rows) \
    --reference-rows "$REFERENCE_ROWS" \
    --methods "$METHODS" \
    --primary-method "$PRIMARY_METHOD" \
    --keep-sizes "$KEEP_SIZES" \
    --eval-keep "$EVAL_KEEP" \
    --pca-variance "$PCA_VARIANCE" \
    --n-components "$N_COMPONENTS" \
    --structural-rows "$STRUCTURAL_ROWS" \
    --epochs "$EPOCH_CEILING" \
    --patience "$PATIENCE" \
    --min-delta "$MIN_DELTA" \
    --active-unit-threshold "$AU_THRESHOLD" \
    --batch-size "$BATCH_SIZE" \
    --latent-dim "$LATENT_DIM" \
    --hidden-dims "$HIDDEN_DIMS" \
    --learning-rate "$LEARNING_RATE" \
    --kl-weight "$KL_WEIGHT" \
    --train-fraction "$TRAIN_FRACTION" \
    --clustering "$CLUSTERING" \
    --threads "$THREADS" \
    --seed "$SEED" \
    ${FORCE_KEEP:+--force-keep "$FORCE_KEEP"}
select_seconds=$((SECONDS - stage_start))
echo "[$(date '+%F %T')] ===== END: feature selection ($(fmt_seconds "$select_seconds")) ====="
