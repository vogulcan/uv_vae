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

# Fast-iteration sizes. Bump SAMPLE_ROWS toward the 2.5M-5M sweet spot for a final call.
SAMPLE_ROWS="${SAMPLE_ROWS:-200000}"      # training rows per VAE
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

echo "[$(date '+%F %T')] ===== BEGIN: VAE feature selection ====="
echo "  parquet   : $PARQUET"
echo "  methods   : $METHODS  (primary: $PRIMARY_METHOD)"
echo "  sizes     : keep=$KEEP_SIZES  eval-keep=$EVAL_KEEP"
echo "  sample    : train=$SAMPLE_ROWS  reference=$REFERENCE_ROWS"
echo "  clustering: $CLUSTERING"
echo "  output    : $RUN_ROOT"

python scripts/feature_selection_test.py \
    --parquet-path "$PARQUET" \
    --feature-spec-path "$UV_VAE_DIR/ml_features.json" \
    --output-dir "$RUN_ROOT" \
    --row-filter "$ROW_FILTER" \
    --sample-rows "$SAMPLE_ROWS" \
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

echo "[$(date '+%F %T')] ===== END: feature selection ($SECONDS s) ====="
echo "  Ranking + evaluation JSON : $RUN_ROOT/feature_selection_results.json"
echo "  Plot                      : $RUN_ROOT/feature_selection.png"
echo "  (The recommended minimal feature set is printed at the end of the log above.)"
