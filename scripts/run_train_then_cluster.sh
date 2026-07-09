#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_train_then_cluster.sh --cluster-row-filter "<duckdb sql filter>"

Required arguments:
  --cluster-row-filter   DuckDB SQL filter passed to scripts/run_variant_cluster_pipeline.py
EOF
}

CLUSTER_ROW_FILTER=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cluster-row-filter)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --cluster-row-filter" >&2
        usage >&2
        exit 2
      fi
      CLUSTER_ROW_FILTER="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$CLUSTER_ROW_FILTER" ]]; then
  echo "Missing required argument: --cluster-row-filter" >&2
  usage >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

UV_PATH="${UV_PATH:-}"
if [[ -z "$UV_PATH" ]]; then
  UV_PATH="$(command -v uv || true)"
fi
if [[ -z "$UV_PATH" || ! -x "$UV_PATH" ]]; then
  echo "Unable to find executable uv. Set UV_PATH=/path/to/uv and rerun." >&2
  exit 1
fi
export UV_PATH

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/artifacts/train_then_cluster_${TIMESTAMP}}"

PARQUET_PATH="${PARQUET_PATH:-$REPO_ROOT/data/ddbR9-b2-ppm0029.featuremap.parquet}"
FEATURE_SPEC_PATH="${FEATURE_SPEC_PATH:-$REPO_ROOT/ml_features.json}"

TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-$RUN_ROOT/training}"
CLUSTER_OUTPUT_ROOT="${CLUSTER_OUTPUT_ROOT:-$RUN_ROOT/clustering}"

TRAIN_ROW_FILTER="${TRAIN_ROW_FILTER:-st = 'MIXED' AND et = 'MIXED' AND FILT = 1}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4096}"
TRAIN_LATENT_DIM="${TRAIN_LATENT_DIM:-16}"
TRAIN_HIDDEN_DIMS="${TRAIN_HIDDEN_DIMS:-256,128}"
TRAIN_LEARNING_RATE="${TRAIN_LEARNING_RATE:-1e-3}"
TRAIN_KL_WEIGHT="${TRAIN_KL_WEIGHT:-0.05}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.9}"
TRAIN_SEED="${TRAIN_SEED:-42}"
TRAIN_THREADS="${TRAIN_THREADS:-12}"

CLUSTER_SEED="${CLUSTER_SEED:-$TRAIN_SEED}"
CLUSTER_SAMPLE_ROWS="${CLUSTER_SAMPLE_ROWS:-1000000}"
CLUSTER_USE_ALL="${CLUSTER_USE_ALL:-0}"
CLUSTER_THREADS="${CLUSTER_THREADS:-12}"
DEVICE="${DEVICE:-auto}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-4096}"
SCAN_BATCH_ROWS="${SCAN_BATCH_ROWS:-100000}"
UMAP_FIT_ROWS="${UMAP_FIT_ROWS:-100000}"
PLOT_ROWS="${PLOT_ROWS:-100000}"
UMAP_N_NEIGHBORS="${UMAP_N_NEIGHBORS:-30}"
UMAP_MIN_DIST="${UMAP_MIN_DIST:-0.05}"
UMAP_METRIC="${UMAP_METRIC:-euclidean}"
HDBSCAN_MIN_CLUSTER_SIZE="${HDBSCAN_MIN_CLUSTER_SIZE:-250}"
HDBSCAN_MIN_SAMPLES="${HDBSCAN_MIN_SAMPLES:-25}"
SIGPROFILER_CPU="${SIGPROFILER_CPU:-12}"
COSMIC_VERSION="${COSMIC_VERSION:-3.5}"
GENOME_BUILD="${GENOME_BUILD:-GRCh38}"
DUCKDB_MEMORY_LIMIT="${DUCKDB_MEMORY_LIMIT:-4GB}"
COLOR_COLUMNS="${COLOR_COLUMNS:-BCSQ,RAW_VAF,DP,SMQ_BEFORE,SMQ_AFTER,EDIST,MAPQ,SNVQ}"

export PYTHONHASHSEED="${PYTHONHASHSEED:-$TRAIN_SEED}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

TRAIN_JSON="$(mktemp)"
cleanup() {
  rm -f "$TRAIN_JSON"
}
trap cleanup EXIT

TRAIN_SAMPLE_ARGS=()
if [[ "${TRAIN_USE_ALL:-0}" == "1" ]]; then
  TRAIN_SAMPLE_ARGS+=(--use-all)
else
  TRAIN_SAMPLE_ROWS="${TRAIN_SAMPLE_ROWS:-1000000}"
  TRAIN_SAMPLE_ARGS+=(--sample-rows "$TRAIN_SAMPLE_ROWS")
fi

CLUSTER_SAMPLE_ARGS=()
if [[ "$CLUSTER_USE_ALL" == "1" ]]; then
  CLUSTER_SAMPLE_ARGS+=(--use-all)
else
  CLUSTER_SAMPLE_ARGS+=(--sample-rows "$CLUSTER_SAMPLE_ROWS")
fi

echo "Training pipeline output dir: $TRAIN_OUTPUT_DIR"
echo "Clustering pipeline output dir: $CLUSTER_OUTPUT_ROOT"
echo "Clustering row filter: $CLUSTER_ROW_FILTER"

if [[ -n "${MODEL_PATH:-}" ]]; then
  if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Configured MODEL_PATH does not exist: $MODEL_PATH" >&2
    exit 1
  fi
  MODEL_PATH="$(cd "$(dirname "$MODEL_PATH")" && pwd)/$(basename "$MODEL_PATH")"
  TRAIN_RUN_DIR="$(dirname "$MODEL_PATH")"
  echo "Skipping training and using existing checkpoint: $MODEL_PATH"
else
  "$UV_PATH" run python scripts/run_model_training_pipeline.py \
    --parquet-path "$PARQUET_PATH" \
    --feature-spec-path "$FEATURE_SPEC_PATH" \
    --output-dir "$TRAIN_OUTPUT_DIR" \
    --row-filter "$TRAIN_ROW_FILTER" \
    --epochs "$TRAIN_EPOCHS" \
    --batch-size "$TRAIN_BATCH_SIZE" \
    --latent-dim "$TRAIN_LATENT_DIM" \
    --hidden-dims "$TRAIN_HIDDEN_DIMS" \
    --learning-rate "$TRAIN_LEARNING_RATE" \
    --kl-weight "$TRAIN_KL_WEIGHT" \
    --train-fraction "$TRAIN_FRACTION" \
    --seed "$TRAIN_SEED" \
    --threads "$TRAIN_THREADS" \
    "${TRAIN_SAMPLE_ARGS[@]}" > "$TRAIN_JSON"
  cat "$TRAIN_JSON"

  TRAIN_RUN_DIR="$(
    "$UV_PATH" run python - "$TRAIN_JSON" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)
print(payload["run_dir"])
PY
  )"

  MODEL_PATH="$TRAIN_RUN_DIR/model.pt"
  if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Expected checkpoint was not created: $MODEL_PATH" >&2
    exit 1
  fi
fi

echo "Using trained checkpoint: $MODEL_PATH"

"$UV_PATH" run python scripts/run_variant_cluster_pipeline.py \
  --checkpoint-path "$MODEL_PATH" \
  --parquet-path "$PARQUET_PATH" \
  --row-filter "$CLUSTER_ROW_FILTER" \
  --output-root "$CLUSTER_OUTPUT_ROOT" \
  --seed "$CLUSTER_SEED" \
  --threads "$CLUSTER_THREADS" \
  --device "$DEVICE" \
  --embed-batch-size "$EMBED_BATCH_SIZE" \
  --scan-batch-rows "$SCAN_BATCH_ROWS" \
  --umap-fit-rows "$UMAP_FIT_ROWS" \
  --plot-rows "$PLOT_ROWS" \
  --umap-n-neighbors "$UMAP_N_NEIGHBORS" \
  --umap-min-dist "$UMAP_MIN_DIST" \
  --umap-metric "$UMAP_METRIC" \
  --hdbscan-min-cluster-size "$HDBSCAN_MIN_CLUSTER_SIZE" \
  --hdbscan-min-samples "$HDBSCAN_MIN_SAMPLES" \
  --sigprofiler-cpu "$SIGPROFILER_CPU" \
  --cosmic-version "$COSMIC_VERSION" \
  --genome-build "$GENOME_BUILD" \
  --duckdb-memory-limit "$DUCKDB_MEMORY_LIMIT" \
  --color-columns "$COLOR_COLUMNS" \
  "${CLUSTER_SAMPLE_ARGS[@]}"

echo "Training run dir: $TRAIN_RUN_DIR"
echo "Clustering output root: $CLUSTER_OUTPUT_ROOT"
