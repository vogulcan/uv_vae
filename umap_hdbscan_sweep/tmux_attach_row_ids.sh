#!/usr/bin/env bash
# Backfill file_row_number into per_parquet_inference results.
# Runs fast — DuckDB scan only, no GPU needed.
#
# Usage:
#   tmux new-session -d -s attach_ids 'bash umap_hdbscan_sweep/tmux_attach_row_ids.sh'
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

if command -v micromamba &>/dev/null; then
    eval "$(micromamba shell hook -s bash)"
    micromamba activate "${MAMBA_ENV:-uv_vae}"
elif command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-patrickg}"
fi

PARQUET_GLOB="${PARQUET_GLOB:-/data/lab/ppmseq_parquets/*.parquet}"
RESULTS_DIR="${RESULTS_DIR:-$HOME/pure-internship/umap_hdbscan_sweep/per_parquet_inference}"

LOG="$RESULTS_DIR/attach_row_ids_$(date -u +%Y%m%dT%H%M%SZ).log"
mkdir -p "$RESULTS_DIR"

echo "===== attach_row_ids  $(date -u) =====" | tee -a "$LOG"
echo "  results_dir:  $RESULTS_DIR"           | tee -a "$LOG"
echo "  parquet_glob: $PARQUET_GLOB"          | tee -a "$LOG"

sed -i 's/\r$//' "$SCRIPT_DIR/attach_row_ids.py" 2>/dev/null || true

python "$SCRIPT_DIR/attach_row_ids.py" \
    --results-dir  "$RESULTS_DIR" \
    --parquet-glob "$PARQUET_GLOB" \
    2>&1 | tee -a "$LOG"

echo "===== DONE  $(date -u) =====" | tee -a "$LOG"
