#!/usr/bin/env bash
# Trial -> projection -> (conditional) full parametric UMAP sweep, for miletus.
#
# miletus has no SLURM, so this is a plain script meant to be launched detached under tmux
# (see the bottom of this comment). It does four things in order:
#
#   1. WAITS for the GPU. Either until a named tmux session disappears, or until nvidia-smi
#      reports enough free memory, or both. This is the contingency for launching it while
#      another job still owns the card -- the script parks itself and starts on its own.
#   2. Runs a TRIAL: two fit sizes, one graph config, every training objective, one
#      replicate. Two sizes is the minimum that pins the UMAP fit scaling exponent, and
#      running all three objectives costs only their training time because they share the
#      one fit. This is what answers "is kNN-overlap acceptable" and "what does the UMAP
#      loss phase actually cost".
#   3. PROJECTS the full grid from those measured timings and checks it against MAX_HOURS.
#   4. Runs the FULL SWEEP only if the projection fits the budget and the trial's encoders
#      were actually usable.
#
# Everything is an environment variable, nothing needs editing:
#
#   WAIT_FOR_SESSION   tmux session to wait for (default: none). e.g. WAIT_FOR_SESSION=paramsweep
#   REQUIRE_FREE_GB    wait until this many GB are free on the GPU (default 40)
#   MAX_WAIT_HOURS     give up waiting after this long (default 24)
#   MAX_HOURS          budget the projection is checked against (default 48)
#   GPU_GB             per-process GPU budget passed to the sweep (default 40)
#   SEEDS              replicates in the full sweep (default 3)
#   SIZES/MIN_DIST/NN/MODES   the grid (defaults are the full one)
#   AUTO_RUN           set to 0 to stop after the projection (default 1)
#   TRIAL_ONLY         set to 1 to stop after the trial, before projecting (default 0)
#
# Launch it detached:
#
#   sed -i 's/\r$//' ~/pure-internship/umap_hdbscan_sweep/*.sh
#   tmux new-session -d -s paramsweep3 \
#     'WAIT_FOR_SESSION=<your-running-session> bash ~/pure-internship/umap_hdbscan_sweep/run_parametric_sweep.sh'
#   tmux attach -t paramsweep3        # to watch
set -uo pipefail

REPO="${REPO:-$HOME/pure-internship}"
SWEEP_DIR="$REPO/umap_hdbscan_sweep"
EMBED_DIR="${EMBED_DIR:-$REPO/uv_vae/runs/train_multi_20260802T192756Z/stage1_embed}"
OUT_ROOT="${OUT_ROOT:-$SWEEP_DIR/umap_tests/parametric_sweep}"

WAIT_FOR_SESSION="${WAIT_FOR_SESSION:-}"
REQUIRE_FREE_GB="${REQUIRE_FREE_GB:-40}"
MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-24}"
MAX_HOURS="${MAX_HOURS:-48}"
GPU_GB="${GPU_GB:-40}"
SEEDS="${SEEDS:-3}"
SIZES="${SIZES:-2000000,5000000,10000000,25000000}"
MIN_DIST="${MIN_DIST:-0.0,0.1,0.25}"
NN="${NN:-15,30,50}"
MODES="${MODES:-regress,umap,hybrid}"
AUTO_RUN="${AUTO_RUN:-1}"
TRIAL_ONLY="${TRIAL_ONLY:-0}"

# The trial: smallest two sizes so it is cheap, min_dist 0.1 because sweep 1 put the
# fidelity/stability knee there, n_neighbors 15 as the sweep 1 baseline.
TRIAL_SIZES="${TRIAL_SIZES:-2000000,5000000}"
TRIAL_MIN_DIST="${TRIAL_MIN_DIST:-0.1}"
TRIAL_NN="${TRIAL_NN:-15}"

export TQDM_DISABLE=1

MAMBA_ENV="${MAMBA_ENV:-uv_vae}"
RUN=(micromamba run -n "$MAMBA_ENV" python)
command -v micromamba >/dev/null 2>&1 || RUN=(python)

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

mkdir -p "$OUT_ROOT"
TRIAL_DIR="$OUT_ROOT/trial"
FULL_DIR="$OUT_ROOT/full"
mkdir -p "$TRIAL_DIR" "$FULL_DIR"

# ── 1. wait for the card ──────────────────────────────────────────────────────
free_gb() {
  local used total
  read -r used total < <(nvidia-smi --query-gpu=memory.used,memory.total \
      --format=csv,noheader,nounits | head -1 | tr -d ',')
  echo $(( (total - used) / 1024 ))
}

session_alive() {
  [ -n "$WAIT_FOR_SESSION" ] && tmux has-session -t "$WAIT_FOR_SESSION" 2>/dev/null
}

say "waiting for the GPU (need ${REQUIRE_FREE_GB} GB free${WAIT_FOR_SESSION:+, and for tmux session '$WAIT_FOR_SESSION' to end})"
deadline=$(( $(date +%s) + MAX_WAIT_HOURS * 3600 ))
while :; do
  blocked=""
  if session_alive; then
    blocked="tmux session '$WAIT_FOR_SESSION' still running"
  else
    available=$(free_gb)
    if [ "$available" -lt "$REQUIRE_FREE_GB" ]; then
      blocked="only ${available} GB free, need ${REQUIRE_FREE_GB} GB"
    fi
  fi
  [ -z "$blocked" ] && break
  if [ "$(date +%s)" -ge "$deadline" ]; then
    say "GAVE UP after ${MAX_WAIT_HOURS} h: $blocked"
    exit 1
  fi
  say "  blocked: $blocked -- checking again in 5 min"
  sleep 300
done
say "GPU is free ($(free_gb) GB) -- starting"

# The scripts are edited on Windows; CRLF makes python choke on the shebang line and bash
# on everything. Cheap to redo every run.
sed -i 's/\r$//' "$SWEEP_DIR"/*.py 2>/dev/null || true

# ── 2. the trial ──────────────────────────────────────────────────────────────
say "=== TRIAL: sizes $TRIAL_SIZES, min_dist $TRIAL_MIN_DIST, nn $TRIAL_NN, modes $MODES, 1 replicate ==="
"${RUN[@]}" "$SWEEP_DIR/parametric_sweep.py" \
  --embed-dir "$EMBED_DIR" \
  --output-dir "$TRIAL_DIR" \
  --fit-rows "$TRIAL_SIZES" \
  --min-dist "$TRIAL_MIN_DIST" \
  --n-neighbors "$TRIAL_NN" \
  --modes "$MODES" \
  --seeds 1 \
  --gpu-budget-gb "$GPU_GB" \
  2>&1 | tee "$OUT_ROOT/trial.log"
trial_rc=${PIPESTATUS[0]}

if [ "$trial_rc" -ne 0 ]; then
  say "TRIAL FAILED (exit $trial_rc) -- see $OUT_ROOT/trial.log. Not starting the full sweep."
  exit "$trial_rc"
fi
say "trial finished, wrote $TRIAL_DIR/parametric_sweep.json"

if [ "$TRIAL_ONLY" = "1" ]; then
  say "TRIAL_ONLY=1 -- stopping here for you to read the numbers."
  exit 0
fi

# ── 3. is the encoder worth sweeping? ─────────────────────────────────────────
# Deliberately NOT gated on kNN-overlap alone. The known regress failure scored 0.055
# overlap, but the reverse also happens: an encoder can disagree with transform() about
# neighbour *order* inside a tight island while still producing the same clusters. The gate
# is therefore "did the encoders work at all" -- no failures, a map as faithful to the 16-D
# latent as cuML's, and cluster labels that broadly agree. Judging which objective is best
# is what the full sweep is for.
say "=== GATE: did the trial's encoders do anything real? ==="
"${RUN[@]}" - "$TRIAL_DIR/parametric_sweep.json" <<'PY'
import json, sys
payload = json.loads(open(sys.argv[1]).read())
summaries = payload["summaries"]
floor = payload["reference_lines"]["determinism_floor_knn_overlap"]

rows = []
for key, s in summaries.items():
    if s.get("failed_replicates"):
        rows.append((key, None, None, None, s.get("failure", "failed")))
        continue
    rows.append((
        key,
        s["distillation"]["knn_overlap_k15"],
        s.get("rnx_ratio"),
        s.get("clustering", {}).get("vs_cuml_adjusted_rand"),
        None,
    ))

print(f"  {'cell':<40}{'knn15':>9}{'rnx_ratio':>11}{'ARIvs':>9}")
for key, knn, ratio, ari, failure in rows:
    if failure:
        print(f"  {key:<40}  FAILED: {failure}")
    else:
        print(f"  {key:<40}{knn:>9.4f}{ratio:>11.3f}{ari:>9.3f}")

ok = [r for r in rows if r[4] is None]
if not ok:
    print("\n  GATE FAILED: every trial cell failed to produce an embedding.")
    sys.exit(2)

best_knn = max(r[1] for r in ok)
best_ratio = max(r[2] for r in ok if r[2] is not None)
best_ari = max(r[3] for r in ok if r[3] is not None)
print(f"\n  best knn15 {best_knn:.4f}  (determinism floor {floor:.4f})")
print(f"  best rnx_ratio {best_ratio:.3f}  (1.0 = as faithful to the 16-D latent as cuML)")
print(f"  best ARI vs cuML clustering {best_ari:.3f}")

if best_knn < floor:
    print(f"\n  NOTE: no encoder reaches the determinism floor, so none of them reproduces a")
    print(f"  given fit as closely as re-running UMAP would. That is not automatically fatal")
    print(f"  -- check rnx_ratio and ARIvs above -- but it does mean the encoder is not a")
    print(f"  drop-in for transform(); it is a different map of the same data.")

if best_ratio < 0.9 or best_ari < 0.5:
    print(f"\n  GATE FAILED: rnx_ratio {best_ratio:.3f} / ARIvs {best_ari:.3f} say the encoder")
    print(f"  is not learning a usable map. Fix that before spending a day on the grid.")
    sys.exit(3)

print("\n  GATE PASSED: the encoders are learning a usable map.")
PY
gate_rc=$?
if [ "$gate_rc" -ne 0 ]; then
  say "GATE FAILED (exit $gate_rc) -- not starting the full sweep."
  exit "$gate_rc"
fi

# ── 4. project the full grid ──────────────────────────────────────────────────
say "=== PROJECTION: full grid cost from the trial's measured timings ==="
"${RUN[@]}" "$SWEEP_DIR/parametric_sweep.py" \
  --plan-from "$TRIAL_DIR/parametric_sweep.json" \
  --fit-rows "$SIZES" --min-dist "$MIN_DIST" --n-neighbors "$NN" \
  --modes "$MODES" --seeds "$SEEDS" --max-hours "$MAX_HOURS" \
  2>&1 | tee "$OUT_ROOT/projection.log"
plan_rc=${PIPESTATUS[0]}

if [ "$plan_rc" -ne 0 ]; then
  say "PROJECTION EXCEEDS ${MAX_HOURS} h -- not starting. See the suggested cuts above,"
  say "then rerun with e.g. SIZES=2000000,5000000,10000000 or SEEDS=2."
  exit 0
fi

if [ "$AUTO_RUN" != "1" ]; then
  say "AUTO_RUN=$AUTO_RUN -- projection fits the budget but stopping as asked."
  exit 0
fi

# ── 5. the full sweep ─────────────────────────────────────────────────────────
say "=== FULL SWEEP: sizes $SIZES, min_dist $MIN_DIST, nn $NN, modes $MODES, $SEEDS replicates ==="
say "    partial results land in $FULL_DIR/parametric_sweep_partial.json after each replicate"
"${RUN[@]}" "$SWEEP_DIR/parametric_sweep.py" \
  --embed-dir "$EMBED_DIR" \
  --output-dir "$FULL_DIR" \
  --fit-rows "$SIZES" \
  --min-dist "$MIN_DIST" \
  --n-neighbors "$NN" \
  --modes "$MODES" \
  --seeds "$SEEDS" \
  --gpu-budget-gb "$GPU_GB" \
  2>&1 | tee "$OUT_ROOT/full_sweep.log"
full_rc=${PIPESTATUS[0]}

if [ "$full_rc" -ne 0 ]; then
  say "FULL SWEEP FAILED (exit $full_rc). Partial results, if any, are in"
  say "  $FULL_DIR/parametric_sweep_partial.json"
  exit "$full_rc"
fi
say "DONE. Results in $FULL_DIR/parametric_sweep.json"
