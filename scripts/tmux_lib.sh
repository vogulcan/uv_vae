#!/bin/bash
# Shared plumbing for the tmux runners that replace the SLURM scripts.
#
# The new node has no scheduler, so everything SLURM used to provide has to be
# rebuilt here: a work queue with bounded concurrency (was --array + the
# partition's slot limit), per-task logs (was %A_%a.log), resume after a crash
# (was requeue), and a resource budget per task (was --cpus-per-task / --gres).
#
# Source it, do not execute it:
#     source "${UV_VAE_DIR:-$HOME/uv_vae}/scripts/tmux_lib.sh"
#
# Every runner honours the same environment variables -- see TMUX_RUNNERS.md:
#   CONCURRENCY     tasks in flight at once (default 1)
#   GPU_TOTAL_GB    GPU ceiling for the WHOLE sweep, split across workers (16)
#   THREADS_TOTAL   CPU threads for the whole sweep, split across workers (nproc)
#   MAMBA_ENV       micromamba env name (default 'uv_vae') -- the miletus setup
#   CONDA_ENV       conda env name; overrides MAMBA_ENV when set
#   FORCE           1 = re-run tasks that already have a .done marker
#   DRY_RUN         1 = print what would run, touch nothing

set -euo pipefail

# bash 4.3 introduced `wait -n`, which is what bounds the queue.
if [ "${BASH_VERSINFO[0]}" -lt 4 ] || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -lt 3 ]; }; then
    echo "ERROR: bash >= 4.3 required (found ${BASH_VERSION}); 'wait -n' is unavailable." >&2
    exit 1
fi

# ── Formatting ──────────────────────────────────────────────────────────────
uvv_fmt_seconds() {
    local s=$1
    printf '%dh %02dm %02ds' $((s / 3600)) $(((s % 3600) / 60)) $((s % 60))
}

uvv_log() { echo "[$(date '+%F %T')] $*"; }

uvv_rule() { printf '%.0s=' {1..72}; echo; }

# ── Environment ─────────────────────────────────────────────────────────────
uvv_detect_cores() {
    if command -v nproc >/dev/null 2>&1; then
        nproc
    else
        getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8
    fi
}

# Resolution order:
#   1. CONDA_ENV  -- explicit conda env; setting it is an unambiguous instruction
#   2. MAMBA_ENV  -- micromamba env, default 'uv_vae'. This is the miletus path.
#   3. $UV_VAE_DIR/.venv
#   4. system python
#
# micromamba needs its OWN hook: `conda shell.bash hook` is not a micromamba
# subcommand, and on a node with no conda binary that line expands to nothing and
# the following `conda activate` aborts the whole runner under `set -e`. Each
# branch is therefore guarded by `command -v` rather than assuming the manager
# exists.
uvv_activate_env() {
    local root="${UV_VAE_DIR:-$HOME/uv_vae}"
    local mamba_env="${MAMBA_ENV:-uv_vae}"

    if [ -n "${CONDA_ENV:-}" ]; then
        if command -v conda >/dev/null 2>&1; then
            eval "$(conda shell.bash hook)"
            conda activate "$CONDA_ENV"
            uvv_log "environment: conda '$CONDA_ENV' ($(command -v python))"
            return 0
        fi
        uvv_log "WARNING: CONDA_ENV='$CONDA_ENV' set but no conda binary found; trying micromamba"
    fi

    if command -v micromamba >/dev/null 2>&1; then
        # micromamba refuses to hook without a root prefix; $HOME/micromamba is its
        # own install default, so this only fills in when the shell rc did not.
        export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
        eval "$(micromamba shell hook -s bash)"
        if micromamba activate "$mamba_env" 2>/dev/null; then
            uvv_log "environment: micromamba '$mamba_env' ($(command -v python))"
            return 0
        fi
        uvv_log "WARNING: micromamba env '$mamba_env' not found; falling back"
    fi

    if [ -f "$root/.venv/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$root/.venv/bin/activate"
        uvv_log "environment: venv $root/.venv ($(command -v python))"
        return 0
    fi

    uvv_log "environment: system python ($(command -v python)) -- no CONDA_ENV, no micromamba env '$mamba_env', no .venv"
    return 0
}

# Exported once per RUNNER process, then inherited by every task.
uvv_export_determinism() {
    local seed="${1:-42}"
    export PYTHONHASHSEED="$seed"
    # Must be in the environment BEFORE python starts: cuBLAS reads it when it
    # creates its handle, so training.seed_everything() setting it is too late and
    # torch.use_deterministic_algorithms(True) then raises or silently does nothing.
    export CUBLAS_WORKSPACE_CONFIG=":4096:8"
    export TQDM_DISABLE=1
    # cuML patches sklearn/UMAP/HDBSCAN globally and its HDBSCAN has no
    # relative_validity_. Training does not want it; opt in per-stage instead.
    export UV_VAE_ENABLE_CUML="${UV_VAE_ENABLE_CUML:-0}"
}

# Divide the sweep-wide budgets across concurrent workers and report the result.
# Both caps are per PROCESS, so N workers each taking the full budget would be N
# times the intended footprint.
uvv_plan_resources() {
    local concurrency="$1"
    local gpu_total="${GPU_TOTAL_GB:-16}"
    local threads_total="${THREADS_TOTAL:-$(uvv_detect_cores)}"

    UVV_GPU_PER_WORKER=$(awk -v t="$gpu_total" -v c="$concurrency" 'BEGIN{printf "%.2f", t/c}')
    UVV_THREADS_PER_WORKER=$(( threads_total / concurrency ))
    [ "$UVV_THREADS_PER_WORKER" -lt 1 ] && UVV_THREADS_PER_WORKER=1

    export UVV_GPU_PER_WORKER UVV_THREADS_PER_WORKER
    uvv_log "resources: concurrency=$concurrency  gpu=${gpu_total}GB total -> ${UVV_GPU_PER_WORKER}GB/worker  threads=${threads_total} total -> ${UVV_THREADS_PER_WORKER}/worker"
}

# What batch size fits in one worker's slice of the GPU. Printed before the queue
# starts so an impossible config is caught in seconds rather than hours.
uvv_report_batch_ceiling() {
    local budget="$1"; shift
    local batches=("$@")
    python - "$budget" "${batches[@]}" <<'PY' || true
import sys
sys.path.insert(0, __import__("os").environ.get("UV_VAE_DIR", "."))
try:
    from uv_vae import gpu_budget
except Exception as exc:
    print(f"  (batch ceiling unavailable: {exc})")
    raise SystemExit(0)

budget = float(sys.argv[1])
# ml_features.json shape; matches scripts/gpu_preflight.py defaults.
per_row = gpu_budget.bytes_per_row(
    n_categorical=11, n_numeric=30, embedding_dims=[8] * 11,
    hidden_dims=[256, 128], latent_dim=16,
    categorical_cardinalities=[16] * 11, amp=True,
)
ceiling = gpu_budget.max_batch_rows(per_row, budget_gb=budget)
print(f"  GPU budget {budget:.2f} GB/worker -> largest safe batch ~{ceiling:,} rows")
over = []
for raw in sys.argv[2:]:
    batch = int(raw)
    gb = batch * per_row / gpu_budget.BYTES_PER_GB
    flag = "" if batch <= ceiling else "   <-- EXCEEDS BUDGET"
    print(f"    batch {batch:>9,} -> {gb:6.2f} GB{flag}")
    if batch > ceiling:
        over.append(batch)
if over:
    print("  Lower CONCURRENCY, raise GPU_TOTAL_GB, or set UV_VAE_GPU_OOM_POLICY=clamp.")
PY
}

# Windows-edited scripts arrive with CRLF and fail with 'bad interpreter'.
uvv_strip_crlf() {
    local target
    for target in "$@"; do
        [ -f "$target" ] || continue
        if grep -qU $'\r' "$target" 2>/dev/null; then
            sed -i 's/\r$//' "$target"
            uvv_log "stripped CRLF from $target"
        fi
    done
}

# ── Work queue ──────────────────────────────────────────────────────────────
# uvv_run_queue <concurrency> <state_dir> <log_dir> <task_fn> <task_id>...
#
# task_fn is a shell function name called as `task_fn <task_id>` in a subshell
# with UVV_GPU_PER_WORKER / UVV_THREADS_PER_WORKER already exported. Its stdout
# and stderr go to <log_dir>/task_<id>.log.
uvv_run_queue() {
    local concurrency="$1"; shift
    local state_dir="$1"; shift
    local log_dir="$1"; shift
    local task_fn="$1"; shift
    local tasks=("$@")

    mkdir -p "$state_dir" "$log_dir"

    local started=0 skipped=0
    local queue_start=$SECONDS

    local task
    for task in "${tasks[@]}"; do
        local marker="$state_dir/task_${task}.done"
        if [ -f "$marker" ] && [ "${FORCE:-0}" != "1" ]; then
            uvv_log "task $task: SKIP (already done -- set FORCE=1 to re-run)"
            skipped=$((skipped + 1))
            continue
        fi
        rm -f "$state_dir/task_${task}.failed"

        if [ "${DRY_RUN:-0}" = "1" ]; then
            uvv_log "task $task: DRY_RUN, would run $task_fn $task -> $log_dir/task_${task}.log"
            continue
        fi

        # Block until a worker slot frees up. `|| true` because wait -n reports the
        # finished job's exit status, and a failed task must not kill the queue.
        while [ "$(jobs -rp | wc -l)" -ge "$concurrency" ]; do
            wait -n || true
        done

        uvv_log "task $task: START -> $log_dir/task_${task}.log"
        (
            local task_start=$SECONDS
            if "$task_fn" "$task" > "$log_dir/task_${task}.log" 2>&1; then
                date -u +"%Y-%m-%dT%H:%M:%SZ" > "$marker"
                echo "$((SECONDS - task_start))" >> "$marker"
                echo "[$(date '+%F %T')] task $task: DONE in $(uvv_fmt_seconds $((SECONDS - task_start)))"
            else
                local code=$?
                echo "$code" > "$state_dir/task_${task}.failed"
                echo "[$(date '+%F %T')] task $task: FAILED (exit $code) -- see $log_dir/task_${task}.log"
            fi
        ) &
        started=$((started + 1))
    done

    wait

    local failed
    failed=$(find "$state_dir" -maxdepth 1 -name 'task_*.failed' 2>/dev/null | wc -l)
    uvv_rule
    uvv_log "queue finished in $(uvv_fmt_seconds $((SECONDS - queue_start))): started=$started skipped=$skipped failed=$failed"
    if [ "$failed" -gt 0 ]; then
        find "$state_dir" -maxdepth 1 -name 'task_*.failed' -printf '  failed: %f\n' 2>/dev/null || true
        uvv_log "re-run just the failures by launching again -- completed tasks are skipped"
    fi
    uvv_rule
    return 0
}

# ── tmux launch ─────────────────────────────────────────────────────────────
# Export everything the child re-reads, so an override given on the launch line
# survives into the tmux session. tmux inherits the caller's environment, so
# exporting is both simpler and safer than rebuilding a quoted VAR=... prefix --
# a variable left out of that prefix silently reverted to its $HOME default.
uvv_export_child_env() {
    local name
    for name in "$@"; do
        # Explicit `if`, not `[ ... ] && export`: the && form returns non-zero when
        # the last name in the list happens to be unset, and `set -e` would then
        # abort the launcher on the line before tmux starts.
        if [ -n "${!name+set}" ]; then
            export "${name?}"
        fi
    done
    return 0
}

# uvv_launch_tmux <session> <log_dir> <command...>
# Starts the command detached in window 'run' and adds a 'watch' window with
# nvidia-smi + a tail of the logs. Re-attaches instead of clobbering a live run.
uvv_launch_tmux() {
    local session="$1"; shift
    local log_dir="$1"; shift
    local command="$*"

    if ! command -v tmux >/dev/null 2>&1; then
        echo "ERROR: tmux is not installed." >&2
        return 1
    fi

    if tmux has-session -t "$session" 2>/dev/null; then
        echo "Session '$session' already exists -- attach instead of starting a second run:"
        echo "    tmux attach -t $session"
        return 1
    fi

    mkdir -p "$log_dir"
    tmux new-session -d -s "$session" -n run "$command 2>&1 | tee '$log_dir/runner.log'; echo; echo '[runner exited -- press any key to close]'; read -n 1"
    tmux new-window -t "$session" -n watch \
        "watch -n 10 'nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv 2>/dev/null; echo; ls -1 '\"$log_dir\"' 2>/dev/null | tail -20'"
    tmux select-window -t "$session":run

    echo
    echo "Started detached tmux session '$session'."
    echo "    tmux attach -t $session       # watch it"
    echo "    Ctrl-b d                      # detach, leaves it running"
    echo "    Ctrl-b n / p                  # switch between the run and watch windows"
    echo "    tail -f $log_dir/runner.log   # follow without attaching"
    echo "    tmux kill-session -t $session # stop everything"
    echo
}
