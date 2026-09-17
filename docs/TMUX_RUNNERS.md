# tmux runners (no SLURM)

The new node — **`miletus.sabanciuniv.edu`** — has a single **NVIDIA RTX PRO 5000
Blackwell, 48 GB**, a **micromamba env named `uv_vae`**, and no scheduler.
These runners rebuild what SLURM used to provide — a bounded work queue, per-task
logs, resume after a crash, and a resource budget per task — on top of tmux.

| SLURM script | tmux replacement |
|---|---|
| `Batch_Size_Learning_Rate_Testing/scripts/run_batch_lr_sweep_gpu.sh` (`--array=0-9`) | `Batch_Size_Learning_Rate_Testing/scripts/tmux_batch_lr_sweep.sh` |
| `KL_Weight_Testing/scripts/run_kl_sweep.sh` (serial for-loop in one job) | `KL_Weight_Testing/scripts/tmux_kl_sweep.sh` |
| `Early_Stopping_Tests/scripts/run_train_only.sh` and `run_train_only_dropout0.2.sh` | `Early_Stopping_Tests/scripts/tmux_train_only.sh` |
| — | `uv_vae/scripts/tmux_lib.sh` (shared queue + env plumbing) |
| — | `uv_vae/scripts/gpu_preflight.py` (fails fast on a misconfigured GPU) |

The old `.sh` files are left in place unchanged, as a record of what was run before.

---

## Quick start

```bash
sed -i 's/\r$//' uv_vae/scripts/tmux_lib.sh */scripts/tmux_*.sh
```

```bash
COMBINED=/path/to/combined.featuremap.parquet CONCURRENCY=1 bash Early_Stopping_Tests/scripts/tmux_train_only.sh
```

That one run is the smoke test: it exercises the same streaming trainer, GPU cap and
determinism settings the sweeps use. Once it finishes cleanly, launch a sweep:

```bash
COMBINED=/path/to/combined.featuremap.parquet CONCURRENCY=3 bash Batch_Size_Learning_Rate_Testing/scripts/tmux_batch_lr_sweep.sh
```

Each script launches **detached** and returns immediately. Attach with `tmux attach -t
batch_lr` (or `kl_sweep`, `train_only`), detach again with `Ctrl-b d`. The session has
two windows: `run` (the queue) and `watch` (a 10-second `nvidia-smi` refresh plus the
log directory). Everything also lands on disk, so `tail -f <root>/logs/runner.log`
works without attaching.

---

## Choosing CONCURRENCY

**This is the setting that matters, and there is no single right answer — here is what
each value actually costs.**

`GPU_TOTAL_GB` (default **16**) is the ceiling for the *whole sweep*. Both caps it
installs — the torch allocator limit and the RMM pool that cuDF/cuML draw from — are
**per process**, so the runner divides the budget by `CONCURRENCY`. Four workers each
claiming 16 GB would be 64 GB on a 48 GB card.

> **These numbers are provisional — recalibrate before relying on them.** Two known
> issues: (1) `bytes_per_row` under-counts activations by roughly 1.8× (it misses the
> pre-concat embedding outputs, the non-inplace `nn.ReLU` outputs in both encoder and
> decoder, and the `reparameterize` temporaries), only partly offset by
> `AUTOGRAD_OVERHEAD = 2.5`; and (2) the table below assumes RMM is absent, so torch
> gets the whole budget. With cuDF/cuML installed, `UV_VAE_RMM_SHARE` (default 0.25)
> reserves a quarter for RMM and every ceiling drops accordingly — at 16 GB, from
> ~1,710,000 rows to ~1,282,000. Measure `torch.cuda.max_memory_allocated()` against
> the prediction on the real node and regenerate this table.

Derived from `uv_vae.gpu_budget` for this model (11 categorical + 30 numeric features,
hidden `256,128`, latent 16, AMP on): **~6.0 KB of GPU memory per row in a training
batch**, which is the figure the caveat above says is low. That gives:

| CONCURRENCY | GPU per worker | Largest safe batch | Threads/worker (32 cores) | Batch-LR configs that fit |
|---|---|---|---|---|
| **1** | 16.00 GB | ~1,710,000 rows | 32 | all 10 |
| **2** | 8.00 GB | ~855,000 rows | 16 | 8 of 10 — tasks 2 and 8 exceed |
| **3** | 5.33 GB | ~570,000 rows | 10 | 8 of 10 — tasks 2 and 8 exceed |
| **4** | 4.00 GB | ~427,000 rows | 8 | 6 of 10 — tasks 1, 2, 7, 8 exceed |

Tasks 2 and 8 are the `batch=1,048,576` configs (~5.9 GB each); tasks 1 and 7 are
`batch=524,288` (~2.9 GB). Every runner prints this table for the batch sizes you
actually queued, before it starts, so an impossible plan costs seconds not hours.

### Which is faster?

**The sweeps are I/O and CPU bound, not GPU bound.** The model is tiny — two hidden
layers of 256 and 128 — so a training step is microseconds of GPU work. The wall clock
goes to the DuckDB parquet scan and the per-chunk categorical/numeric encoding, and at
5 B rows every epoch re-scans the whole dataset. Consequences:

- **Raising CONCURRENCY helps** up to the point where workers start contending for
  parquet I/O and cores, because one worker's GPU idles while it waits for the next
  chunk. Overlapping runs fills those gaps.
- **It stops helping** once `THREADS_TOTAL / CONCURRENCY` drops below what DuckDB needs
  to keep a scan saturated, or once the storage backing the parquet is saturated.
- **GPU memory is rarely the binding constraint** — only the two 1 M-row-batch configs
  come close.

Recommended starting points:

| Situation | Setting |
|---|---|
| First run on the new node, or debugging | `CONCURRENCY=1` — one log to read, full budget, no contention |
| Batch-LR sweep, the 8 small-batch configs | `CONCURRENCY=3 TASKS="0 1 3 4 5 6 7 9"` |
| Batch-LR sweep, the two 1 M-batch configs | `CONCURRENCY=1 TASKS="2 8"` |
| KL sweep (all `batch=32768`, 0.18 GB each) | `CONCURRENCY=3` or `4` — GPU memory is a non-issue here; CPU threads decide |
| Node shared with other people | `CONCURRENCY=1` and lower `GPU_TOTAL_GB` |

Splitting the batch-LR sweep in two is the concrete recipe that gets both throughput
and the large-batch configs:

```bash
CONCURRENCY=3 TASKS="0 1 3 4 5 6 7 9" bash Batch_Size_Learning_Rate_Testing/scripts/tmux_batch_lr_sweep.sh
```

```bash
CONCURRENCY=1 TASKS="2 8" RUN_ID=same-as-above bash Batch_Size_Learning_Rate_Testing/scripts/tmux_batch_lr_sweep.sh
```

Pass the same `RUN_ID` to land both halves in one sweep root so the final
`evaluate_sweep.py --sweep-root` collection sees all ten configs.

### If a config exceeds the budget

"Exceeds budget" is a projection, not a crash. `UV_VAE_GPU_OOM_POLICY` decides:

| Value | Behaviour |
|---|---|
| `warn` (default) | Keep the requested batch, log the projection, let the torch cap raise a clean OOM if it really does not fit. **Default because batch size is the swept variable** — silently shrinking it would corrupt the batch-size sweep. |
| `clamp` | Shrink to the largest batch that fits. Use for production runs where finishing matters more than the exact batch size. The effective value is recorded as `effective_batch_size` in `training_report.json`. |
| `error` | Refuse to start. Use to fail fast. |

The per-row estimate is deliberately pessimistic (2.5× the forward activations, to
cover what autograd keeps alive), so `warn` is often fine — the cap means the worst
case is a clean OOM on that one task, not a run that takes the whole card.

---

## Environment variables

Shared by all three runners:

| Variable | Default | What it does |
|---|---|---|
| `CONCURRENCY` | `1` | Tasks in flight at once |
| `GPU_TOTAL_GB` | `16` | GPU ceiling for the whole sweep, divided by `CONCURRENCY` |
| `THREADS_TOTAL` | `nproc` | CPU threads for the whole sweep, divided by `CONCURRENCY` |
| `UV_VAE_GPU_OOM_POLICY` | `warn` | `warn` / `clamp` / `error` — see above |
| `MAMBA_ENV` | `uv_vae` | micromamba env name — the miletus setup |
| `CONDA_ENV` | *(unset)* | conda env name; overrides `MAMBA_ENV` when set |
| `MAMBA_ROOT_PREFIX` | `$HOME/micromamba` | only needed if micromamba lives elsewhere |
| `UV_VAE_DIR` | `$HOME/uv_vae` | Package root |
| `SEED` | `42` | Also sets `PYTHONHASHSEED` |
| `TASKS` | all | Space-separated subset, e.g. `TASKS="0 3 4"` |
| `FORCE` | `0` | `1` re-runs tasks that already have a `.done` marker |
| `DRY_RUN` | `0` | `1` prints the plan and exits |
| `FOREGROUND` | `0` | `1` runs in the current shell instead of a tmux session |
| `SKIP_PREFLIGHT` | `0` | `1` skips `gpu_preflight.py` |
| `RUN_ID` | UTC timestamp | Names the output root; reuse it to add tasks to an existing sweep |
| `SESSION` | per-script | tmux session name |

Per-script: `COMBINED` / `PARQUET` (input), `ROW_FILTER`, `BATCH_SIZE`,
`EPOCH_CEILING`, `PATIENCE`, `INPUT_DROPOUT`, `HIDDEN_DROPOUT`, `KL_WEIGHTS`,
`CLUSTER`, `TEST_PARQUET`. Read the header of each script for the full list.

---

## Resume and failures

Each finished task writes `<root>/state/task_<id>.done`; a failure writes
`task_<id>.failed` with the exit code. **A failing task does not stop the queue** —
the remaining tasks keep going, and the summary lists what failed.

Re-launch with the same `RUN_ID` and only the unfinished tasks run:

```bash
RUN_ID=20260727T120000Z CONCURRENCY=3 bash Batch_Size_Learning_Rate_Testing/scripts/tmux_batch_lr_sweep.sh
```

`FORCE=1` re-runs everything regardless.

Per-task output is in `<root>/logs/task_<id>.log`; the runner's own output is in
`<root>/logs/runner.log`.

---

## Deployment checklist for the new node

1. **Copy the tree.** Transferring the *repo* as-is works — `Python Files/` scripts
   resolve the package with `parents[2] / "uv_vae"`, which is satisfied both by the
   repo layout (root = `PURE Files/`) and by the sibling layout (root = `$HOME`).
   What matters is only that each script stays exactly two levels under a directory
   containing `uv_vae/`; changing its depth breaks the import silently.
   Then point the runners at it:
   ```bash
   export UV_VAE_DIR=$HOME/pure-internship/uv_vae
   export EARLY_STOPPING_DIR=$HOME/pure-internship/Early_Stopping_Tests
   export BATCH_LR_DIR=$HOME/pure-internship/Batch_Size_Learning_Rate_Testing
   export KL_TESTING_DIR=$HOME/pure-internship/KL_Weight_Testing
   ```
2. **Strip CRLF — always, whatever the transfer method.** The repo has no
   `.gitattributes`, so CRLF is committed into the blobs themselves: every tracked
   `.sh` file comes out of `git archive` *and* `git clone` with CRLF, not just out of
   an `scp`/`rsync` off Windows. bash rejects those with `bad interpreter:
   /bin/bash^M`. `uvv_strip_crlf` self-heals the files a runner is given, but the
   launcher has to be readable before it can run, so this step cannot be skipped:
   ```bash
   find ~/pure-internship -name '*.sh' -exec sed -i 's/\r$//' {} +
   ```
   (Adding `*.sh text eol=lf` to a `.gitattributes` and renormalising would retire
   this step permanently, at the cost of touching every script in one commit.)
3. **Environment**: `micromamba activate uv_vae` — the runners do this themselves via
   `MAMBA_ENV`, so this step is just to verify the env exists and has the stack.
   Confirm torch was built for Blackwell (`sm_120`); a cu126-or-older wheel has no
   `sm_120` kernels and every CUDA call fails at runtime:
   ```bash
   micromamba run -n uv_vae python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"
   ```
4. **Preflight**: `python uv_vae/scripts/gpu_preflight.py --batch-size 131072`. It
   checks that `sm_120` is in the wheel's arch list, that `CUBLAS_WORKSPACE_CONFIG` was
   exported *before* python started, that deterministic mode survives a real matmul and
   embedding backward, that the memory cap installs, and that a batch of the size you
   plan to use actually allocates.
5. **Smoke test**: `tmux_train_only.sh` on a small parquet.
6. **Then sweep.**

---

## Why some things changed

Notes on differences from the SLURM scripts, so results stay comparable:

- **`CUBLAS_WORKSPACE_CONFIG` is exported by the runner, before python.** cuBLAS reads
  it when it creates its handle. `training.seed_everything()` sets it too, but that runs
  after `import torch` — too late on a GPU, where
  `torch.use_deterministic_algorithms(True)` then either raises at the first matmul or
  is silently non-deterministic. The batch-LR and KL SLURM scripts already got this
  right; it is now uniform.
- **cuML no longer loads during training.** `train_with_early_stopping.py` used to call
  `cuml.accel.install()` whenever CUDA was present. cuML accelerates sklearn/UMAP/HDBSCAN
  — none of which the training path uses — while costing a CUDA context and an RMM pool
  that competes with the trainer for the same card. It is now behind
  `UV_VAE_ENABLE_CUML=1`, which the KL runner sets for the clustering stage only.
- **`preserve_insertion_order` is pinned on the streaming scans.** The streaming
  train/val split is decided by a row's *position* in the scan, and train and val each
  run their own scan — so both must agree on row order or rows land in both splits and
  the validation loss driving early stopping is measured on trained-on rows. DuckDB's
  default makes this true today; it is now asserted rather than assumed.
- **Output roots are timestamped.** They used to key off `$SLURM_JOB_ID` /
  `$SLURM_ARRAY_JOB_ID`, which collapse to the literal string `manual` with no
  scheduler — so consecutive runs quietly merged into one directory.
- **`THREADS` is detected, not read from `$SLURM_CPUS_PER_TASK`,** which does not exist
  here and silently fell back to 32 regardless of the real core count.
- **The environment resolves micromamba `uv_vae` first, then `.venv`; `CONDA_ENV`
  overrides both.** conda `patrickg` was hardcoded and does not exist on miletus.
  Critically, `conda shell.bash hook` is *not* a micromamba subcommand — on a node
  with no conda binary that line expands to nothing and the next `conda activate`
  aborts the runner under `set -e`, so each branch is guarded with `command -v`.
- **`training_report.json` now records `gpu_budget`, `gpu_environment` and
  `effective_batch_size`,** so a result can be traced to the device, driver and cap it
  was produced under.
