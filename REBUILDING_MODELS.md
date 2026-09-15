# Rebuilding the models

Retraining the pipeline from scratch. **Nothing here is needed to apply the existing
models** — for that see [`RUNNING_INFERENCE.md`](RUNNING_INFERENCE.md).

> Each stage consumes the one above it, so retraining any stage invalidates everything below.
> Retraining the VAE retires the UMAP encoder, `coords.npy`, the HDBSCAN model,
> `cohort_labels.npy`, every per-sample assignment and every SigProfiler result. Budget for
> rebuilding the whole chain, not one stage.

| | |
|---|---|
| Environment and GPU requirements | [`RUNNING_INFERENCE.md` §1](RUNNING_INFERENCE.md) |
| Parameters the shipped models used | [`pipeline_parameters.md`](pipeline_parameters.md) |
| Applying the result once rebuilt | [`RUNNING_INFERENCE.md`](RUNNING_INFERENCE.md) |

---

## 1. The chain

| # | Stage | Entry point | Produces | Cost |
|---|---|---|---|---|
| 0 | Dedup | `stage0_dedup.py` | 157.5 M loci from 5.08 B reads, one per `(CHROM,POS,REF,ALT)` | — |
| 1 | VAE train | `Early_Stopping_Tests/scripts/tmux_train_multi.sh` | `model.pt` | 4 h 25 m |
| 2 | VAE encode | `stage1_embed.py` | `latent.npy` (N×16), `context.parquet` | — |
| 3 | UMAP fit | `run_parametric_sweep.sh` | candidate encoders across the grid | — |
| 4 | UMAP apply | `apply_parametric_full.py` | `coords.npy` (N×2) | ~22 s |
| 5 | HDBSCAN | `tmux_final_models.sh` | model, `fit_indices.npy`, labels, SigProfiler | 18.7 s fit + 127 s label per cell |
| 6 | Per-sample | `tmux_per_parquet_inference.sh` | the 95-sample outputs | ~24 h |

Only stages 1, 4, 5 and 6 have measured timings; the others were not separately recorded.

**Stage 0 → 2 order is the contract.** `latent.npy` and `context.parquet` are row-aligned by
position, and so is everything derived from them. Re-running dedup produces a different row
order and silently invalidates every downstream array.

---

## 2. Retraining the VAE (stage 1)

```bash
# preflight once — cheap, and the training run reuses its cache
STATS_ONLY=1 bash Early_Stopping_Tests/scripts/tmux_train_multi.sh

# then train
tmux new-session -d -s train_multi 'bash Early_Stopping_Tests/scripts/tmux_train_multi.sh'
```

The runner's defaults **are** the shipped configuration, so this reproduces
`models/vae/model.pt`. Everything is overridable — 41 environment variables in the runner, 33
`--flags` on the CLI beneath it:

```bash
KL_WEIGHT=0.01 HIDDEN_DROPOUT=0.2 EPOCH_CEILING=60 \
  bash Early_Stopping_Tests/scripts/tmux_train_multi.sh
```

| | |
|---|---|
| Architecture | `LATENT_DIM=16`, `HIDDEN_DIMS=256,128` |
| Optimisation | `BATCH_SIZE=1048576`, `LEARNING_RATE=1e-3`, `KL_WEIGHT=0.005` |
| Regularisation | `INPUT_DROPOUT=0.1`, `HIDDEN_DROPOUT=0.1` |
| Stopping | `EPOCH_CEILING=40`, `PATIENCE=8`, `MIN_DELTA=0.001`, `AU_THRESHOLD=0.01` |
| Data | `TRAIN_FRACTION=0.9`, `SPLIT_STRATEGY=global_site_hash`, `SEED=42` |
| Throughput | `EPOCH_SHARDS=20`, `SHUFFLE_BUFFER_ROWS=32768`, `DECODE_WORKERS=8`, `VAL_MAX_ROWS=5000000` |

`--parquet-paths` (globs) selects the interleaved trainer; it is *refused* alongside
`--parquet-path`, `--streaming`, `--use-all` or `--sample-rows` rather than silently ignored.

Verify before building on it:

```bash
python -c "
import json; s=json.load(open('<run>/summary.json'))['early_stopping']
print('best epoch', s['best_epoch'], '| val', round(s['best_val_total_loss'],4),
      '| active units', s['final_active_units'])"
```

Shipped model reads `best epoch 30 | val 0.2209 | active units 16`. Active units below 16
means latent dimensions collapsed — don't use that run.

---

## 3. Refitting UMAP (stages 3–4)

```bash
bash umap_hdbscan_sweep/run_parametric_sweep.sh
```

Sweeps `SIZES=2000000,5000000,10000000,25000000` × `NN=15,30,50` ×
`MIN_DIST=0.0,0.1,0.25` × `MODES=regress,umap,hybrid` × `SEEDS=3` replicates, reading
`EMBED_DIR` for `latent.npy`. Point `EMBED_DIR` at the new stage-1 output.

**You usually don't need the sweep.** It exists to *choose* the parameters, and that choice is
already made — `models/umap/final_models_README.md` records how the shipped encoder was
selected and against which controls. If you are rebuilding only because the latent changed,
fit the one configuration directly. Every grid variable is a comma-separated list, so a single
value each gives a one-cell grid:

```bash
SIZES=25000000 NN=15 MIN_DIST=0.1 MODES=umap SEEDS=1 \
  bash umap_hdbscan_sweep/run_parametric_sweep.sh
```

That still runs the trial, gate and cost-projection stages. To skip straight to the fit:

```bash
python umap_hdbscan_sweep/parametric_sweep.py \
    --embed-dir <new stage-1 output> --output-dir <out> \
    --fit-rows 25000000 --min-dist 0.1 --n-neighbors 15 --modes umap --seeds 1 \
    --gpu-budget-gb 40
```

> The shipped parameters were selected against the *old* latent space. Reusing them on a new
> one is the sensible default, not a guarantee they remain optimal — if the VAE changed
> materially, a reduced sweep (say `NN=15,30` × `MIN_DIST=0.1,0.25` at one size) is the cheap
> middle ground.

Then project the whole cohort to get a new `coords.npy`:

```bash
python umap_hdbscan_sweep/apply_parametric_full.py --help
```

> **Cluster counts are seed-sensitive** — 508–533 at 25 M and 142–184 at 5 M with nothing
> changing but the training seed. Treat any single count as approximate when comparing
> candidates.

---

## 4. Refitting HDBSCAN (stage 5)

```bash
tmux new-session -d -s final_models 'bash umap_hdbscan_sweep/tmux_final_models.sh'
```

**This is already a single cell** — it defaults to the selected configuration
(`FIT_SIZES=1000000`, `MIN_CLUSTER_SIZES=2500`, `MIN_SAMPLES=15`, `METHODS=eom`,
`EPSILONS=0.0`) for both backends (`BACKENDS=cuml,cpu`), with `DBCV_BACKEND=hdbscan` pinned so
the two are comparable. The full 24-cell sweep is opt-in via `FULL_GRID=1`. Use
`BACKENDS=cuml` to skip the CPU cross-check, which exists only for the diagnostics cuML does
not expose.

**After a retrain you must override `COORDS` and `CONTEXT`.** They default to the
August-2026 run's paths, so a bare invocation clusters the *old* embedding while appearing to
succeed:

```bash
COORDS=<new>/coords.npy CONTEXT=<new>/context.parquet \
  bash umap_hdbscan_sweep/tmux_final_models.sh
```

> **Always name the backend in the output directory, and check it afterwards.**
> `metrics.json` records both `cluster_backend_requested` and the `cluster_backend` that
> actually ran — compare them before reporting anything. A sweep once written to a directory
> called `param_sweep_refit_cpu` was in fact cuML, and "backends identical, ARI 1.0000" was
> published from a cuML-vs-cuML comparison. Both runners now default to `cuml` explicitly
> rather than resolving `auto` at import time, but the directory name is still just a name.

cuML HDBSCAN is **not bit-reproducible**: two fits on identical input with the same seed hold
the cluster count stable but move the noise boundary by ~0.3 pp. Refitting will not return the
shipped labels — use the saved `hdbscan_model.pkl` and `fit_indices.npy` instead.

Verify each cell with `verify_saved_models.py` before consuming it — see
[`RUNNING_INFERENCE.md` §5](RUNNING_INFERENCE.md).

---

