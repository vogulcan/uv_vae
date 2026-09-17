# Loading all 95 per-sample parquets into the VAE

How batches are drawn proportionally from every sample, and how the read-order
clustering in each file is destroyed before it reaches the model.

Implements the recommendation in [SAMPLING_STRATEGY.md](SAMPLING_STRATEGY.md)
§10 — strategy C, proportional interleave, plus epoch sharding. That document
remains the decision record for *why* the other six strategies were rejected;
this one covers what was built and what it measures.

Everything here is **additive**: no existing module was modified, so every prior
sweep result stays comparable. The only pre-existing file changed is the CLI,
which gained a flag.

---

## 1. The two problems

### Composition

95 files of unequal size. Concatenating them, or globbing them, gives 95
contiguous blocks — and the streaming shuffle buffer is 500 000 rows against a
~50M-row block, about 1% of one sample. Every batch for the first ~50M rows is
100% sample 1, then 100% sample 2. That is not a mildly unbalanced batch; it is
sequential fine-tuning on 95 datasets in turn, and the final latent geometry
would disproportionately reflect whichever samples were read last.

### Physical layout

Measured on the two production featuremaps present locally:

| | wt0-12-ppm0050 | wtR12-c1-ppm0018 |
|---|---|---|
| total rows | 192,810,508 | 203,091,722 |
| post-filter rows | 50,591,627 (26.2%) | 48,478,197 (23.9%) |
| row groups | 1,570 | 1,653 |
| avg rows / row group | 122,809 | 122,862 |
| out-of-order steps in a 3M-row probe | **0** | **0** |
| consecutive rows at the **same locus** | **88.1%** | **87.2%** |
| mean depth per site (3M probe) | 8.37 | 7.76 |

The files are perfectly sorted by `(CHROM, POS)`, and ~88% of adjacent rows are
reads of the *same* variant site. All structural correlation in the data is
local and is an artefact of read order.

A shuffle buffer cannot fix this at any affordable size, because a contiguous
500 000-row window *is* a narrow genomic window — shuffling inside it still
leaves every row in the batch drawn from the same small region. 95 readers ×
500 000 rows would also be ~10 GB of host RAM.

---

## 2. What was built

| module | role |
|---|---|
| `uv_vae/uv_vae/multi_parquet.py` | the interleaved reader — proportional draws, row-group shuffling, epoch sharding |
| `uv_vae/uv_vae/splitting.py` | content-hash train/val predicates (a positional split cannot survive interleaving) |
| `uv_vae/uv_vae/stats_cache.py` | per-file row counts and normalisation statistics, cached and combined analytically |
| `uv_vae/uv_vae/multi_streaming.py` | the trainer — imports its maths from `streaming.py`, adds nothing |
| `uv_vae/scripts/multi_parquet_stats.py` | stage-1 preflight: build the cache, print the plan |
| `Early_Stopping_Tests/scripts/tmux_train_multi.sh` | two-stage runner |

`streaming.py`, `training.py`, `early_stopping.py`, `data.py`, `preprocess.py`
and `model.py` are **untouched**. `multi_streaming.py` imports
`_run_training_epoch`, `run_val_epoch_with_diagnostics` and `_build_model` from
`streaming.py`, so a 95-file run is numerically comparable to a single-file
streaming run by construction rather than by inspection.

### Composition: proportional interleave

One reader per file. Each batch draws `k_i = w_i × batch_size` rows from file
*i*, where `w_i` is that file's share of the total post-filter rows.

Allocation uses **largest-remainder (Hamilton)** apportionment, not truncation.
Truncation floors any sample below `1/batch_size` of the pool to zero draws, so a
sample at 0.02% of the cohort would never appear in a batch at all.

### Layout: shuffle row-group *order*

Each file's ~1,570 row groups are independently addressable via
`ParquetFile.read_row_group`. Two levels of shuffling:

1. **row-group order**, reshuffled every epoch — consecutive reads come from
   unrelated genomic locations;
2. **within the decode buffer**, permuted on every refill — so a batch's
   contribution from one file is a random sample of that window rather than a run
   of adjacent reads at one site.

Level 2 is not optional. Without it, taking `k_i ≈ 86` rows sequentially out of
one decoded row group hands the model ~86 reads spanning ~10 adjacent loci.

Neither level costs extra I/O.

### Measured effect

On the real files, one 8,192-row batch:

| | sequential scan | interleaved reader |
|---|---|---|
| consecutive rows at the same locus | **81.3%** | **0.02%** |
| POS span | 792,533 – 1,382,375 | 26,730,536 – 130,844,920 (**177× wider**) |
| distinct sites in 8,192 rows | 1,534 | 5,012 (**3.3×**) |
| composition served vs target | — | 51.062% / 48.938% vs 51.067% / 48.933% |

---

## 3. Coverage: epoch sharding

A full pass over ~4.75B rows per epoch, for 10–20 epochs, is 10–20 passes over a
terabyte of parquet.

With `--epoch-shards E`, each file's row-group order is shuffled once per *cycle*
of E epochs, and epoch *e* takes the stride `order[e % E :: E]`. The E strides
are disjoint and their union is every row group, so:

- every row passes through the model **exactly once per cycle**,
- each epoch costs 1/E of a pass,
- each epoch is still a proportional mix of all 95 samples.

Seeding on the cycle rather than the epoch is what makes the strides disjoint.

`E = 20` with early stopping around epoch 15–20 makes the whole run cost roughly
what a *single* naive epoch would.

### Per-epoch rebalancing

A stride of row groups does not hold exactly 1/E of every file. Weighting draws
by whole-file counts therefore makes readers exhaust at different times, leaving
a tail of short batches that over-represents whoever still has rows.

Draws are recomputed each epoch from the rows actually available in that epoch's
row groups. Measured on the real files at a deliberately extreme
`epoch_shards=400` (only ~4 row groups per file per epoch):

| | before rebalancing | after |
|---|---|---|
| composition drift | 7.5 pp | **0.54 pp** |
| rows in short "ragged" batches | 14.8% of the epoch | **2.25%** |
| ragged batches | 19 of 70 | 3 of 65 |

Cumulative composition over a full cycle is exactly the true proportion, because
a cycle covers every row group once.

The reader warns when `epoch_shards` leaves a sample fewer than 20 row groups per
epoch. At 1,570 groups per file, `E = 20` gives ~78 and is comfortably clear.
`ragged_row_fraction` is recorded in every run's `training_report.json`.

---

## 4. The train/val split

`streaming.py` splits by a row's *position* in the scan
(`row_index % val_denominator == seed % val_denominator`). Interleaving 95 files
and reading row groups in shuffled order destroys any meaningful global scan
position, so that split cannot survive. It also admits only 10 distinct splits at
`train_fraction=0.9` (the remainder is `seed % 10`, so seeds 42, 52 and 62 give
the byte-identical validation set) and every script in the repo pins `SEED=42`.

`splitting.py` hashes a row *key* instead — order-independent, 2⁶⁴ distinct
splits, continuous `val_fraction`.

| strategy | key | cost | leakage |
|---|---|---|---|
| `per_sample_row_hash` | physical `(row_group, offset)`, salted per sample | free | **leaks at the locus level** |
| `per_sample_site_hash` | `(CHROM, POS, REF, ALT)`, salted per sample | reads `CHROM`, `POS` | **leaks across samples** |
| `global_site_hash` **(default)** | same key, unsalted | reads `CHROM`, `POS` | none — a locus lands on the same side in every sample |

The default is `global_site_hash`, matching the decision recorded for this
project. Early stopping watches validation ELBO, so a leaky validation set biases
that signal in one direction — it stops later than it should, contaminating the
epoch count, which is itself one of the axes this project measures.

*Row-level* hashing leaks within a sample: measured on the real data, **46.4% of
sites straddle train/val, covering 76.2% of rows**, and those rows share `REF`,
`ALT` and `X_PREV1..3`/`X_NEXT1..3` — 8 of the 11 categorical features. (The prior
session measured 44.8% / 75.3% on a different slice; same conclusion.)

*Per-sample site* hashing fixes that within a file but still leaks across the
cohort: all 95 are human genomes sharing loci, so a per-sample salt puts a site in
validation for sample 3 and in training for sample 60. Only the unsalted global
key is a clean locus holdout. It is also cheaper than the prior design's
`(CHROM,POS,REF,ALT,RN)` row key, which cost 2.8 s per 50.6M rows against 1.2 s
without `RN`.

Verified by the stage-1 probe on the real files at `val_fraction=0.1`: realised
validation row share 10.38% / 9.67%, site share 10.34% / 9.93%, and **0.000% of
sites straddling** the split.

Hashing is SplitMix64 with every constant pinned in `splitting.py`, not DuckDB's
`hash()` or polars' `hash_rows()` — neither guarantees stability across versions,
and a split that silently changes on a dependency upgrade would invalidate every
cross-run comparison. Verified against the published reference vectors.

**Salting folds into the hash input, never onto its output.** `hash(k,A) XOR
hash(k,B)` is a constant offset independent of `k`, so XOR-salting gives draws
that are structured across salts rather than independent — two salts can even
produce mutually exclusive splits. Measured overlap of two independent 10% draws
on the real files: 1.03% by sample id, 0.93% by seed, against the 1.00% that
independence predicts.

Note that with site-level hashing the split is uniform over **sites**; the *row*
fraction inherits variance from unequal site depth. On a 42,090-site slice the
row fraction landed at 9.77–9.94% against a predicted sd of 0.23 pp. Over 5.6M
sites per file that shrinks by ~11×.

---

## 5. Statistics

`streaming.py` makes three complete passes before the first gradient step —
`get_row_count`, `get_non_null_counts` and `compute_streaming_stats` are the same
scan with different aggregates — and writes the result into a freshly timestamped
run directory, so every run recomputes them.

`stats_cache.py` folds those into one query and caches per file, keyed on path +
size + mtime + row filter + feature fingerprint. Measured: **31.1 s cold, 0.004 s
warm** for the two production files. Storing per file means adding a 96th sample
costs one scan, not 96.

Files combine with the Chan–Golub–LeVeque pairwise update (1983, *Am. Stat.*
37(3):242–247) as a **balanced binary tree** rather than a left fold — the update
is least accurate when the two partitions differ greatly in size, which is what a
fold produces once it has accumulated a few files.

Two details matter as much as the formula:

- **each column carries its own `n`** — its non-null count, not the row count,
  because SQL aggregates skip NULLs, and 14 of the 29 numeric features are 100%
  null on these featuremaps;
- **the all-null drop uses counts summed across all files**, so a feature null in
  one sample but populated in another survives. Deriving the model shape from one
  file would build the wrong network.

Verified against a single DuckDB query over both files: max relative error
**8×10⁻¹⁴** on means, **6×10⁻¹⁴** on standard deviations.

Statistics are computed over all filtered rows, not training rows only — matching
`streaming.compute_streaming_stats` exactly, so runs stay comparable, and keeping
the cache independent of the split.

---

## 6. Validation cost

A site-keyed split scatters validation rows through every row group, so reading
"the validation set" means touching the whole dataset — at 10% of 4.75B rows that
is 475M rows per epoch, which would cost **more** than a sharded training epoch.

`--val-max-rows` (default 5,000,000) caps it by restricting each reader to a
subset of row groups, sized proportionally so validation composition still mirrors
training. The subset is shuffled — so it spreads across the genome rather than
being the first K groups, one contiguous region — and **fixed across epochs**,
because early stopping compares validation loss between epochs.

---

## 7. Performance

Measured on this Windows machine, single-threaded, 41 of 70 columns:

| stage | rate |
|---|---|
| `read_row_group` | 7.5M raw rows/s (16 ms/group) |
| filter (polars) | 16 ms/group |
| encode (categorical + numeric) | ~1.1M / 1.7M rows/s |
| **full path** | **~0.57M post-filter rows/s per reader** |

At that rate one complete pass over 4.75B rows is ~2.3 h of reader time; at
`E = 20` an epoch is ~7 min. The GPU node will differ — this is a floor, not a
forecast.

Filter strategies compared (all four agreed on the kept count):

| approach | ms/group |
|---|---|
| DuckDB over an in-memory Arrow table | 83 |
| `pyarrow.dataset` fragment + pushdown | 55 |
| `pyarrow.compute` mask | 26 |
| **polars** | **16** |

polars evaluates the `--row-filter` **SQL string directly** through
`polars.SQLContext` (0.09 s vs 0.07 s for a native expression), so the flag keeps
working unchanged and no SQL parser was needed.

The categorical encoder is a vectorised rewrite of
`preprocess.encode_categorical_column` (1.7× faster, `replace_strict` instead of
`np.fromiter` over a Python generator). `preprocess.py` is left alone so
single-file runs stay byte-identical; `test_encoder_matches_preprocess_helpers`
pins the two together.

### Memory

Decode is **strictly sequential**; buffers are concurrent. The GPU decode path
allocates from a 4 GB RMM pool inside the 16 GB per-process budget, and 95
concurrent decodes would want ~5.7 GB and spill. The batch loop pulls from one
reader at a time, so peak decode memory is one row group regardless of how many
files are open. Host buffers cost ~320 B/row × `--shuffle-buffer-rows` × 95 ≈
1 GB at the default 32,768.

No DuckDB connection is opened by the reader at all, which sidesteps the
"95 in-memory database instances, each with its own buffer manager and thread
pool" problem.

---

## 8. Usage

Stage 1 — build the cache and check the plan. **Do this first.** It confirms all
95 files parse, the row filter behaves, no sample is starved, and — by probing a
few random row groups per file — reports the *realised* validation row share,
site share and straddling-site fraction, which site hashing plus variable depth
make impossible to predict from `val_fraction` alone. The cache it writes is
reused by every later run:

```bash
STATS_ONLY=1 bash Early_Stopping_Tests/scripts/tmux_train_multi.sh
```

Stage 2 — train:

```bash
EPOCH_SHARDS=20 EPOCH_CEILING=40 PATIENCE=8 bash Early_Stopping_Tests/scripts/tmux_train_multi.sh
```

Or the CLI directly:

```bash
python "Early_Stopping_Tests/Python Files/train_with_early_stopping.py" --parquet-paths '/path/parquet_files/*.featuremap.parquet' --feature-spec-path uv_vae/ml_features.json --output-dir artifacts --epoch-shards 20 --epochs 40 --patience 8 --stats-cache-path ~/uv_vae/stats_cache.json
```

`--parquet-paths` implies streaming and is refused in combination with
`--streaming`, `--use-all` or `--sample-rows` rather than silently ignoring them.
Remember `sed -i 's/\r$//'` on the runner before its first use.

### Knobs

| flag / env | default | effect |
|---|---|---|
| `--epoch-shards` / `EPOCH_SHARDS` | 1 | epochs per full pass; 20 recommended |
| `--split-strategy` / `SPLIT_STRATEGY` | `global_site_hash` | see §4 |
| `--val-fraction` / `VAL_FRACTION` | `1 - train_fraction` | validation share |
| `--val-max-rows` / `VAL_MAX_ROWS` | 5,000,000 | cap on validation I/O per epoch |
| `--shuffle-buffer-rows` | 32,768 | per-reader buffer; ~320 B/row × 95 |
| `--stats-cache-path` / `STATS_CACHE` | none | reuse statistics across runs |

`VAL_FRACTION` is derived once in the runner and passed to *both* stages. Deriving
it per stage is a real bug: in float `1 - 0.9` is `0.09999999999999998`, which
yields a different 64-bit threshold than a literal `0.1` and therefore a genuinely
different partition — the statistics stage would describe one split and the
trainer would use another.

---

## 9. Artifacts

The run-artifact contract is unchanged: `model.pt`, `feature_report.json`,
`preprocess_report.json`, `training_report.json`, `diagnostics_report.json`,
`summary.json`, plus `convergence_report.json` when `--test-parquet-path` is
given. Verified by loading a produced checkpoint back through
`LatentInference.from_checkpoint` and encoding.

`training_report.json` and `summary.json` gain a `sampling` block: per-file row
counts, global and per-epoch interleave weights, per-file batch draws, rows served
per sample, row groups per epoch, ragged-tail fraction, split strategy and exact
threshold, validation row-group limits, and the stats cache path.

---

## 10. Test status

79 tests pass, including the 10 pre-existing ones (which is what demonstrates
`streaming.py` is untouched).

| suite | tests |
|---|---|
| `tests/test_splitting.py` | 22 |
| `tests/test_multi_parquet.py` | 28 |
| `tests/test_stats_cache.py` | 19 |
| pre-existing | 10 |

```bash
cd uv_vae && uv run pytest tests/
```

Verified end-to-end: 4 synthetic samples built from the real `ml_features.json`
trained through `multi_streaming` for 4 epochs (correct model shape — 11
categorical, 15 numeric, 14 all-null dropped — all artifacts written, checkpoint
reloaded through `LatentInference`); and the real CLI run against both production
parquets for 3 epochs, reaching 16/16 active units.

### What has **not** been verified

- **No GPU run.** All of the above ran on CPU torch 2.13. The AMP path, the cuDF
  GPU encode path, `gpu_budget.apply()` with a real RMM pool, and the batch-size
  ceiling have not executed.
- **Never run at 95 files** — only 2 real files and 4 synthetic ones. The
  95-reader open-file cost (~170 ms per handle → ~16 s of startup) and the peak
  host memory of 95 concurrent buffers are projections.
- **No full pass.** The longest real run was 3 epochs at `epoch_shards=400`,
  ~245k training rows per epoch.
- Throughput figures are from a Windows laptop, not from miletus.

First action on the GPU node: `cd uv_vae && uv run pytest tests/`, then stage 1
with `STATS_ONLY=1`.

---

## 11. Still open

The advisor questions in [SAMPLING_STRATEGY.md](SAMPLING_STRATEGY.md) §9 are
unchanged, and two of them bear directly on this code:

1. **Proportional or balanced weighting?** Proportional weighting treats
   sequencing depth as importance. If depth reflects library preparation rather
   than biology, this encodes a technical artefact as a statistical weight.
2. **Should validation hold out loci or whole samples?** Site-level hashing
   answers "does the latent space generalise to unseen loci?". Holding out whole
   samples answers "does it generalise to unseen individuals?" — arguably the
   more relevant question for the subsampling hypothesis, and it needs a
   file-level split, not a row predicate.
3. Strategy G (per-sample capped subsample) is still worth running alongside as
   the control the full-data result is measured against.
4. `stats_cache` is not wired into `streaming.py`; the single-file path still
   makes its three startup scans.
