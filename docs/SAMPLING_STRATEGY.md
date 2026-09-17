# Training the VAE on all 95 per-sample parquets

> **Superseded in part.** The strategy selected here (proportional interleave + row-group shuffling) has been implemented and tested. See `MULTI_PARQUET_LOADING.md` for what was actually built, measured, and what remains unverified.

**Status of the numbers below.** Figures marked *(verified)* were read directly out of
the code in this repository. Figures marked *(estimated)* are arithmetic from those
constants and have **not** been measured against the real 95-file dataset — in
particular no throughput measurement exists yet, so every wall-clock statement is a
formula, not a result. Nothing in this document has been implemented; it is a menu of
options with trade-offs.

Prepared for review before committing to an approach.

---

## 1. The problem

We now hold 95 per-sample parquet feature maps, roughly 50M rows each, so on the order
of **4.75 billion rows** before filtering. Sample sizes are unequal. The default row
filter (`st = 'MIXED' AND et = 'MIXED' AND FILT = 1`) removes an unknown fraction —
**measuring the post-filter row count per file is the first thing that has to happen**,
because it sets every other number here.

Two requirements have been stated:

1. **All data must pass through the model** — no sample silently excluded.
2. **Batches must be representative** — a batch should not be drawn from one sample.

And one hard constraint: **a 16 GB GPU memory ceiling** per process, enforced by
`uv_vae/uv_vae/gpu_budget.py` on a 48 GB card shared with other work.

---

## 2. The GPU ceiling is not the binding constraint

This is worth establishing first, because it redirects the whole design.

The model is small: 11 categorical features (embedding dims 6–8) plus ~15 surviving
numeric features, hidden layers 256/128, latent dimension 16.

| quantity | value | source |
|---|---|---|
| GPU budget per process | 16 GB | `UV_VAE_GPU_MEM_GB` default *(verified)* |
| ├─ RMM pool (cuDF/cuML) | 4.0 GB | `DEFAULT_RMM_SHARE = 0.25` *(verified)* |
| └─ torch allocator | 12.0 GB | remainder *(verified)* |
| usable for one batch | 7.2 GB | `DEFAULT_BATCH_HEADROOM = 0.60` of torch's share *(verified)* |
| GPU cost per row, training | ~6 KB | `gpu_budget.bytes_per_row`, AMP *(verified)* |
| **largest batch that fits** | **~1.29M rows** | 7.2 GB ÷ 6 KB *(estimated)* |
| batch of 8,192 | ~49 MB | *(estimated)* |
| model + gradients + AdamW state | ~2.4 MB | ~150k parameters × 4 *(estimated)* |

A batch of 8,192 rows uses about **0.4% of the budget**. The dataset being 4.75B rows
instead of 5M does not change this at all, because a batch is a batch regardless of how
many batches follow it.

What *does* scale with dataset size:

- **Host RAM** — loading everything into memory would need ~208 B/row × 4.75e9 ≈
  **990 GB** *(estimated)*. This is the "out of memory" that would actually occur, and
  it is why the streaming trainer exists.
- **Wall clock** — one pass over 4.75B rows, dominated by parquet scan and per-chunk
  encoding, not by GPU compute. Prior sweeps on this pipeline were found to be I/O and
  CPU bound.
- **Batch composition** — the subject of the rest of this document.

---

## 3. Two independent questions

"Representative batches" and "all data seen" are often conflated. They are separable:

- **Composition** — *which samples appear in a single batch, and in what ratio?*
- **Coverage** — *over what span of training is every row used at least once?*

Any composition strategy (§4) can be combined with any coverage strategy (§5).

---

## 4. Composition strategies

### A. Concatenate all 95 into one parquet

Use the existing `uv_vae/scripts/combine_parquets.py`, then the existing streaming
trainer unchanged.

- Rows are copied **in input order** *(verified)*, so the merged file is 95 contiguous
  sample blocks.
- The streaming shuffle buffer is 500,000 rows *(verified)* against a ~50M-row block —
  about 1% of one sample. Every batch for the first ~50M rows is 100% sample 1, then
  100% sample 2, and so on.
- This is not "slightly unbalanced batches"; it is **sequential fine-tuning on 95
  datasets in turn**. The final latent geometry would disproportionately reflect
  whichever samples were read last.
- Output would be ~1–1.5 TB on a ~1.3 TB shared SSD.

**Verdict: reject.** No shuffle-buffer size fixes it short of buffering the dataset.

### B. Point the existing trainer at a glob of all 95

- Identical ordering problem to A, since DuckDB reads globbed files in sequence and the
  streaming dataset pins `preserve_insertion_order=true` *(verified)* — which it must,
  because the train/val split is defined by scan position.
- Additionally breaks on contact: `stream_parquet_batches` calls
  `pyarrow.parquet.read_schema(parquet_path)`, which fails on a glob, and
  `get_row_count` without a filter reads `parquet_file_metadata` and takes only the
  first row — i.e. one file's count *(both verified)*.

**Verdict: reject.**

### C. Proportional interleave *(recommended)*

Open one reader per file. Each batch draws from every reader, weighted by that file's
post-filter row count:

```
w_i = n_i / Σn          k_i = round(w_i × batch_size)
```

- **Every batch contains all 95 samples in their true proportions.** Even a sample at
  0.2% of the pool contributes ~17 rows to an 8,192-row batch.
- **All readers exhaust simultaneously**, because draw rate is proportional to size.
  One pass = every row used exactly once, with no truncation and no repetition.
- Batch composition matches the pooled population, so the VAE's implied prior over
  samples is the empirical one.

*Caveat worth discussing:* proportional weighting means a sample with twice the
sequencing depth gets twice the influence. If depth reflects library preparation rather
than biology, this encodes a technical artefact as a statistical weight. See open
question 1.

### D. Balanced interleave (equal rows per sample)

Same machinery, but `k_i = batch_size / 95` for every file.

- Each sample contributes equally regardless of size — the right choice if the question
  is a per-sample comparison.
- **But coverage breaks.** Small files exhaust long before large ones, so they must be
  cycled and re-read. A full pass over the largest file implies an epoch of
  `95 × max(n_i)` rows, which is *larger than the dataset*, and small samples get
  repeated many times within a single epoch — a real overfitting risk on exactly the
  samples with the least data.

### E. Shuffled row-group plan

Read parquet row-group metadata for all 95 files, build the list of `(file, row_group)`
pairs, shuffle it under the epoch seed, and read groups in that order into a large
shuffle buffer.

- Cheaper: a handful of concurrent readers instead of 95.
- Each batch mixes however many row groups the buffer spans — roughly 2–20 samples, not
  95. Unbiased in expectation across an epoch, but **noisy batch to batch**.
- Coverage is exact (every row group read once per pass).

**Verdict: good fallback** if 95 concurrent readers prove too heavy in practice.

### F. One-time global shuffle to disk

Rewrite all 95 files once into N globally pre-shuffled shards, then read shards
sequentially thereafter.

- Fastest training loop by a wide margin, and trivially correct composition.
- Requires writing ~1–1.5 TB to a ~1.3 TB shared SSD, plus hours of one-time I/O.

**Verdict: does not fit current storage.** Reconsider only if a dedicated volume appears.

### G. Per-sample capped subsample *(the control arm)*

Reservoir-sample a fixed budget per file (e.g. 5M rows × 95 = 475M rows), then train
normally.

- Does **not** satisfy "all data seen", so it does not meet the stated requirement.
- But prior stability sweeps in this project found the latent geometry stabilises around
  ~5M rows (seed CV% ≈ 0.19%), and that additional data did not improve latent geometry
  monotonically.
- Cheap, fully reproducible, and it is the honest baseline the full-data run must beat.

**Verdict: run this regardless, as the comparison the full-data result is measured against.**

---

## 5. Coverage: "all data seen" ≠ "all data every epoch"

A full pass over 4.75B rows per epoch, for 10–20 epochs, is 10–20 complete passes over
~a terabyte of parquet.

The alternative: partition each file's rows into `E` disjoint strata and give epoch *e*
stratum *e*. Then

- every row passes through the model **exactly once across the run**,
- each epoch costs 1/E of a pass,
- each epoch is still a proportional mix of all 95 samples,
- and `E` epochs constitute one complete pass over the data.

With `E = 20` and early stopping around 15–20 epochs, the whole run costs roughly what a
*single* naive epoch would. This combines with strategy C, D, or E.

**Interaction to get right:** the current validation split is
`row_index % val_denominator == seed % val_denominator` with `val_denominator = 10` at
`train_fraction = 0.9` *(verified)*. If `E` shares a factor with 10, whole epoch strata
land entirely in train or entirely in val. Assign validation **first**, then stratify
only the surviving training rows with an independent counter.

---

## 6. Comparison

| | composition per batch | coverage per pass | extra disk | main risk |
|---|---|---|---|---|
| **A** concatenate | 1 sample | exact | ~1–1.5 TB | sequential fine-tuning; does not fit disk |
| **B** glob | 1 sample | exact | none | same, plus two code paths break |
| **C** proportional interleave | all 95, true ratios | exact | none | 95 concurrent readers |
| **D** balanced interleave | all 95, equal | over-covers large, repeats small | none | overfits small samples |
| **E** row-group shuffle | ~2–20 samples, varies | exact | none | noisy batch composition |
| **F** pre-shuffled shards | all 95 | exact | ~1–1.5 TB | exceeds available storage |
| **G** capped subsample | all 95, equal | **partial by design** | small | does not meet the requirement |

---

## 7. Resource budget for strategy C

The one way strategy C *could* breach the 16 GB ceiling is if all 95 readers decode
concurrently. Per-chunk GPU decoding goes through cuDF, which allocates from the 4 GB
RMM pool; 95 simultaneous chunks at the current default chunk size would be
`95 × 100,000 × ~600 B ≈ 5.7 GB` *(estimated)* — over the pool. It would not crash
(`CUDF_SPILL=1` is set *(verified)*) but it would spill to host and thrash.

**The fix is a design constraint: decode sequentially, buffer concurrently.** One reader
decodes one chunk on the GPU, the result is copied straight to host memory, the GPU
allocation is released, and the next reader proceeds. Peak GPU decode memory is then one
chunk, not 95.

| knob | proposed value | cost *(estimated)* |
|---|---|---|
| read chunk | 16,384 rows | ~10 MB RMM (one decode in flight) |
| per-file shuffle buffer | 32,768 rows | 95 × 32,768 × 208 B ≈ **650 MB host** |
| batch size | 8,192–32,768 | 49–197 MB torch, against 7.2 GB allowed |
| DuckDB | **1 instance + 95 cursors** | see below |
| GPU budget | 16 GB, unchanged | peak ~0.5 GB actually used |

**DuckDB caveat.** `connect_duckdb` calls `duckdb.connect(":memory:")` *(verified)*,
which creates a *separate database instance* per call — each with its own buffer manager,
a default memory limit near 80% of system RAM, and its own thread pool. Opening 95 of
those over-commits memory and oversubscribes threads. One instance with 95 cursors shares
a single buffer pool and thread pool; the memory limit should then be set explicitly.

---

## 8. A related issue: the validation split

Independent of the sampling strategy, the current split has properties worth flagging,
because it affects how any of the above is evaluated.

`row_index % val_denominator == seed % val_denominator` *(verified)*:

1. **Only 10 distinct splits exist** at `train_fraction = 0.9`, since the remainder is
   `seed % 10`. Seeds 42, 52 and 62 give the byte-identical validation set. Every script
   in the repository pins `SEED=42`.
2. **It leaks at the locus level.** The feature map holds single-read SNVs, so many rows
   share one `CHROM/POS/REF/ALT` site — the repository already treats that as the row key
   and has to deduplicate on it elsewhere *(verified)*. Taking every 10th row splits reads
   from the *same locus* across train and val: a site with 30 reads puts ~27 in train and
   ~3 in val. Validation loss is therefore measured on loci the model has already seen,
   which biases it optimistically — and since early stopping watches validation ELBO, it
   **stops later than it should**. This is a directional bias, not noise.
3. **It depends on scan order**, which is why `preserve_insertion_order` must be pinned,
   and which any multi-file interleave would disturb.

**Proposed replacement:** assign the split by hashing the site key rather than the row
position.

```sql
-- val when the site key hashes into the bottom `val_fraction` of the 64-bit range
WHERE hash(CHROM, POS, REF, ALT, {seed}) < {int(val_fraction * 2**64)}
```

- Order-independent, so it survives interleaved multi-file reading unchanged.
- All reads at a locus land on the same side — a genuine held-out set.
- 2⁶⁴ distinct splits instead of 10, so split sensitivity becomes measurable.
- `train_fraction` becomes continuous rather than quantised to `1/round(1/(1-f))`.

Note that hash *collisions* are harmless here: this is a partition, not a lookup, so two
distinct sites landing on the same value simply share a split. Only output uniformity
matters. Costs: `CHROM` and `POS` are not in `ml_features.json`, so they add ~20–30 B/row
of read I/O; and DuckDB's `hash()` is not guaranteed stable across major versions, so the
version should be pinned and recorded.

---

## 9. Questions for the advisor

1. **Proportional or balanced weighting (C vs D)?** Does a sample having more rows
   reflect something real, or just sequencing depth and library preparation? Proportional
   weighting treats depth as importance.

2. **What should the validation set hold out — loci or samples?** Site-level hashing
   answers *"does the latent space generalise to unseen loci?"*. Holding out whole samples
   answers *"does it generalise to unseen individuals?"*. These are different scientific
   claims and imply different splits. The second is arguably the more relevant question
   for the subsampling hypothesis this project is testing.

3. **Is a full pass over 4.75B rows scientifically necessary?** This project's own sweeps
   found latent geometry stabilising near 5M rows, and that more data did not improve it
   monotonically. If the claim to be defended is *"the full data was used"*, strategy C +
   epoch sharding delivers it. If the claim is *"more data would not have changed the
   answer"*, strategy G plus a stability curve may be the stronger evidence, at a small
   fraction of the cost. These are not mutually exclusive — G is cheap enough to run
   either way.

4. **Should sample identity be a model input?** It is not currently in
   `ml_features.json` *(verified)*, so the VAE cannot distinguish samples and any
   per-sample batch effect is unmodelled — it will surface as latent structure without an
   attached label. Worth deciding deliberately rather than by omission.

---

## 10. Recommendation

**Strategy C (proportional interleave) with sequential decode, plus epoch sharding at
E ≈ 20**, and **strategy G (per-sample capped subsample) run alongside as the control**.

Two supporting changes are worth making first, since both improve existing single-file
runs and are independent of the 95-file work:

- **Fold the three startup scans into one, and cache the result.** The streaming trainer
  currently makes three complete passes before the first gradient step — row count,
  non-null counts, and normalisation statistics are three separate queries over the same
  scan *(verified)*. They fit in one query. Because the statistics are additive across
  files, per-file caching means adding a 96th sample later costs one scan rather than 96.
- **Move the validation split off row position** (§8), before the large run rather than
  during it, so results before and after remain comparable.
