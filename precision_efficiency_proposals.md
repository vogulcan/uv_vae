# Precision / dtype efficiency proposals (separate pass)

Status: **proposed, not implemented.** These are intentionally kept *out* of the
shard cache (`uv_vae/shard_cache.py`) so that the cache is byte-for-byte
identical to the live streaming path today. Each item below is an independent,
reviewable change. Several also **shrink the on-disk cache**, which is the lever
that makes full-dataset caching feasible on limited SSD.

Context: the data pipeline currently stores categorical codes as `int64`,
numeric features as `float32`, and the numeric mask as `float32`. At the target
scale (95 samples × ~50M rows ≈ 4.75B rows) that is ~320 bytes/row, so a full
encoded cache is **~1.52 TB** — larger than node-local SSD (~50–100 GB) and a
big fraction of the shared pool (~1.3 TB). The changes here cut that to
~350–675 GB with no change to model quality (except #3, which carries a small,
bounded precision risk).

---

## Cache size as a function of dtype choices

Per-row bytes = `n_cat·cat_size + n_num·num_size + mask_size`, with
`n_cat = 11`, `n_num = 29`. Total = per-row × 4.75B rows.

| Configuration | cat | num | mask | B/row | Full cache (4.75B) | Per sample (50M) |
|---|---|---|---|---|---|---|
| **Current** (int64 / f32 / f32) | 88 | 116 | 116 | 320 | ~1.52 TB | ~16.0 GB |
| + int16 cat (#1) | 22 | 116 | 116 | 254 | ~1.21 TB | ~12.7 GB |
| + int16 cat + bit-packed mask (#1,#2) | 22 | 116 | 4 | 142 | ~675 GB | ~7.1 GB |
| + int16 + bitmask + fp16 num (#1,#2,#3) | 22 | 58 | 4 | 84 | ~399 GB | ~4.2 GB |
| + int8 cat + bitmask + fp16 num | 11 | 58 | 4 | 73 | ~347 GB | ~3.6 GB |

Takeaways:
- **#1 + #2 alone** (both exact, zero quality risk) roughly **halve** the cache
  to ~675 GB — fits the shared 1.3 TB pool with headroom.
- Adding **#3** (fp16 numeric, small risk) brings it under **~400 GB**.
- At the aggressive config, a *single sample* is ~3.6–4.2 GB, so a handful of
  samples fit even on a 50–100 GB node-local disk.

---

## Proposed changes

### #1 — `int64 → int16` categorical codes  ·  exact  ·  4× on cat
- **What:** store integer category codes as `int16` instead of `int64`. Max
  cardinality across all 11 categorical features is **16** (`REF`), so `int16`
  (range ±32,767) is comfortable; `int8` (range ±127) is even viable for a
  further 2×, though `int16` is the safer default.
- **Where:** every categorical code array —
  `uv_vae/preprocess.py:44`, `:107`, `:182`;
  `uv_vae/streaming.py:215`, `:245`, `:249`;
  `uv_vae/convergence.py:123`; `uv_vae/inference.py:206`.
  In `shard_cache.py`, update `_CAT_ITEMSIZE` and the manifest `cat_dtype`.
- **Constraint:** PyTorch `nn.Embedding` requires a `LongTensor` at
  forward time (`model.py:60`). Keep the `.long()` cast at the **batch
  boundary** (`streaming.py:281`, `preprocess.py:146-147`, `:201`) — storage and
  host→device transfer shrink 4×, and the per-batch upcast of a few hundred rows
  is negligible.
- **Risk:** none. Values are exact; only the container width changes.

### #2 — `float32 mask → bit-packed` (or `uint8`)  ·  exact  ·  up to 32×
- **What:** the numeric mask is strictly binary (observed / imputed). Storing it
  as `float32` costs 4 bytes/entry. Options: `uint8` (4×) or **bit-pack across
  the 29 features** into `ceil(29/8) = 4` bytes/row (**~29×** vs float32).
- **Where:** mask creation `preprocess.py:57`, `streaming.py:225`, `:230`,
  `:261`, `:265`; consumed only in the masked reconstruction loss
  (`training.py:87`, `streaming.py:446`, `:518`). Unpack/cast to `float` at the
  batch boundary just before the loss multiply. In `shard_cache.py`, this is the
  single biggest cache-size lever (116 → 4 B/row).
- **Risk:** none. Mask is exactly recoverable.

### #3 — `float32 → float16/bf16` numeric **cache storage**  ·  small risk  ·  2×
- **What:** store the *cached, normalised* numeric matrix in half precision.
  Numerics are z-score normalised (≈ unit variance), so **bf16** (8-bit
  exponent, same range as f32) preserves scale with ~3 significant digits;
  `float16` is smaller-range and riskier on distribution tails.
- **Where:** numeric matrix storage in `shard_cache.py` (cache only) and,
  optionally, the normalised output in `preprocess.py:110`/`streaming.py:227`.
  **Upcast to float32 at the batch boundary** for the loss; keep the model math
  in f32/AMP.
- **Risk:** low–moderate and *bounded* — this is a **storage** choice, not a
  compute one. Recommend bf16, and validate that val loss is unchanged vs the f32
  cache on one sample before adopting. Keep it opt-in (a `cache_dtype` flag).

### #4 — enable AMP in the **in-memory** training path  ·  low risk  ·  GPU compute + memory
- **What:** the streaming path already wraps the forward/loss in
  `torch.amp.autocast` (`streaming.py:440`), but the in-memory
  `run_epoch`/`compute_loss` (`training.py:110`, `:77`) run full fp32.
- **Where:** add an `autocast` + `GradScaler` around the in-memory training step,
  mirroring `_run_training_epoch`.
- **Risk:** low; standard mixed precision. Gives faster GPU steps and lower
  activation memory on CUDA, no effect on CPU.

### #5 — precompute `inv_std = 1/std`, multiply instead of divide  ·  exact  ·  micro
- **What:** normalisation does `(x - mean) / std` every chunk
  (`preprocess.py:142`, `:193`; `streaming.py:227`, `:262`). Precompute
  `inv_std` once and multiply.
- **Risk:** none (identical up to floating-point ULP). Marginal, but free.

### #6 — collapse the redundant `fill_null(NaN) → where(mask, x, mean)`  ·  exact  ·  one pass
- **What:** `encode_numeric_column` fills nulls with `NaN`
  (`preprocess.py:53`), then a later `np.where(mask>0, x, mean)`
  (`preprocess.py:141`, `streaming.py:226`) overwrites those NaNs with the mean —
  two passes over the array. When values are cached already-normalised, the
  imputation can be folded into a single pass.
- **Risk:** none if the mask is preserved (it is — it's a separate array / #2).

### #7 — cheaper batch-slice copies  ·  exact  ·  follows #1–#3
- **What:** `torch.from_numpy(cat[start:end].copy())` etc.
  (`streaming.py:281-283`) copy every slice. No code change needed, but the
  copies become 4×/29×/2× cheaper automatically once #1/#2/#3 land, and reduce
  host→device transfer volume.
- **Risk:** none.

---

## Recommended sequencing

1. **#1 (int16 cat)** and **#2 (bit-packed mask)** first — both exact, together
   they halve the cache (~1.52 TB → ~675 GB) and shrink host→device transfer.
   Update `shard_cache.py`'s `_CAT_ITEMSIZE` / mask handling and bump
   `_CACHE_VERSION` so old caches invalidate cleanly.
2. **#4 (AMP in-memory)** — independent, quick GPU win for the sample-based path.
3. **#5, #6** — free correctness-neutral cleanups, bundle with #1/#2.
4. **#3 (bf16 numeric cache)** last and behind a flag — validate val-loss parity
   on one sample before enabling; it takes the cache under ~400 GB.

None of these change the model architecture or the training result (except #3's
bounded storage-precision trade-off, which should be A/B-checked).
