# Production pipeline data processing and VAE input parity

This document describes the data processing that the production pipeline performs before the VAE is used, and it checks whether that data path matches the data the training pipeline feeds into the VAE.

## 1. Production pipeline overview

The production pipeline is implemented in [scripts/run_variant_cluster_pipeline.py](scripts/run_variant_cluster_pipeline.py). Its flow is:

1. Read the source parquet file.
2. Apply the requested row filter.
3. Deduplicate variants by the identity columns CHROM, POS, REF, and ALT.
4. Keep one row per variant using a ranking rule that prefers higher SNVQ, then QUAL, MAPQ, DP, RAW_VAF, and then stable ordering.
5. Optionally sample the deduplicated population.
6. Load a trained VAE checkpoint.
7. Encode the sampled deduplicated rows into latent embeddings using the checkpoint’s saved preprocessing configuration.
8. Run UMAP and HDBSCAN on those embeddings.

## 2. Production pipeline data processing steps

### 2.1 Row filtering

The production pipeline starts by applying a DuckDB row filter from the CLI argument `--row-filter`.

This filter is applied before:

- deduplication
- sampling
- embedding

So the production pipeline does not see the full parquet file; it sees only the rows that pass the filter.

### 2.2 Deduplication by variant identity

The production pipeline does not simply sample rows from parquet. It first creates a deduplicated variant table keyed by:

- CHROM
- POS
- REF
- ALT

The logic is in [scripts/run_variant_cluster_pipeline.py](scripts/run_variant_cluster_pipeline.py) inside `write_deduplicated_sample()`.

For each variant identity, the pipeline keeps the row with the highest ranking according to:

1. SNVQ
2. QUAL
3. MAPQ
4. DP
5. RAW_VAF
6. stable tie-break order

That means the production pipeline is operating on a different data set than the simple training-sample rows from the training pipeline.

### 2.3 Sampling after deduplication

After deduplication, the production pipeline may sample a subset of the deduplicated population.

This sampling is also done with DuckDB reservoir sampling and a repeatable seed, but it happens after deduplication rather than directly from the original filtered parquet rows.

### 2.4 Feature selection for the VAE

The production pipeline loads the checkpoint payload and uses the feature names recorded in the checkpoint’s feature report.

It builds a selected feature list from:

- active categorical features
- active numeric features
- passthrough columns for downstream analysis

This ensures the same feature schema used during training is used again during embedding.

## 3. How the VAE receives data in production

The VAE is used in production through [uv_vae/inference.py](uv_vae/inference.py) via `LatentInference.from_checkpoint()` and `embed_parquet()`.

The production pipeline does the following before the model sees the data:

1. Reads the sampled deduplicated parquet rows.
2. Selects the model feature columns.
3. Encodes categorical values with the same category mapping used during training.
4. Encodes numeric values as float32 and creates a missing-value mask.
5. Normalizes numeric values using the training-time means and standard deviations saved in the checkpoint.

This means the production path uses the same preprocessing logic for feature encoding and numeric normalization as the training path, but it does not perform the training split or validation split that the training pipeline uses.

## 4. How this compares to the training pipeline

The training pipeline is implemented in [uv_vae/training.py](uv_vae/training.py) and [uv_vae/preprocess.py](uv_vae/preprocess.py).

The training pipeline does the following:

1. Reads parquet rows after applying the row filter.
2. Samples rows from the filtered parquet source.
3. Builds categorical and numeric feature matrices.
4. Drops unusable or all-null features.
5. Splits rows into train and validation sets.
6. Computes numeric normalization statistics from the training subset.
7. Feeds the prepared tensors into the VAE during training.

## 5. Are the production pipeline rows the same as the training pipeline rows?

Short answer: no, not by default.

The production pipeline and the training pipeline use different row-selection stages:

- Training pipeline: filtered parquet rows -> direct reservoir sample
- Production pipeline: filtered parquet rows -> deduplicate by variant identity -> optional reservoir sample of the deduplicated set

Because the production pipeline first deduplicates variants and then samples from that deduplicated set, the rows reaching the VAE in production are not guaranteed to be the same rows used to train the VAE.

## 6. When would they be the same?

The production pipeline and the training pipeline would use the same data for the VAE only if all of the following are true:

1. The same parquet file is used.
2. The same row filter is used.
3. The same feature spec is used.
4. The same seed is used.
5. The production pipeline is configured to sample the same set of rows that the training pipeline sampled.
6. The production pipeline is not changing the rows through deduplication in a way that removes or reorders them differently.

In practice, that means a test harness that wants exact parity should either:

- reuse the same sampled rows that the training pipeline produced, or
- explicitly make the production pipeline use the same sample rows and ordering as the training pipeline.

## 7. Practical takeaway

The production pipeline does use the same VAE preprocessing recipe for feature encoding and normalization, but it does not automatically use the exact same rows that the training pipeline used to train the model.

So if your goal is to make test data identical to the eventual production data seen by the VAE, you must explicitly align:

- the row filter
- the sample rows
- the deduplication behavior
- the feature spec
- the checkpoint preprocessing statistics

That is the key requirement for parity.
