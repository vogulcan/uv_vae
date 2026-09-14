# Accessing the UMAP coordinates and cluster labels

Every read carries two derived facts: where it landed in the UMAP projection, and which
HDBSCAN cluster it belongs to. This is how to read those back **alongside the original
columns of the source parquet**.

The three columns were never written into the source files. A DuckDB database of views joins
them on demand instead — it is kilobytes, and `SELECT *` returns the same result a merged
table would. [§5](#5-why-nothing-was-merged) covers the trade.

> **Per read, or per site?** This document is about the **per-read** view: 5.08 B reads across
> 95 samples, joined to their source parquets. It needs miletus, because the join reads those
> 569 GB in place. For **per-site** work — 157.5 M deduplicated loci — the arrays in
> `models/coords/` are already aligned and need no join, no database and no cluster access.
> See [§7b](#7b-cohort-level-data-needs-none-of-this).

---

## 1. Query it

```python
import duckdb

con = duckdb.connect("/home/patrick/pure-internship/umap_hdbscan_sweep/enriched.duckdb",
                     read_only=True)

con.execute("SELECT * FROM sample_index").df()                  # what's in here
con.execute("SELECT * FROM csb0_1_ppm0058 LIMIT 5").df()        # one sample
```

Keep `read_only=True`: these views point at the lab's shared parquets in
`/data/lab/ppmseq_parquets/`.

Each view returns every source column plus:

| Column | Type | Meaning |
|---|---|---|
| `file_row_number` | int64 | positional row id in the source — the join key |
| `umap_1`, `umap_2` | float32 | parametric UMAP coordinates |
| `cluster_label` | int32 | HDBSCAN cluster id; `-1` = noise, `NULL` = row didn't pass the filter |

**`NULL` and `-1` are different populations.** `NULL` never entered inference; `-1` was
clustered and came out noise. Conflating them folds ~79 % of the file into the noise cluster.
Use `WHERE cluster_label IS NOT NULL` for the analysed set. Measured on `csb0_1_ppm0058`:
255,292,378 source rows, 53,857,133 labelled (exactly the filter count), 4,948,253 of those
(9.19 %) noise.

`.df()` materialises into pandas — put a `LIMIT` or `WHERE` on exploratory queries, since one
view is ~255 M rows. Use `.fetch_record_batch(1_000_000)` to stream a large result instead.

---

## 2. What's in the database

| Object | What it is |
|---|---|
| 95 per-sample views | one per source parquet |
| `all_samples` | all 95 unioned, with a `sample` column in front |
| `sample_index` | view name ↔ sample name ↔ source path ↔ row counts |

View names drop `.featuremap` and turn `-` into `_`, because `csb0-1-ppm0058.featuremap` isn't
a legal SQL identifier. `sample_index.sample` keeps the original spelling:

```python
con.execute("SELECT view_name FROM sample_index WHERE sample = ?",
            ["csb0-1-ppm0058.featuremap"]).fetchone()[0]
```

---

## 3. Examples

Source and derived columns are peers in every query.

```python
# a cluster's mutation spectrum
"""SELECT REF || '>' || ALT AS substitution, count(*) AS n
   FROM csb0_1_ppm0058 WHERE cluster_label = 106
   GROUP BY 1 ORDER BY n DESC"""

# source columns aggregated by cluster
"""SELECT cluster_label, count(*) AS reads,
          round(avg(rq), 4) AS mean_rq, round(avg(RL), 1) AS mean_read_length
   FROM csb0_1_ppm0058 WHERE cluster_label IS NOT NULL
   GROUP BY 1 ORDER BY reads DESC LIMIT 10"""

# one cluster across the cohort
"""SELECT sample, count(*) AS reads,
          round(100.0 * count(*) / sum(count(*)) OVER (), 1) AS pct
   FROM all_samples WHERE cluster_label = 106 GROUP BY 1 ORDER BY reads DESC"""

# a locus to its cluster
"""SELECT sample, CHROM, POS, REF, ALT, cluster_label, umap_1, umap_2
   FROM all_samples WHERE CHROM = 'chr1' AND POS BETWEEN 17518000 AND 17519000
     AND cluster_label IS NOT NULL"""

# export without going through pandas
"""COPY (SELECT * FROM all_samples WHERE cluster_label = 106)
     TO 'cluster_106.parquet' (FORMAT PARQUET)"""
```

---

## 4. Performance

DuckDB pushes predicates into both parquets and skips row groups that can't match.

- **Per-sample views are the fast path.** `all_samples` spans ~24 B source rows (5.08 B
  labelled); an unfiltered aggregate over it reads everything.
- **`WHERE cluster_label IS NOT NULL` is a large win** — it discards ~79 % of rows.
- **Set these before anything cohort-wide.** The default spill directory is on the root
  partition (~49 GB free); `/data/lab` has 13 TB:

```python
con.execute("SET memory_limit='32GB'")
con.execute("SET threads=8")
con.execute("SET temp_directory='/data/lab/ppmseq_parquets/duckdb_tmp'")
```

---

## 5. Why nothing was merged

Parquet has no append-column operation — every tool that appears to add one rewrites the whole
file. Merging would have meant ~590 GB of new parquet duplicating 569 GB of sources so three
columns could ride along, and if done in place, overwriting lab-shared files owned by another
user with no way back.

The view stores the sentence describing the join instead of the rows it produces. Results are
identical.

What a materialised merge *would* add is independence from the source paths. If that's needed,
`export_assignments.py --attach` writes the enriched files and `/data/lab` has room — at the
cost of 590 GB and files that go stale when inference is re-run, against a database that stays
correct or fails loudly.

---

## 6. The one thing that breaks this silently

**`file_row_number` is positional.** DuckDB derives it from physical row order rather than
reading it from the file, so it identifies a read only while the source parquet is
byte-identical to the one that was labelled. A rewrite, re-sort or recompaction repoints every
id at a different read — the join still succeeds, and every number it returns is wrong.

Three guards, in order:

1. `per_parquet_inference.py` records `source_fingerprint.json` — a hash of the parquet
   footer, which is what any rewrite disturbs.
2. `export_assignments.py` re-checks all 95 when building the manifest.
3. `build_enriched_views.py` **refuses to build** over a drifted source.

Re-check before trusting a join made months earlier:

```bash
python umap_hdbscan_sweep/verify_source_fingerprint.py \
    --results-dir ~/pure-internship/umap_hdbscan_sweep/per_parquet_inference_cuml
```

If a source legitimately changed, re-run inference for that sample. Don't use `--allow-drift`
to get past the refusal.

---

## 7. Rebuilding

**Three inputs are required, and none of them are in `UV_VAE_Deployment/`:**

| Input | Where | Size |
|---|---|---|
| the 95 source parquets | `/data/lab/ppmseq_parquets/` | 569 GB, read in place |
| the 95 `row_assignments.parquet` | `per_parquet_inference_cuml/<sample>/` | 43 GB |
| `assignments_manifest.json` | `per_parquet_inference_cuml/` | derived from the above |

All three live on miletus, so the database cannot be rebuilt from the deployment folder alone.
That is a property of the join, not an oversight: it reads the sources in place, and they are
569 GB.

The database itself is disposable — kilobytes, seconds to rebuild:

```bash
python umap_hdbscan_sweep/build_enriched_views.py \
    --manifest ~/pure-internship/umap_hdbscan_sweep/per_parquet_inference_cuml/assignments_manifest.json \
    --database ~/pure-internship/umap_hdbscan_sweep/enriched.duckdb --verify 3
```

`--verify N` re-labels rows from N views and fails one whose join matched nothing — the quiet
failure a LEFT join otherwise hides behind an all-NULL result.

If the manifest is missing, regenerate it first (this also re-checks every fingerprint):

```bash
python umap_hdbscan_sweep/export_assignments.py \
    --results-dir ~/pure-internship/umap_hdbscan_sweep/per_parquet_inference_cuml
```

It embeds absolute paths, so copying the file elsewhere won't work. To build against copies on
another machine, rewrite the prefixes with `--source-root` / `--new-source-root` and
`--assignments-root` / `--new-assignments-root`.

---

## 7b. Cohort-level data needs none of this

`UV_VAE_Deployment/models/coords/` holds a **different** set of coordinates and labels — one
row per deduplicated locus rather than per read:

| File | Rows |
|---|---|
| `umap_coords_2d.npy`, `cohort_labels.npy`, `cohort_probabilities.npy`, `vae_latent_16d.npy`, `context.parquet` | 157,501,580 |

These are **positionally aligned** — row *i* of each describes the same locus — so they need no
join, no database and no cluster access:

```python
import numpy as np
xy  = np.load("models/coords/umap_coords_2d.npy", mmap_mode="r")
lab = np.load("models/coords/cohort_labels.npy",  mmap_mode="r")   # row i ↔ row i
```

The two answer different questions. The cohort arrays count each **site** once (157.5 M); the
views count each **read** (5.08 B), so a locus seen in 40 samples appears 40 times. Use the
cohort arrays for questions about sites, the views for questions about coverage or for
anything that has to reach the original parquet columns.

## 8. Provenance

Labels come from the cuML cell `fit1000000_mcs2500_ms15_eom` — **175 clusters, 7.36 % cohort
noise**. Parameters are in [`pipeline_parameters.md`](pipeline_parameters.md); files in
[`DEPLOYMENT_MANIFEST.md`](DEPLOYMENT_MANIFEST.md).

Cluster ids are **not comparable across backends** — the CPU refit of this same cell gives 181
clusters, and its cluster 106 is not this cluster 106. Everything here is cuML.

Per-sample noise doesn't equal the cohort's 7.36 %: that figure describes the 157.5 M-row
cohort embedding HDBSCAN was fit on, while each sample is labelled by prediction and lands
wherever its own reads fall. Measured values run around 9.2 %; `sample_index.noise_pct` has the
real figure per sample.
