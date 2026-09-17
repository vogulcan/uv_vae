# Pipeline Parameters Master Sheet

---

## 1. VAE — Tabular Variational Autoencoder

### Architecture

| Parameter | Value |
|---|---|
| Model type | VAE |
| Latent dimension | 16 |
| Encoder hidden dims | [256, 128] |
| Decoder hidden dims | [128, 256] (symmetric) |
| Categorical input | Learned embedding per feature, dim = min(16, 2·⌈card^0.5⌉) |
| Numeric input | z-score standardised; NULL → training mean (= 0 post-standardisation) |
| Reconstruction heads | Separate numeric MSE head + per-categorical cross-entropy head |

**Reconstruction heads explained:** The decoder has one output head per feature type. Numeric features (e.g. read quality, length) use MSE loss — the head predicts a single continuous value and is penalised by squared error. Categorical features (e.g. REF base A/C/G/T) use cross-entropy loss — the head predicts a probability distribution over the feature's vocabulary and is penalised by log-loss. Separate heads are required because numeric and categorical outputs need fundamentally different loss functions; a shared head cannot optimise both simultaneously.

### Training — Cohort Run (final model)

| Parameter | Value |
|---|---|
| Dataset | 95 per-sample parquet files, 5.08 billion rows post-filter |
| Row filter | `st = 'MIXED' AND et = 'MIXED' AND FILT = 1` |
| Epochs (requested / run / best) | 40 / 38 / **30** |
| Batch size | 1,048,576 rows |
| Learning rate | 0.001 (Adam) |
| KL weight (β) | 0.005 |
| Input dropout | 0.1 |
| Hidden dropout | 0.1 |
| patience | 8 |
| Train / val split | 90 / 10 (content-hash split on genomic site, seed 42) |
| Seed | 42 |
| Best val ELBO | 0.2209 |
| Final active units | 16 / 16 (no collapsed dimensions) |
| Hardware | NVIDIA RTX PRO 5000 Blackwell (48 GB) |
| Mixed precision | AMP (float16 forward, float32 gradients) |

**Training time:** Total wall-clock time 4 h 25 m 24 s (38 epochs on 5.08 B rows). Torch VRAM budget capped at 16 GB; projected per-batch allocation ~5.63 GB (batch size 1,048,576 × 5,765 bytes/row); peak VRAM not separately recorded.

### Early Stopping

Stopped when **both** conditions held for `patience = 8` consecutive epochs:
- Relative val ELBO improvement < 0.001
- Active unit count (Burda et al. 2016 criterion, threshold 0.01) stable

### Features Used

**Categorical (11):** REF, ALT, X_PREV1, X_NEXT1, X_PREV2, X_NEXT2, X_PREV3, X_NEXT3, tm, st, et

**Numeric (29):** X_HMER_REF, X_HMER_ALT, BCSQ, BCSQCSS, RL, INDEX, REV, SCST, SCED, SMQ_BEFORE, SMQ_AFTER, rq, sd, ed, l1–l7, q2–q6, EDIST, HAMDIST, HAMDIST_FILT

*(l1–l7, sd, ed, q2–q6 non-null in 63.8 M / 5.08 B rows, sourced exclusively from sample ddb0-2-ppm0028; no features were dropped — `dropped_all_null_features` and `dropped_sample_null_features` are both empty. All 5.08 B rows passed through training; NULLs in sparse features were imputed to the training mean, which equals 0.0 post-standardisation, with the mask tensor distinguishing real values from imputed ones.)*

---

## 2. UMAP — Parametric UMAP

### Selected Configuration (25M-row fit, applied to full cohort)

| Parameter | Value |
|---|---|
| Mode | Parametric UMAP (neural encoder, trained once, applied to all rows) |
| Fit rows | 25,000,000 |
| n_neighbors | 15 |
| min_dist | 0.1 |
| n_components | 2 |
| metric | Euclidean |
| n_epochs | 200 |
| spread | 1.0 |
| repulsion_strength | 1.0 |
| set_op_mix_ratio | 1.0 |
| Seed | 42 |
| Encoder fit time (25M rows) | 102 s |
| Full-cohort apply time (157.5M rows) | 22 s |
| GPU VRAM, apply stage | see §3 *GPU VRAM* — measured jointly with the VAE and HDBSCAN apply |

---

## 3. HDBSCAN — Cohort Clustering

### Selected Configuration (sweep cell `fit1000000_mcs2500_ms15_eom`)

| Parameter | Value |
|---|---|
| min_cluster_size (mcs) | 2,500 |
| min_samples (ms) | 15 |
| cluster_selection_method | eom (excess of mass) |
| cluster_selection_epsilon | 0.0 |
| Fit rows | 1,000,000 (random subsample of 2D UMAP coordinates, seed 42) |
| Input | 2D UMAP coordinates (157,501,580 × 2, from the rank-13 parametric encoder) |
| **Clustering backend** | **cuML (RAPIDS GPU)** — the production model |
| Clusters found | 175 |
| Noise fraction (cohort) | 7.36 % |
| Noise fraction (fit set) | 7.2375 % |
| Fit time | 18.7 s |
| Label time (full cohort, 157.5 M rows, RBC fast-predict) | 127.0 s |

### Cluster Quality (cuML)

| Metric | Value |
|---|---|
| DBCV (stratified, 400 pts/cluster, `hdbscan` backend) | 0.4074 |
| DBCV, per-cluster median / min / max | 0.4729 / −0.3186 / 0.9052 |
| Points in negative-DBCV clusters | 11.85 % |
| Clusters with negative DBCV | 10.86 % (19 / 175) |
| Relative validity (`relative_validity_`) | *not available* — cuML does not expose it |
| Connectivity (k = 10, mean / fraction pure) | 0.999964 / 0.999855 |
| Mean membership probability | 0.8284 |
| Fraction of points with probability > 0.8 | 0.7216 |
| Mean probability, held-out rows | 0.8339 |
| Cluster persistence (median) | 1.0000 — *degenerate, see note* |

> An earlier version of this sheet, and the sweep dashboard, report **0.4313** for this cell.
> That is the same partition scored with the `kdbcv` backend instead of `hdbscan`. The
> deployed run pins `DBCV_BACKEND=hdbscan`.

### Signature Fit — SigProfilerAssignment (`uv_only`, GRCh38, v3.5)

| Metric | Value |
|---|---|
| Total mutations in SBS96 matrix | 145,916,931 |
| Cosine similarity, mean / median | 0.2804 / 0.2390 |
| Cosine similarity, mutation-weighted mean | 0.2774 |
| Clusters with cosine > 0.7 | 11 (6.29 %), carrying 5.65 % of mutations |
| Clusters with cosine > 0.8 | 3 (1.71 %), carrying 1.39 % of mutations |
| L1 residual, mean % | 169.73 |
| Top-channel share, median | 0.6526 |
| Fraction of clusters with top-channel share > 0.5 | 0.6629 |
| Distinct dominant channels | 38 across 175 clusters |

### CPU Cross-Check (reference implementation, *not* the deployed model)

The same cell refit with CPU `hdbscan` 0.8.44 to obtain the diagnostics cuML does not expose.
These come from the `param_sweep_refit_cpu` run, not the sweep that produced the deployed
model — same parameters, but not measured alongside it:

| Metric | cuML (deployed) | CPU `hdbscan` |
|---|---|---|
| Clusters | 175 | 181 |
| Noise (cohort) | 7.36 % | 8.00 % |
| DBCV (`hdbscan` backend, both) | 0.4074 | 0.3674 |
| Relative validity | not exposed | 0.2160 |
| Points in negative-DBCV clusters | 11.85 % | 13.12 % |
| Mean membership probability | 0.8284 | 0.8105 |
| Cluster persistence (median / min / max) | 1.0 / 1.0 / 1.0 (degenerate) | 0.1086 / 0.0009 / 0.6069 |
| Fit / label time | 18.7 s / 127.0 s | 10.7 s / 120.1 s |

**Backend note.** cuML and CPU `hdbscan` are different implementations of the same algorithm,
so the same parameters give different partitions — across the 24-cell sweep grid they differ on
every cell. cuML is the deployed model and the source of every reported clustering and
signature metric. Persistence, GLOSH outlier scores and exemplars are available only from the
CPU refit, and describe its **181-cluster** partition, not the deployed 175-cluster one.

Full 24-cell comparison: `umap_hdbscan_sweep/hdbscan/hdbscan_sweep_comparison.xlsx`.

### GPU VRAM

Measured on the deployed hardware (RTX PRO 5000 Blackwell, 48 GB / 48,935 MiB), sampled every
2 s across the full 95-sample per-parquet run — 22,427 samples over 12 h 37 m:

| | VRAM |
|---|---|
| Peak, whole pipeline (VAE encode + UMAP apply + HDBSCAN label + SigProfiler) | **19.7 GB** (20,214 MiB) |
| Median while running | 7.2 GB (7,384 MiB) |
| Idle floor between samples | 0.09 GB |
| Budget requested (`GPU_BUDGET_GB`) | 44 GB |

The three stages share one process and one budget, so these are joint figures rather than
per-stage ones. Peak sits at **45 % of the requested budget**, so 24 GB is ample for inference
and 44 GB is generous headroom rather than a requirement.

Budgets are split by stage — `apply` gives RMM half (cuML) and torch half, `sweep` gives RMM
0.9 since no torch is loaded:

| Stage | Budget | Split |
|---|---|---|
| VAE training | 16 GB | all torch (RMM absent); ~5.63 GB projected per 1,048,576-row batch |
| Per-parquet inference (`apply`) | 44 GB | 22 GB RMM + 22 GB torch |
| Clustering sweep (`sweep`) | 40 GB | 36 GB RMM + 4 GB torch |

**HDBSCAN fit VRAM is not directly measured** — only the budget it ran under. The binding
constraint is known from failures rather than instrumentation: at 25 M fit rows, `ms=15` OOMs
on this card while `ms=5` completes, which is why `--max-ms-at-25m 5` exists. The deployed cell
fits 1 M rows and is nowhere near that ceiling.

### Saved Artefacts

| Artefact | cuML | CPU |
|---|---|---|
| `cohort_labels.npy`, `cohort_probabilities.npy` | ✓ | ✓ |
| `cluster_persistence.npy` | ✓ (degenerate) | ✓ |
| `outlier_scores.npy` | — | ✓ |
| `exemplars.npz` | — | ✓ |
| `hdbscan_model.pkl` | ✓ | ✓ |
| `fit_indices.npy` | ✓ | ✓ |
| SigProfiler `uv_only` assignment | ✓ | — |