# Running inference

Applying the shipped checkpoints to new parquet files:
**parquet → filter → VAE encode → parametric UMAP → HDBSCAN label → SBS96 → SigProfiler
`uv_only`**, then joining coordinates and labels back onto the source rows.

Nothing here retrains anything — every model is loaded from `models/` and applied, so results
stay comparable to the cohort run and to each other. To rebuild the models themselves, see
[`REBUILDING_MODELS.md`](REBUILDING_MODELS.md).

| | |
|---|---|
| Parameters behind these models | [`pipeline_parameters.md`](pipeline_parameters.md) |
| What each file is | [`DEPLOYMENT_MANIFEST.md`](DEPLOYMENT_MANIFEST.md) |
| Querying the results | [`ACCESSING_ASSIGNMENTS.md`](ACCESSING_ASSIGNMENTS.md) |
| Retraining from scratch | [`REBUILDING_MODELS.md`](REBUILDING_MODELS.md) |

---

## 1. Environment

**cuML is mandatory.** `hdbscan_model.pkl` is a `cuml.cluster.hdbscan.HDBSCAN`; a CPU-only
machine can read every `.npy` and `.parquet` here but cannot load it. Everything else falls
back to CPU.

On miletus: `micromamba activate uv_vae`. Elsewhere, note that **`uv sync` alone is not
enough** — `cuml`, `cupy` and `rmm` are imported by the pipeline but install from the
`rapidsai` conda channel, not PyPI, so they're absent from `pyproject.toml`:

```bash
micromamba create -n uv_vae -c rapidsai -c conda-forge -c nvidia \
    cuml cupy rmm python=3.12 cuda-version=12.8
micromamba activate uv_vae
cd uv_vae && uv sync --active
```

Everything else (`torch`, `numpy`, `polars`, `duckdb`, `pyarrow`, `scikit-learn`, `hdbscan`,
`umap-learn`, `joblib`, `matplotlib`, `pillow`, `sigprofilerassignment`, `tqdm`) comes from
`pyproject.toml`. The TensorFlow banner in the logs comes from inside `sigprofilerassignment`
— no pipeline file imports it.

Confirm the stack before a long run:

```bash
python -c "
import cuml, cupy, rmm, torch, joblib
print('cuml', cuml.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())
joblib.load('models/hdbscan/hdbscan_model.pkl'); print('model unpickles OK')"
```

**Input** must carry the feature columns in `uv_vae/ml_features.json` plus `CHROM`, `POS`,
`REF`, `ALT`, `st`, `et`, `FILT`. The row filter is fixed at
`st = 'MIXED' AND et = 'MIXED' AND FILT = 1` and deliberately not a flag — the VAE's
normalisation statistics were computed over exactly that population. Survival varies by sample — **21-28 %** across the two measured (53.9 M of 255.3 M for
csb0-1-ppm0058; 14.4 M of 51.3 M for xpcR2-b2-ppm0016). Source sizes differ widely too, so
size the output by the labelled count in the manifest rather than by a fixed fraction.

---

## 2. Run it

```bash
tmux new-session -d -s inference 'bash umap_hdbscan_sweep/tmux_per_parquet_inference.sh'
tmux attach -t inference
```

The runner has the deployment paths filled in. Override with `PARQUET_GLOB`, `OUTPUT_DIR`,
`MODEL_DIR`, `GPU_BUDGET_GB`; `SKIP_DONE=1` resumes without redoing finished samples.

**Check the header before walking away**: the model path, and that the cohort cluster count
reads **175**. A run pointed at the wrong cell is only obvious here.

Cost: ~15 min/sample, so ~24 h for 95. Peak VRAM 19.7 GB against the 44 GB default budget.

<details>
<summary>Direct CLI form</summary>

```bash
DEPLOY=~/pure-internship/UV_VAE_Deployment

python umap_hdbscan_sweep/per_parquet_inference.py \
    --parquet-glob   '/path/to/new_samples/*.parquet' \
    --checkpoint     $DEPLOY/models/vae/model.pt \
    --feature-spec   $DEPLOY/uv_vae/ml_features.json \
    --umap-model     $DEPLOY/models/umap/13_BEST_25M_nn15_md0.1_umap.pt \
    --coords         $DEPLOY/models/coords/umap_coords_2d.npy \
    --context        $DEPLOY/models/coords/context.parquet \
    --hdbscan-model  $DEPLOY/models/hdbscan/hdbscan_model.pkl \
    --fit-indices    $DEPLOY/models/hdbscan/fit_indices.npy \
    --output-dir     /path/to/output \
    --mcs 2500 --ms 15 --epsilon 0.0 --fit-rows 1000000 --seed 42 \
    --genome-build GRCh38 --cosmic-version 3.5 \
    --n-workers 4 --sigprofiler-cpu 4 \
    --gpu-budget-gb 44 --predict-batch-rows 5000000 --device auto
```
</details>

**One trap:**

- **`fit_indices.npy` is load-bearing.** `fast_predict` indexes the fit set *positionally*.
  The wrong index file — or one re-derived from a mismatched `(seed, fit_rows)` — labels every
  read against the wrong neighbours. No error; just wrong.

`--mcs`, `--ms`, `--fit-rows` and `--seed` are also ignored when a model is supplied. They
apply only if you omit `--hdbscan-model` and let the script fit its own, which produces a
partition whose cluster ids mean nothing next to the cohort's.

---

## 3. Outputs

```
<output-dir>/<sample>/
├── row_assignments.parquet     ← the joinable artefact
├── labels.npy, umap_coords.npy ← redundant with the above; safe to delete
├── source_fingerprint.json
├── sigprofiler_input/cluster_sbs96_matrix.tsv
├── sigprofilerassignment_uv_only_grch38_v3.5/
└── plots/  umap_{cluster,cosine,sigprofiler,substitution}.png
```

`row_assignments.parquet` holds one row per *surviving* read: `file_row_number` (int64),
`umap_1`, `umap_2` (float32), `cluster_label` (int32, `-1` = noise).

The two `.npy` files are exactly redundant with columns of that parquet — verified identical,
zero differences — and are ~60 % of the output size. Nothing downstream reads them.

---

## 4. Attaching results to the sources

**First, check the sources haven't drifted:**

```bash
python umap_hdbscan_sweep/export_assignments.py --results-dir /path/to/output
```

Writes `assignments_manifest.csv`/`.json` and re-checks every source fingerprint. **Non-zero
exit means something drifted** — don't join past that.

**Then join at query time** (recommended — no data copied):

```bash
python umap_hdbscan_sweep/build_enriched_views.py \
    --manifest /path/to/output/assignments_manifest.json \
    --database /path/to/enriched.duckdb --verify 3
```

This builds a DuckDB database of views where `SELECT *` returns every source column plus the
three new ones. Full query guide: [`ACCESSING_ASSIGNMENTS.md`](ACCESSING_ASSIGNMENTS.md).

<details>
<summary>Or materialise standalone parquets (~590 GB for 95 samples)</summary>

```bash
python umap_hdbscan_sweep/export_assignments.py \
    --results-dir /path/to/output --attach --out-dir /path/to/enriched \
    --threads 8 --memory-limit 32GB --temp-dir /data/lab/duckdb_tmp --skip-existing
```

Writes `<sample>.enriched.parquet` = every source column plus the three. `--filtered-only`
gives an INNER join instead of LEFT. This duplicates the entire source dataset; the view above
gives identical query results for kilobytes.

Check each sample reports `status: ok` — the script counts non-null labels and fails a sample
unless that equals the assignment row count, which is what catches a LEFT join that matched
nothing and wrote an all-NULL file.
</details>

The join key `file_row_number` is **positional** and only valid while each source stays
byte-identical. See [`ACCESSING_ASSIGNMENTS.md` §6](ACCESSING_ASSIGNMENTS.md) for the guards
and the re-check command.

---

## 5. Verifying models before a long run

```bash
python umap_hdbscan_sweep/verify_saved_models.py \
    --cell-dir <cell directory> --expect-complete --backend sklearn
```

Three checks from disk alone: the files exist and the pickle loads; `clusterer.labels_`
matches `cohort_labels.npy` at `fit_indices`; and held rows re-label to the values the sweep
recorded. Only the third exercises the reloaded model end to end.

The shipped model passes all three — 200,000/200,000 probe rows reproduced.

---

## 6. Latent space only

To encode without projecting or clustering:

```python
import sys, duckdb
sys.path.insert(0, "<deploy>/uv_vae")
from uv_vae.inference import LatentInference

inf = LatentInference.from_checkpoint(
    "<deploy>/models/vae/model.pt",
    feature_spec_path="<deploy>/uv_vae/ml_features.json", device="cuda")

frame = duckdb.connect().execute("""
    SELECT * FROM read_parquet('new_sample.parquet')
    WHERE st = 'MIXED' AND et = 'MIXED' AND FILT = 1""").pl()

mu = inf.encode_frame(frame, batch_size=4096)     # (n, 16) float32
```

`encode_frame` takes a **polars** DataFrame and `batch_size` is required — DuckDB's `.pl()`
returns the right type. Apply the same row filter the pipeline uses; the normalisation
statistics were computed over that population, and encoding a different one produces a latent
space the UMAP encoder was never fitted against.
