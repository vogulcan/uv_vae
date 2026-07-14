# VAE Stability Test

This file was created for the requested VAE stability test notes.

## Evaluation harness for VAE subsampling

The repository now includes a lightweight evaluation workflow for comparing how well a VAE preserves latent structure when trained on smaller subsamples of the data. The implementation lives in [uv_vae/evaluation.py](uv_vae/evaluation.py), and the command-line entry point is in [uv_vae/cli.py](uv_vae/cli.py).

### Workflow overview

1. Sample one training subset for each requested fraction (for example 5%, 10%, 25%, 50%, 75%, 100%).
2. Train a VAE on each subset using the existing uv_vae preprocessing and training utilities.
3. Encode a fixed held-out reference set with each trained model.
4. Compare the resulting latent spaces with Procrustes distance, CKA similarity, k-NN Jaccard overlap, trustworthiness, continuity, and optional clustering metrics.
5. Aggregate the results into a table and optionally save a CSV report.

### Main Python entry points

- [uv_vae/evaluation.py](uv_vae/evaluation.py)
  - `run_subsample_experiment(...)`: runs the single-draw experiment for in-memory or dataframe-like inputs.
  - `run_parquet_subsample_experiment(...)`: runs the same workflow directly against a parquet file using DuckDB and the project preprocessing pipeline.
  - `diagnose_latent_collapse(...)`: flags latent dimensions whose KL term or variance is near zero, which helps diagnose posterior collapse.
  - `compare_latent_spaces(...)`: compares two latent spaces using Procrustes distance, linear CKA, k-NN Jaccard overlap, trustworthiness, and continuity.
  - `compare_clusterings(...)`: compares two cluster-label assignments with ARI and NMI.
  - `aggregate_results(...)`: assembles a pandas DataFrame with all comparison metrics for downstream analysis.
  - `plot_performance_vs_n(...)`: creates a multi-panel matplotlib figure for time, memory, and latent-space quality versus sample size.

- [uv_vae/cli.py](uv_vae/cli.py)
  - `evaluate-subsamples`: command-line wrapper that runs the parquet-backed evaluation workflow and writes a CSV summary.

### Example CLI run

```bash
conda activate base
cd /path/to/uv_vae
python -m uv_vae.cli evaluate-subsamples \
  --parquet-path path/to/features.parquet \
  --feature-spec-path ml_features.json \
  --fractions 0.05,0.1,0.25,0.5,0.75,1.0 \
  --output-dir artifacts/subsample_evaluation \
  --epochs 10 \
  --batch-size 4096 \
  --latent-dim 16 \
  --hidden-dims 256,128
```

The command prints a summary table to the terminal and writes the results to `artifacts/subsample_evaluation/subsample_evaluation_summary.csv`.