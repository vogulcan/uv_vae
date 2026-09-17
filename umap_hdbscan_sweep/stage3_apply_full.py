"""Stage 3 -- project all 5,078,201,907 reads through a fitted cell, then SigProfiler.

Stage 2 clusters the *deduplicated* loci: one row per (CHROM, POS, REF, ALT), so every
locus counts once regardless of how many reads support it. This stage answers the other
question -- what SigProfiler says when every read votes -- by pushing the entire cohort
through VAE -> UMAP.transform -> HDBSCAN.approximate_predict.

**Nothing per-read is written to disk.** SigProfiler consumes a 96 x n_clusters count
matrix, and that matrix is associative: it can be accumulated batch by batch. Labelling
5.08B reads would be ~20 GB of int32 that nothing ever reads back, so instead each batch
scatter-adds into the count matrix and is discarded. Peak memory is one batch.

Reads at the same locus share an SBS96 context but *not* a cluster -- they differ in SNVQ,
QUAL, MAPQ, DP and RAW_VAF, so the VAE places them at different latent positions. That is
precisely why this cannot be derived from stage 2's ``locus_reads`` column, and why the
full pass is needed to answer the read-weighted question honestly.

Cost: one full encode+transform+predict pass over the cohort. Budget several hours, and run
it for the configurations worth the time rather than all 480.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

# Walk up to the directory that holds uv_vae/ rather than counting levels: these scripts
# get reorganised into subfolders (umap/, hdbscan/, tests/), and a hard-coded parents[N]
# silently resolves to the wrong root the moment one moves, failing on `import uv_vae`.
REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "uv_vae").is_dir()),
                 Path(__file__).resolve().parents[1])
for candidate in (REPO_ROOT / "uv_vae", REPO_ROOT, REPO_ROOT / "uv_vae" / "scripts",
                  Path(__file__).resolve().parent):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np
import polars as pl
import pyarrow.parquet as pq

import sweep_core as core
from stage0_dedup import DEFAULT_ROW_FILTER, resolve_parquet_paths
from uv_vae.inference import LatentInference

COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}
PYRIMIDINE_REFS = ["C", "T"]


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def complement_expr(expr: pl.Expr) -> pl.Expr:
    """Polars complement, matching run_variant_cluster_pipeline.complement_expr."""
    result = pl.when(expr == pl.lit("A")).then(pl.lit("T"))
    for base, complement in [("C", "G"), ("G", "C"), ("T", "A")]:
        result = result.when(expr == pl.lit(base)).then(pl.lit(complement))
    return result.otherwise(pl.lit(None))


def sbs96_expr() -> pl.Expr:
    """Canonical SBS96 label, identical to run_variant_cluster_pipeline.annotate_trinuc_counts.

    Variants with a purine reference are reverse-complemented onto the pyrimidine strand,
    which swaps the flanking bases as well as complementing them.
    """
    ref, alt = pl.col("REF"), pl.col("ALT")
    prev_base, next_base = pl.col("X_PREV1"), pl.col("X_NEXT1")
    is_pyrimidine = ref.is_in(PYRIMIDINE_REFS)

    canonical_ref = pl.when(is_pyrimidine).then(ref).otherwise(complement_expr(ref))
    canonical_alt = pl.when(is_pyrimidine).then(alt).otherwise(complement_expr(alt))
    canonical_prev = pl.when(is_pyrimidine).then(prev_base).otherwise(complement_expr(next_base))
    canonical_next = pl.when(is_pyrimidine).then(next_base).otherwise(complement_expr(prev_base))

    substitution = canonical_ref + pl.lit(">") + canonical_alt
    return (
        pl.when(substitution.is_in(["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]))
        .then(canonical_prev + pl.lit("[") + substitution + pl.lit("]") + canonical_next)
        .otherwise(pl.lit(None))
    )


def load_cell(sweep_root: Path, cell: str) -> tuple[dict, Path]:
    cell_dir = sweep_root / cell
    metrics_path = cell_dir / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"no metrics.json for cell {cell!r} under {sweep_root}")
    payload = json.loads(metrics_path.read_text())
    if "error" in payload:
        raise SystemExit(f"cell {cell!r} failed during the sweep: {payload['error']}")
    return payload, cell_dir


def restore_models(
    cell_payload: dict, cell_dir: Path, embed_dir: Path, use_gpu: bool,
) -> tuple[core.FittedUmap, core.FittedHdbscan]:
    """Reload the sweep's fitted models, refitting from the recorded config if joblib failed.

    A refit is deterministic given the same seed and fit rows, so it reproduces the sweep's
    cell rather than approximating it -- it just costs the UMAP fit again.
    """
    import joblib

    umap_config_payload = json.loads((cell_dir.parent / "umap_config.json").read_text())
    umap_config = core.UmapConfig(
        n_neighbors=umap_config_payload["n_neighbors"],
        min_dist=umap_config_payload["min_dist"],
        n_components=umap_config_payload["n_components"],
        metric=umap_config_payload["metric"],
        seed=umap_config_payload["seed"],
        # Carried through explicitly: a refit here with a different epoch count would not
        # reproduce the sweep's cell, and transform() would stop being batch-invariant.
        n_epochs=umap_config_payload["n_epochs"],
    )
    hdbscan_config = core.HdbscanConfig(**cell_payload["hdbscan"])

    summary = json.loads((embed_dir / "embed_summary.json").read_text())
    latent = np.load(summary["latent_path"], mmap_mode="r")
    fit_rows = cell_payload["fit_rows"]
    if fit_rows >= latent.shape[0]:
        fit_latent = latent
    else:
        indices = np.sort(
            np.random.default_rng(umap_config.seed).choice(latent.shape[0], size=fit_rows, replace=False)
        )
        fit_latent = np.asarray(latent[indices])

    model_path = umap_config_payload.get("model_path")
    if model_path and Path(model_path).exists():
        log(f"Loading UMAP model from {model_path}")
        umap_model = joblib.load(model_path)
        fitted_umap = core.FittedUmap(
            model=umap_model,
            embedding=np.empty((0, umap_config.n_components), dtype=np.float32),
            config=umap_config,
            backend=umap_config_payload.get("backend", "cuml" if use_gpu else "umap-learn"),
        )
        space = fitted_umap.transform(np.asarray(fit_latent))
    else:
        log("No persisted UMAP model; refitting from the recorded config")
        fitted_umap = core.fit_umap(np.asarray(fit_latent), umap_config, use_gpu=use_gpu)
        space = fitted_umap.embedding

    log(f"Refitting HDBSCAN {hdbscan_config.label()} on {space.shape[0]:,} rows")
    fitted_hdbscan = core.fit_hdbscan(space, hdbscan_config, use_gpu=use_gpu)
    return fitted_umap, fitted_hdbscan


def warn_if_batch_too_large(batch_size: int, budget_report: dict) -> None:
    """Compare the encode batch against torch's slice of the budget and say so if it is over.

    A warning rather than a hard failure: ``max_batch_rows`` is sized for *training*
    (activations kept alive for backward, plus optimizer state), and this stage only does
    inference under no_grad, so the real ceiling is several times higher. Erroring on a
    conservative estimate would block runs that fit fine.
    """
    torch_gb = budget_report.get("torch_budget_gb")
    if not torch_gb:
        return
    try:
        from uv_vae import gpu_budget

        spec_path = Path(REPO_ROOT) / "uv_vae" / "ml_features.json"
        shape = gpu_budget.shape_from_specs(str(spec_path))
        per_row = shape.bytes_per_row([256, 128], 16, amp=True)
        ceiling = gpu_budget.max_batch_rows(per_row, budget_gb=torch_gb)
    except Exception:
        return
    if batch_size > ceiling:
        log(f"NOTE: --batch-size {batch_size:,} exceeds the training-sized estimate of "
            f"{ceiling:,} rows for a {torch_gb} GB torch slice. Inference needs far less "
            f"than training, so this often still fits -- but if it OOMs, halve the batch.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project the full cohort through a fitted cell")
    parser.add_argument("--parquet-paths", nargs="+", required=True)
    parser.add_argument("--sweep-root", required=True, help="stage 2 --output-root")
    parser.add_argument("--cell", required=True, help="e.g. nn30_md0.05_nc2/mcs500_ms25")
    parser.add_argument("--embed-dir", required=True, help="stage 1 output")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--feature-spec-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--row-filter", default=DEFAULT_ROW_FILTER)
    parser.add_argument("--batch-size", type=int, default=1_048_576)
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--force-cpu", action="store_true", help="CPU for UMAP/HDBSCAN only")
    parser.add_argument("--gpu-budget-gb", type=float, default=None,
                        help="GPU ceiling for THIS process, split between torch (encode) "
                             "and RMM (cuML transform/predict).")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--sigprofiler-cpu", type=int, default=16)
    parser.add_argument("--cosmic-version", default="3.5")
    parser.add_argument("--genome-build", default="GRCh38")
    parser.add_argument("--max-files", type=int, default=None, help="debug: stop after N files")
    return parser.parse_args()


def main() -> int:
    started = perf_counter()
    args = parse_args()

    import run_variant_cluster_pipeline as rvcp

    paths = resolve_parquet_paths(args.parquet_paths)
    if args.max_files:
        paths = paths[:args.max_files]
    sweep_root = Path(args.sweep_root).resolve()
    embed_dir = Path(args.embed_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    use_gpu = False if args.force_cpu else core.gpu_available()

    # Applied before any model is restored or built. This stage is the tight one: torch
    # holds a batch of activations while cuML holds the UMAP graph and HDBSCAN tree, and
    # they draw on ONE budget -- so a batch size chosen for a 16 GB run will not fit a
    # 6 GB slice. The check below turns that into an up-front error, not a mid-pass OOM.
    budget_report = None
    if args.device != "cpu" or use_gpu:
        budget_report = core.apply_gpu_budget("apply", budget_gb=args.gpu_budget_gb)
        warn_if_batch_too_large(args.batch_size, budget_report)

    cell_payload, cell_dir = load_cell(sweep_root, args.cell)
    log(f"Cell {args.cell}: {cell_payload['metrics']['n_clusters']} clusters on the dedup set")
    fitted_umap, fitted_hdbscan = restore_models(cell_payload, cell_dir, embed_dir, use_gpu)

    inference = LatentInference.from_checkpoint(
        checkpoint_path=str(Path(args.checkpoint_path).resolve()),
        feature_spec_path=args.feature_spec_path,
        parquet_path=str(paths[0]),
        device=args.device,
    )
    feature_names = list(inference.feature_names)

    # Row order for the SBS96 matrix comes from the signature database, so this matrix is
    # index-compatible with the one stage 2 hands SigProfiler.
    sigprof_root = output_dir / f"sigprofilerassignment_uv_only_{args.genome_build.lower()}_v{args.cosmic_version}"
    input_dir = sigprof_root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    signature_database_path = input_dir / f"uv_only_SBS_{args.genome_build}.tsv"
    row_order = rvcp.write_uv_only_signature_database(
        args.genome_build, args.cosmic_version, signature_database_path
    )
    sbs96_index = {label: position for position, label in enumerate(row_order)}

    n_clusters = int(cell_payload["metrics"]["n_clusters"])
    counts = np.zeros((len(row_order), n_clusters), dtype=np.int64)
    noise_reads = 0
    unclassified_reads = 0
    total_reads = 0

    context_columns = ["REF", "ALT", "X_PREV1", "X_NEXT1"]
    filter_columns = sorted({token.strip('"') for token in ("st", "et", "FILT")})
    read_columns = list(dict.fromkeys([*feature_names, *context_columns, *filter_columns]))

    log(f"Streaming {len(paths)} parquet files -> encode -> transform -> predict -> SBS96 counts")
    for file_index, path in enumerate(paths, start=1):
        file_started = perf_counter()
        parquet_file = pq.ParquetFile(path)
        available = set(parquet_file.schema_arrow.names)
        columns = [name for name in read_columns if name in available]
        file_reads = 0

        for group_index in range(parquet_file.num_row_groups):
            frame = pl.from_arrow(parquet_file.read_row_group(group_index, columns=columns))
            frame = frame.filter(
                (pl.col("st") == "MIXED") & (pl.col("et") == "MIXED") & (pl.col("FILT") == 1)
            )
            if frame.height == 0:
                continue

            for start in range(0, frame.height, args.batch_size):
                batch = frame.slice(start, args.batch_size)
                latent = inference.encode_frame(batch.select(feature_names), batch_size=args.batch_size)
                space = fitted_umap.transform(latent, batch_size=args.batch_size)
                labels, _ = fitted_hdbscan.predict(space, batch_size=args.batch_size)

                sbs96 = batch.select(sbs96_expr().alias("sbs96"))["sbs96"].to_numpy()
                rows = np.array([sbs96_index.get(label, -1) for label in sbs96], dtype=np.int64)

                valid = (rows >= 0) & (labels >= 0) & (labels < n_clusters)
                np.add.at(counts, (rows[valid], labels[valid]), 1)

                noise_reads += int((labels < 0).sum())
                unclassified_reads += int((rows < 0).sum())
                file_reads += batch.height
                total_reads += batch.height

        log(f"  [{file_index}/{len(paths)}] {path.name}: {file_reads:,} reads "
            f"({perf_counter() - file_started:.0f}s), {total_reads:,} cumulative")

    matrix_path = input_dir / "cluster_sbs96_matrix.tsv"
    matrix = {"Type": row_order}
    for cluster in range(n_clusters):
        matrix[f"cluster_{cluster}"] = counts[:, cluster].tolist()
    matrix_df = pl.DataFrame(matrix)
    matrix_df.write_csv(matrix_path, separator="\t")
    log(f"SBS96 matrix {matrix_df.shape} -> {matrix_path}")

    output_sigprof = sigprof_root / "output"
    output_sigprof.mkdir(parents=True, exist_ok=True)
    rvcp.Analyzer.cosmic_fit(
        samples=str(matrix_path),
        output=str(output_sigprof),
        signature_database=str(signature_database_path),
        genome_build=args.genome_build,
        cosmic_version=float(args.cosmic_version),
        make_plots=True,
        collapse_to_SBS96=True,
        connected_sigs=False,
        verbose=False,
        input_type="matrix",
        context_type="96",
        export_probabilities=True,
        sample_reconstruction_plots=False,
        cpu=args.sigprofiler_cpu,
        add_background_signatures=False,
    )
    activities_parquet, stats_parquet = rvcp.convert_sigprofiler_outputs_to_parquet(output_sigprof)
    sorted_txt, sorted_parquet = rvcp.sort_sample_stats_by_cosine(output_sigprof)

    summary = {
        "cell": args.cell,
        "parquet_files": len(paths),
        "total_reads": total_reads,
        "assigned_reads": int(counts.sum()),
        "noise_reads": noise_reads,
        "unclassified_context_reads": unclassified_reads,
        "n_clusters": n_clusters,
        # UMAP.transform is not batch-invariant, so this is part of the result, not a tuning
        # knob -- keep it fixed when comparing full-cohort runs.
        "transform_batch_size": args.batch_size,
        "gpu_budget": budget_report,
        "matrix_path": str(matrix_path),
        "sigprofiler_output_dir": str(output_sigprof),
        "activities_parquet": str(activities_parquet),
        "stats_parquet": str(stats_parquet),
        "sorted_stats_txt": str(sorted_txt),
        "sorted_stats_parquet": str(sorted_parquet),
        "dedup_reference": str(cell_dir / "metrics.json"),
        "seconds": round(perf_counter() - started, 1),
    }
    (output_dir / "full_cohort_summary.json").write_text(json.dumps(summary, indent=2))

    log("")
    log(f"{total_reads:,} reads -> {summary['assigned_reads']:,} assigned "
        f"({noise_reads:,} noise, {unclassified_reads:,} non-SBS96 context)")
    log(f"Done in {summary['seconds']}s -> {output_dir / 'full_cohort_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
