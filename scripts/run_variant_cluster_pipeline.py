import cuml.accel 
cuml.accel.install() 
from cuml.common import logger 
logger.set_level(logger.level_enum.debug)

from __future__ import annotations

import argparse
import colorsys
import json
import os
import pickle
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Callable

import hdbscan
import matplotlib
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw, ImageOps
from SigProfilerAssignment import Analyzer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("NUMBA_CACHE_DIR", str((REPO_ROOT / ".numba_cache").resolve()))

from umap import UMAP

from uv_vae.data import connect_duckdb, quote_ident, stream_parquet_batches
from uv_vae.inference import LatentInference
from uv_vae.training import DEFAULT_TRAINING_SAMPLE_ROWS

matplotlib.use("Agg")

DEFAULT_SAMPLE_ROWS = DEFAULT_TRAINING_SAMPLE_ROWS
DEFAULT_COLOR_COLUMNS = [
    "BCSQ",
    "RAW_VAF",
    "DP",
    "SMQ_BEFORE",
    "SMQ_AFTER",
    "EDIST",
    "MAPQ",
    "SNVQ",
]
DEFAULT_PASSTHROUGH_COLUMNS = ["CHROM", "POS", "REF", "ALT", "X_PREV1", "X_NEXT1"]
SBS96_SIGNATURES = ["SBS7A", "SBS7B", "SBS7C", "SBS7D", "SBS38"]
SBS96_REFERENCE_NAMES = {
    "SBS7A": "SBS7a",
    "SBS7B": "SBS7b",
    "SBS7C": "SBS7c",
    "SBS7D": "SBS7d",
    "SBS38": "SBS38",
}
SBS192_SUBSTITUTIONS = [
    "A>C",
    "A>G",
    "A>T",
    "C>A",
    "C>G",
    "C>T",
    "G>A",
    "G>C",
    "G>T",
    "T>A",
    "T>C",
    "T>G",
]
SBS192_CONTEXTS = [f"{left}{right}" for left in "ACGT" for right in "ACGT"]
CANONICAL_SBS96_SUBSTITUTIONS = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]


def unique_columns(columns: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for column in columns:
        if column in seen:
            continue
        seen.add(column)
        result.append(column)
    return result


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def stable_order_by(columns: list[str]) -> str:
    order_columns = ["row_index"] if "row_index" in columns else columns
    return ", ".join(f"{quote_ident(name)} ASC NULLS LAST" for name in order_columns)


def utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remainder:.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes}m {remainder:.1f}s"


def resolve_output_root(checkpoint_path: Path, output_root: str | None) -> Path:
    if output_root is not None:
        target = Path(output_root).expanduser()
        return target if target.is_absolute() else (Path.cwd() / target).resolve()
    return checkpoint_path.resolve().parent / f"pipeline_{utc_timestamp()}"


def build_palette(size: int) -> list[str]:
    golden_ratio = 0.618033988749895
    colors: list[str] = []
    for index in range(size):
        hue = (0.07 + index * golden_ratio) % 1.0
        saturation = 0.58 + 0.22 * ((index % 3) / 2)
        lightness = 0.38 + 0.18 * ((index % 4) / 3)
        red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}")
    return colors


def parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the VAE -> UMAP -> HDBSCAN -> UV signature pipeline")
    parser.add_argument("--checkpoint-path", required=True, help="Path to the saved model.pt checkpoint")
    parser.add_argument("--parquet-path", required=True, help="Path to the source parquet feature map")
    parser.add_argument(
        "--row-filter",
        required=True,
        help="DuckDB SQL filter applied before deduplication, embedding, and clustering",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional output directory. Defaults to checkpoint_path.parent / pipeline_<timestamp>",
    )
    sample_group = parser.add_mutually_exclusive_group()
    sample_group.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help=f"Number of deduplicated rows to sample. Defaults to {DEFAULT_SAMPLE_ROWS:,} when omitted.",
    )
    sample_group.add_argument(
        "--use-all",
        action="store_true",
        help="Use the full deduplicated population instead of sampling a subset.",
    )
    parser.add_argument("--umap-fit-rows", type=int, default=100_000)
    parser.add_argument("--plot-rows", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--embed-batch-size", type=int, default=4096)
    parser.add_argument("--scan-batch-rows", type=int, default=100_000)
    parser.add_argument("--umap-n-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.05)
    parser.add_argument("--umap-metric", default="euclidean")
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=250)
    parser.add_argument("--hdbscan-min-samples", type=int, default=25)
    parser.add_argument(
        "--color-columns",
        default=",".join(DEFAULT_COLOR_COLUMNS),
        help="Comma-separated numeric columns for UMAP coloring",
    )
    parser.add_argument("--sigprofiler-cpu", type=int, default=12)
    parser.add_argument("--cosmic-version", default="3.5")
    parser.add_argument("--genome-build", default="GRCh38")
    parser.add_argument(
        "--duckdb-memory-limit",
        default="4GB",
        help="Memory limit passed to DuckDB so large windowed queries spill to disk instead of exhausting RAM",
    )
    return parser.parse_args()


def load_checkpoint_payload(checkpoint_path: Path) -> dict:
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def load_feature_names(checkpoint_payload: dict) -> list[str]:
    feature_report = checkpoint_payload["feature_report"]
    return unique_columns(
        feature_report["active_categorical_features"] + feature_report["active_numeric_features"]
    )


def write_deduplicated_sample(
    parquet_path: Path,
    output_path: Path,
    row_filter: str,
    selected_columns: list[str],
    sample_rows: int | None,
    seed: int,
    threads: int,
    database_path: Path,
    temp_directory: Path,
    memory_limit: str,
) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    select_list = ", ".join(quote_ident(name) for name in selected_columns)
    output_order = stable_order_by(selected_columns)
    tie_break_order = stable_order_by(selected_columns)
    table_name = "deduplicated_variants"
    dedup_sql = f"""
        CREATE OR REPLACE TEMP TABLE {table_name} AS
        WITH ranked AS (
            SELECT
                {select_list},
                ROW_NUMBER() OVER (
                    PARTITION BY "CHROM", "POS", "REF", "ALT"
                    ORDER BY
                        "SNVQ" DESC NULLS LAST,
                        "QUAL" DESC NULLS LAST,
                        "MAPQ" DESC NULLS LAST,
                        "DP" DESC NULLS LAST,
                        "RAW_VAF" DESC NULLS LAST,
                        {tie_break_order}
                ) AS rn
            FROM read_parquet({sql_quote(str(parquet_path))})
            WHERE {row_filter}
        )
        SELECT {select_list}
        FROM ranked
        WHERE rn = 1
    """
    count_sql = f"SELECT COUNT(*) FROM {table_name}"
    with connect_duckdb(
        threads=threads,
        database=database_path,
        temp_directory=temp_directory,
        memory_limit=memory_limit,
    ) as conn:
        conn.execute(dedup_sql)
        result = conn.execute(count_sql).fetchone()
        if result is None:
            raise RuntimeError("Unable to determine deduplicated population size")
        dedup_population = int(result[0])
        if dedup_population <= 0:
            raise RuntimeError("Deduplicated population is empty after filtering")
        if sample_rows is None:
            actual_sample_rows = dedup_population
            copy_sql = f"""
                COPY (
                    SELECT {select_list}
                    FROM {table_name}
                    ORDER BY {output_order}
                ) TO {sql_quote(str(output_path))} (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        else:
            actual_sample_rows = min(sample_rows, dedup_population)
            conn.execute("SET threads = 1")
            copy_sql = f"""
                COPY (
                    SELECT {select_list}
                    FROM (
                        SELECT {select_list}
                        FROM {table_name}
                        USING SAMPLE reservoir({int(actual_sample_rows)} ROWS)
                        REPEATABLE ({int(seed)})
                    ) AS sampled_rows
                    ORDER BY {output_order}
                ) TO {sql_quote(str(output_path))} (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        conn.execute(copy_sql)
    return dedup_population, actual_sample_rows


def sample_parquet_exact(
    source_path: Path,
    output_path: Path,
    sample_rows: int,
    seed: int,
    threads: int,
    selected_columns: list[str] | None = None,
    database_path: Path | None = None,
    temp_directory: Path | None = None,
    memory_limit: str | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    select_list = "*" if selected_columns is None else ", ".join(quote_ident(name) for name in selected_columns)
    order_clause = "" if selected_columns is None else f"\n            ORDER BY {stable_order_by(selected_columns)}"
    sql = f"""
        COPY (
            SELECT {select_list}
            FROM (
                SELECT {select_list}
                FROM read_parquet({sql_quote(str(source_path))})
                USING SAMPLE reservoir({int(sample_rows)} ROWS)
                REPEATABLE ({int(seed)})
            ) AS sampled_rows{order_clause}
        ) TO {sql_quote(str(output_path))} (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    with connect_duckdb(
        threads=1,
        database=database_path or ":memory:",
        temp_directory=temp_directory,
        memory_limit=memory_limit,
    ) as conn:
        conn.execute(sql)


def row_count(
    parquet_path: Path,
    threads: int,
    database_path: Path | None = None,
    temp_directory: Path | None = None,
    memory_limit: str | None = None,
) -> int:
    with connect_duckdb(
        threads=threads,
        database=database_path or ":memory:",
        temp_directory=temp_directory,
        memory_limit=memory_limit,
    ) as conn:
        result = conn.execute(
            f"SELECT COUNT(*) FROM read_parquet({sql_quote(str(parquet_path))})"
        ).fetchone()
    if result is None:
        raise RuntimeError(f"Unable to count rows in {parquet_path}")
    return int(result[0])


def verify_unique_variants(
    sample_path: Path,
    threads: int,
    database_path: Path | None = None,
    temp_directory: Path | None = None,
    memory_limit: str | None = None,
) -> None:
    sql = f"""
        SELECT COUNT(*) = COUNT(DISTINCT ("CHROM", "POS", "REF", "ALT"))
        FROM read_parquet({sql_quote(str(sample_path))})
    """
    with connect_duckdb(
        threads=threads,
        database=database_path or ":memory:",
        temp_directory=temp_directory,
        memory_limit=memory_limit,
    ) as conn:
        result = conn.execute(sql).fetchone()
    if result is None or not bool(result[0]):
        raise RuntimeError("Sampled deduplicated parquet is not unique on CHROM/POS/REF/ALT")


def verify_max_snvq(
    sample_path: Path,
    source_path: Path,
    row_filter: str,
    threads: int,
    seed: int,
    verify_rows: int = 1024,
    database_path: Path | None = None,
    temp_directory: Path | None = None,
    memory_limit: str | None = None,
) -> None:
    sampled_rows = min(verify_rows, row_count(sample_path, threads))
    sql = f"""
        WITH sampled AS (
            SELECT "CHROM", "POS", "REF", "ALT", "SNVQ"
            FROM read_parquet({sql_quote(str(sample_path))})
            USING SAMPLE reservoir({int(sampled_rows)} ROWS)
            REPEATABLE ({int(seed)})
        ),
        source_max AS (
            SELECT
                source."CHROM",
                source."POS",
                source."REF",
                source."ALT",
                MAX(source."SNVQ") AS max_snvq
            FROM read_parquet({sql_quote(str(source_path))}) AS source
            JOIN sampled USING ("CHROM", "POS", "REF", "ALT")
            WHERE {row_filter}
            GROUP BY 1, 2, 3, 4
        )
        SELECT COUNT(*)
        FROM sampled
        JOIN source_max USING ("CHROM", "POS", "REF", "ALT")
        WHERE sampled."SNVQ" IS DISTINCT FROM source_max.max_snvq
    """
    with connect_duckdb(
        threads=threads,
        database=database_path or ":memory:",
        temp_directory=temp_directory,
        memory_limit=memory_limit,
    ) as conn:
        mismatches = conn.execute(sql).fetchone()
    if mismatches is None:
        raise RuntimeError("Unable to verify SNVQ maxima")
    if int(mismatches[0]) != 0:
        raise RuntimeError(f"Found {int(mismatches[0])} sampled rows that do not match the group max SNVQ")


def fit_umap_and_hdbscan(
    fit_sample_path: Path,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    min_cluster_size: int,
    min_samples: int,
    seed: int,
) -> tuple[UMAP, hdbscan.HDBSCAN, pl.DataFrame]:
    latent_columns = [name for name in pq.read_schema(fit_sample_path).names if name.startswith("latent_")]
    if not latent_columns:
        raise RuntimeError("No latent columns found in the UMAP fit sample")
    fit_df = pl.read_parquet(fit_sample_path, columns=["row_index", *latent_columns])

    latent_matrix = fit_df.select(latent_columns).to_numpy().astype(np.float32, copy=False)
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
        transform_seed=seed,
    )
    fit_coords = umap_model.fit_transform(latent_matrix).astype(np.float32, copy=False)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="eom",
        metric="euclidean",
        core_dist_n_jobs=1,
        prediction_data=True,
    )
    clusterer.fit(fit_coords)

    fit_df = fit_df.with_columns(
        [
            pl.Series("umap_1", fit_coords[:, 0]),
            pl.Series("umap_2", fit_coords[:, 1]),
            pl.Series("cluster_label", clusterer.labels_.astype(np.int32, copy=False)),
            pl.Series("cluster_probability", clusterer.probabilities_.astype(np.float32, copy=False)),
        ]
    )
    return umap_model, clusterer, fit_df, fit_coords


def transform_embeddings_to_analysis(
    embedding_path: Path,
    analysis_path: Path,
    umap_model: UMAP,
    clusterer: hdbscan.HDBSCAN,
    fit_assignment_rows: np.ndarray,
    fit_assignment_labels: np.ndarray,
    fit_assignment_strengths: np.ndarray,
    passthrough_columns: list[str],
    scan_batch_rows: int,
    threads: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> int:
    latent_columns = [
        name for name in pq.read_schema(embedding_path).names if name.startswith("latent_")
    ]
    selected_columns = unique_columns(["row_index", *passthrough_columns, *latent_columns])
    output_columns = unique_columns(["row_index", *passthrough_columns])

    writer: pq.ParquetWriter | None = None
    rows_written = 0
    batch_index = 0
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with connect_duckdb(threads=threads) as conn:
            for record_batch in stream_parquet_batches(
                conn=conn,
                parquet_path=embedding_path,
                select_columns=selected_columns,
                rows_per_batch=scan_batch_rows,
            ):
                frame = pl.from_arrow(record_batch)
                latents = frame.select(latent_columns).to_numpy().astype(np.float32, copy=False)
                coords = umap_model.transform(latents).astype(np.float32, copy=False)
                labels, strengths = hdbscan.approximate_predict(clusterer, coords)
                labels = labels.astype(np.int32, copy=False)
                strengths = strengths.astype(np.float32, copy=False)

                row_indices = frame["row_index"].to_numpy()
                positions = np.searchsorted(fit_assignment_rows, row_indices)
                valid_positions = positions < fit_assignment_rows.size
                fit_mask = np.zeros(row_indices.shape, dtype=bool)
                fit_mask[valid_positions] = (
                    fit_assignment_rows[positions[valid_positions]] == row_indices[valid_positions]
                )
                if fit_mask.any():
                    labels[fit_mask] = fit_assignment_labels[positions[fit_mask]]
                    strengths[fit_mask] = fit_assignment_strengths[positions[fit_mask]]

                output_frame = frame.select(output_columns).with_columns(
                    [
                        pl.Series("umap_1", coords[:, 0]),
                        pl.Series("umap_2", coords[:, 1]),
                        pl.Series("cluster_label", labels),
                        pl.Series("cluster_probability", strengths),
                    ]
                )
                table = output_frame.to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(analysis_path, table.schema)
                writer.write_table(table)
                rows_written += output_frame.height
                batch_index += 1
                if progress_callback is not None:
                    progress_callback(rows_written, batch_index)
    finally:
        if writer is not None:
            writer.close()

    return rows_written


def save_pickle(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def save_predictor_state(
    output_dir: Path,
    umap_model: UMAP,
    clusterer: hdbscan.HDBSCAN,
    fit_df: pl.DataFrame,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    umap_model_path = save_pickle(output_dir / "umap_model.pkl", umap_model)
    hdbscan_clusterer_path = save_pickle(output_dir / "hdbscan_clusterer.pkl", clusterer)
    fit_assignments_path = output_dir / "fit_assignments.parquet"
    fit_df.select(["row_index", "umap_1", "umap_2", "cluster_label", "cluster_probability"]).write_parquet(
        fit_assignments_path
    )
    return {
        "directory": str(output_dir),
        "umap_model_path": str(umap_model_path),
        "hdbscan_clusterer_path": str(hdbscan_clusterer_path),
        "fit_assignments_path": str(fit_assignments_path),
    }


def make_contact_sheet(
    items: list[tuple[str, Path]],
    output_path: Path,
    columns: int = 2,
    thumb_width: int = 900,
    thumb_height: int = 620,
) -> None:
    rows = (len(items) + columns - 1) // columns
    padding = 24
    label_height = 42
    canvas = Image.new(
        "RGB",
        (
            columns * thumb_width + (columns + 1) * padding,
            rows * (thumb_height + label_height) + (rows + 1) * padding,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, path) in enumerate(items):
        image = Image.open(path).convert("RGB")
        image = ImageOps.contain(image, (thumb_width, thumb_height))
        x0 = padding + (index % columns) * (thumb_width + padding)
        y0 = padding + (index // columns) * (thumb_height + label_height + padding)
        canvas.paste(image, (x0 + (thumb_width - image.width) // 2, y0))
        draw.text((x0, y0 + thumb_height + 8), label, fill="black")
    canvas.save(output_path, quality=95)


def numeric_umap_plot(
    plot_df: pl.DataFrame,
    column: str,
    output_path: Path,
) -> None:
    x = plot_df["umap_1"].to_numpy()
    y = plot_df["umap_2"].to_numpy()
    values = plot_df[column].cast(pl.Float64).to_numpy()
    finite_mask = np.isfinite(values)

    fig, axis = plt.subplots(figsize=(8.5, 6.5), dpi=180)
    if (~finite_mask).any():
        axis.scatter(x[~finite_mask], y[~finite_mask], s=3, c="#d0d0d0", alpha=0.45, linewidths=0)
    if finite_mask.any():
        finite_values = values[finite_mask]
        vmin = float(np.quantile(finite_values, 0.01))
        vmax = float(np.quantile(finite_values, 0.99))
        scatter = axis.scatter(
            x[finite_mask],
            y[finite_mask],
            s=3,
            c=finite_values,
            alpha=0.78,
            linewidths=0,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        colorbar = fig.colorbar(scatter, ax=axis)
        colorbar.set_label(column)
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.set_title(f"UMAP colored by {column}")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def cluster_umap_plot(
    plot_df: pl.DataFrame,
    cluster_sizes: dict[int, int],
    output_path: Path,
) -> None:
    labels = plot_df["cluster_label"].cast(pl.Int32).to_numpy()
    x = plot_df["umap_1"].to_numpy()
    y = plot_df["umap_2"].to_numpy()
    ordered_clusters = sorted(cluster_sizes, key=cluster_sizes.get, reverse=True)
    palette = build_palette(len(ordered_clusters))
    label_to_color = {-1: "#d3d3d3"}
    for index, label in enumerate(ordered_clusters):
        label_to_color[label] = palette[index]

    point_colors = [label_to_color.get(int(label), "#6b6b6b") for label in labels]
    fig, axis = plt.subplots(figsize=(9, 7), dpi=180)
    axis.scatter(x, y, s=3, c=point_colors, alpha=0.82, linewidths=0)
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.set_title("HDBSCAN clusters on UMAP space")
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#d3d3d3",
            markeredgecolor="none",
            markersize=6,
            label=f"noise (n={int(np.sum(labels == -1)):,})",
        )
    ]
    for label in ordered_clusters[:20]:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=label_to_color[label],
                markeredgecolor="none",
                markersize=6,
                label=f"cluster {label} (n={cluster_sizes[label]:,})",
            )
        )
    axis.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def probability_umap_plot(plot_df: pl.DataFrame, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 7), dpi=180)
    scatter = axis.scatter(
        plot_df["umap_1"].to_numpy(),
        plot_df["umap_2"].to_numpy(),
        s=3,
        c=plot_df["cluster_probability"].cast(pl.Float64).to_numpy(),
        alpha=0.82,
        linewidths=0,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.set_title("HDBSCAN membership probability on UMAP space")
    colorbar = fig.colorbar(scatter, ax=axis)
    colorbar.set_label("Cluster probability")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def complement_expr(expr: pl.Expr) -> pl.Expr:
    return (
        pl.when(expr == pl.lit("A"))
        .then(pl.lit("T"))
        .when(expr == pl.lit("C"))
        .then(pl.lit("G"))
        .when(expr == pl.lit("G"))
        .then(pl.lit("C"))
        .when(expr == pl.lit("T"))
        .then(pl.lit("A"))
        .otherwise(pl.lit(None))
    )


def query_frame(
    sql: str,
    threads: int,
    database_path: Path | None = None,
    temp_directory: Path | None = None,
    memory_limit: str | None = None,
) -> pl.DataFrame:
    with connect_duckdb(
        threads=threads,
        database=database_path or ":memory:",
        temp_directory=temp_directory,
        memory_limit=memory_limit,
    ) as conn:
        return conn.execute(sql).pl()


def query_cluster_stats(
    analysis_path: Path,
    threads: int,
    database_path: Path | None = None,
    temp_directory: Path | None = None,
    memory_limit: str | None = None,
) -> pl.DataFrame:
    sql = f"""
        SELECT
            CAST("cluster_label" AS INTEGER) AS cluster_label,
            COUNT(*) AS cluster_size,
            AVG("cluster_probability") AS mean_cluster_probability,
            median("cluster_probability") AS median_cluster_probability
        FROM read_parquet({sql_quote(str(analysis_path))})
        WHERE "cluster_label" >= 0
        GROUP BY 1
        ORDER BY cluster_size DESC, cluster_label ASC
    """
    return query_frame(
        sql,
        threads=threads,
        database_path=database_path,
        temp_directory=temp_directory,
        memory_limit=memory_limit,
    )


def annotate_trinuc_counts(
    counts_df: pl.DataFrame,
    cluster_stats: pl.DataFrame,
) -> pl.DataFrame:
    ref_expr = pl.col("REF")
    alt_expr = pl.col("ALT")
    prev_expr = pl.col("X_PREV1")
    next_expr = pl.col("X_NEXT1")
    canonical_ref = pl.when(ref_expr.is_in(["C", "T"])).then(ref_expr).otherwise(complement_expr(ref_expr))
    canonical_alt = pl.when(ref_expr.is_in(["C", "T"])).then(alt_expr).otherwise(complement_expr(alt_expr))
    canonical_prev = pl.when(ref_expr.is_in(["C", "T"])).then(prev_expr).otherwise(complement_expr(next_expr))
    canonical_next = pl.when(ref_expr.is_in(["C", "T"])).then(next_expr).otherwise(complement_expr(prev_expr))
    canonical_sub = canonical_ref + pl.lit(">") + canonical_alt
    canonical_sbs96 = pl.when(canonical_sub.is_in(CANONICAL_SBS96_SUBSTITUTIONS)).then(
        canonical_prev + pl.lit("[") + canonical_sub + pl.lit("]") + canonical_next
    )
    return (
        counts_df.with_columns(
            [
                (ref_expr + pl.lit(">") + alt_expr).alias("ref_alt"),
                (prev_expr + next_expr).alias("context16"),
                (prev_expr + pl.lit("[") + ref_expr + pl.lit(">") + alt_expr + pl.lit("]") + next_expr).alias(
                    "trinuc192"
                ),
                canonical_sbs96.alias("sbs96"),
            ]
        )
        .join(cluster_stats.select(["cluster_label", "cluster_size"]), on="cluster_label", how="left")
        .with_columns((pl.col("count") / pl.col("cluster_size")).alias("fraction"))
    )

def query_trinuc192_counts(
    analysis_path: Path,
    cluster_stats: pl.DataFrame,
    threads: int,
    database_path: Path | None = None,
    temp_directory: Path | None = None,
    memory_limit: str | None = None,
) -> pl.DataFrame:
    sql = f"""
        SELECT
            CAST("cluster_label" AS INTEGER) AS cluster_label,
            "REF",
            "ALT",
            "X_PREV1",
            "X_NEXT1",
            COUNT(*) AS count
        FROM read_parquet({sql_quote(str(analysis_path))})
        WHERE "cluster_label" >= 0
        GROUP BY 1, 2, 3, 4, 5
        ORDER BY cluster_label ASC, count DESC
    """
    counts_df = query_frame(
        sql,
        threads=threads,
        database_path=database_path,
        temp_directory=temp_directory,
        memory_limit=memory_limit,
    )
    return annotate_trinuc_counts(counts_df, cluster_stats)


def query_noise_count(
    analysis_path: Path,
    threads: int,
    database_path: Path | None = None,
    temp_directory: Path | None = None,
    memory_limit: str | None = None,
) -> int:
    sql = f"""
        SELECT COUNT(*)
        FROM read_parquet({sql_quote(str(analysis_path))})
        WHERE "cluster_label" = -1
    """
    with connect_duckdb(
        threads=threads,
        database=database_path or ":memory:",
        temp_directory=temp_directory,
        memory_limit=memory_limit,
    ) as conn:
        result = conn.execute(sql).fetchone()
    if result is None:
        raise RuntimeError("Unable to count noise assignments")
    return int(result[0])


def read_cluster_points(
    analysis_path: Path,
    cluster_label: int,
    threads: int,
    database_path: Path | None = None,
    temp_directory: Path | None = None,
    memory_limit: str | None = None,
) -> np.ndarray:
    sql = f"""
        SELECT "umap_1", "umap_2", "cluster_probability"
        FROM read_parquet({sql_quote(str(analysis_path))})
        WHERE "cluster_label" = {int(cluster_label)}
    """
    frame = query_frame(
        sql,
        threads=threads,
        database_path=database_path,
        temp_directory=temp_directory,
        memory_limit=memory_limit,
    )
    return frame.select(["umap_1", "umap_2", "cluster_probability"]).to_numpy()


def write_cluster_profiles(
    output_dir: Path,
    plot_df: pl.DataFrame,
    analysis_path: Path,
    cluster_stats: pl.DataFrame,
    trinuc192_counts: pl.DataFrame,
    threads: int,
    database_path: Path | None = None,
    temp_directory: Path | None = None,
    memory_limit: str | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir.parent / "cluster_profiles_manifest.json"
    trinuc192_path = output_dir.parent / "cluster_trinuc192.parquet"
    trinuc192_counts.write_parquet(trinuc192_path)

    plot_x = plot_df["umap_1"].to_numpy()
    plot_y = plot_df["umap_2"].to_numpy()
    limits = (
        float(plot_x.min() - 0.5),
        float(plot_x.max() + 0.5),
        float(plot_y.min() - 0.5),
        float(plot_y.max() + 0.5),
    )
    context_colors = {
        context: color
        for context, color in zip(
            SBS192_CONTEXTS,
            [
                "#1b9e77",
                "#d95f02",
                "#7570b3",
                "#e7298a",
                "#66a61e",
                "#e6ab02",
                "#a6761d",
                "#666666",
                "#1f78b4",
                "#b2df8a",
                "#fb9a99",
                "#fdbf6f",
                "#cab2d6",
                "#6a3d9a",
                "#ff7f00",
                "#b15928",
            ],
            strict=True,
        )
    }
    ordered_labels = cluster_stats["cluster_label"].to_list()
    palette = build_palette(len(ordered_labels))
    label_to_color = {int(label): palette[index] for index, label in enumerate(ordered_labels)}

    count_lookup = {
        (int(row["cluster_label"]), row["ref_alt"], row["context16"]): int(row["count"])
        for row in trinuc192_counts.to_dicts()
    }
    top_lookup: dict[int, list[dict[str, object]]] = {}
    for row in cluster_stats.to_dicts():
        label = int(row["cluster_label"])
        top_lookup[label] = (
            trinuc192_counts.filter(pl.col("cluster_label") == label)
            .sort("count", descending=True)
            .head(6)
            .to_dicts()
        )

    manifest: list[dict[str, object]] = []
    for row in cluster_stats.to_dicts():
        label = int(row["cluster_label"])
        cluster_size = int(row["cluster_size"])
        cluster_color = label_to_color[label]
        cluster_points = read_cluster_points(
            analysis_path=analysis_path,
            cluster_label=label,
            threads=threads,
            database_path=database_path,
            temp_directory=temp_directory,
            memory_limit=memory_limit,
        )

        fig = plt.figure(figsize=(19, 12), dpi=180)
        grid = GridSpec(4, 4, figure=fig, width_ratios=[2.4, 1, 1, 1], hspace=0.45, wspace=0.28)

        umap_axis = fig.add_subplot(grid[:, 0])
        umap_axis.scatter(plot_x, plot_y, s=2, c="#d9d9d9", alpha=0.22, linewidths=0)
        umap_axis.scatter(cluster_points[:, 0], cluster_points[:, 1], s=4, c=cluster_color, alpha=0.88, linewidths=0)
        umap_axis.set_xlim(limits[0], limits[1])
        umap_axis.set_ylim(limits[2], limits[3])
        umap_axis.set_xlabel("UMAP 1")
        umap_axis.set_ylabel("UMAP 2")
        umap_axis.set_title(f"Cluster {label} on UMAP")
        summary_lines = [
            f"n = {cluster_size:,}",
            f"mean p = {float(row['mean_cluster_probability']):.3f}",
            f"median p = {float(row['median_cluster_probability']):.3f}",
            "Top trinucs:",
        ]
        summary_lines.extend(
            f"{item['trinuc192']}  {int(item['count']):,} ({float(item['fraction']):.2%})"
            for item in top_lookup[label]
        )
        umap_axis.text(
            1.02,
            0.98,
            "\n".join(summary_lines),
            transform=umap_axis.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "#cccccc"},
        )

        max_fraction = 0.0
        values_by_substitution: dict[str, np.ndarray] = {}
        for substitution in SBS192_SUBSTITUTIONS:
            values = np.array(
                [
                    count_lookup.get((label, substitution, context), 0) / cluster_size
                    for context in SBS192_CONTEXTS
                ],
                dtype=np.float64,
            )
            values_by_substitution[substitution] = values
            if values.size:
                max_fraction = max(max_fraction, float(values.max()))
        y_max = max_fraction * 1.12 if max_fraction > 0 else 1.0

        for index, substitution in enumerate(SBS192_SUBSTITUTIONS):
            grid_row = index // 3
            grid_col = 1 + (index % 3)
            axis = fig.add_subplot(grid[grid_row, grid_col])
            values = values_by_substitution[substitution]
            axis.bar(
                range(len(SBS192_CONTEXTS)),
                values,
                color=[context_colors[context] for context in SBS192_CONTEXTS],
                width=0.82,
            )
            axis.set_title(substitution, fontsize=10)
            axis.set_ylim(0, y_max)
            axis.set_xticks(range(len(SBS192_CONTEXTS)))
            axis.set_xticklabels(SBS192_CONTEXTS, rotation=90, fontsize=7)
            if grid_col == 1:
                axis.set_ylabel("Fraction")
            else:
                axis.set_yticklabels([])
            axis.grid(axis="y", alpha=0.15)

        fig.suptitle(
            f"HDBSCAN Cluster {label}: strand-specific trinucleotide distribution (192)",
            fontsize=16,
            y=0.995,
        )
        plot_path = output_dir / f"cluster_{label:02d}_n{cluster_size}.png"
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)

        manifest.append(
            {
                "cluster_label": label,
                "cluster_size": cluster_size,
                "mean_probability": float(row["mean_cluster_probability"]),
                "median_probability": float(row["median_cluster_probability"]),
                "plot_path": str(plot_path),
                "top_trinucs": top_lookup[label],
            }
        )

    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path, trinuc192_path


def bundled_reference_signature_path(genome_build: str, cosmic_version: str) -> Path:
    import SigProfilerAssignment

    base = Path(SigProfilerAssignment.__file__).resolve().parent
    reference = base / "data" / "Reference_Signatures" / genome_build / f"COSMIC_v{cosmic_version}_SBS_{genome_build}.txt"
    if not reference.exists():
        raise FileNotFoundError(f"Unable to find packaged COSMIC reference at {reference}")
    return reference


def write_uv_only_signature_database(
    genome_build: str,
    cosmic_version: str,
    output_path: Path,
) -> list[str]:
    reference = pl.read_csv(bundled_reference_signature_path(genome_build, cosmic_version), separator="\t")
    uv_reference = reference.select(
        ["Type"] + [SBS96_REFERENCE_NAMES[name] for name in SBS96_SIGNATURES]
    ).rename({SBS96_REFERENCE_NAMES[name]: name for name in SBS96_SIGNATURES})
    uv_reference.write_csv(output_path, separator="\t")
    return uv_reference["Type"].to_list()


def write_cluster_sbs96_matrix(
    trinuc192_counts: pl.DataFrame,
    output_path: Path,
    row_order: list[str],
) -> pl.DataFrame:
    clustered = trinuc192_counts.filter(pl.col("sbs96").is_not_null())
    counts = (
        clustered.group_by(["cluster_label", "sbs96"])
        .agg(pl.col("count").sum().alias("count"))
    )
    cluster_labels = sorted(clustered["cluster_label"].unique().cast(pl.Int32).to_list())
    lookup = {(int(row["cluster_label"]), row["sbs96"]): int(row["count"]) for row in counts.to_dicts()}

    matrix = {"Type": row_order}
    for label in cluster_labels:
        matrix[f"cluster_{label}"] = [lookup.get((label, sbs96), 0) for sbs96 in row_order]

    matrix_df = pl.DataFrame(matrix)
    matrix_df.write_csv(output_path, separator="\t")
    return matrix_df


def convert_sigprofiler_outputs_to_parquet(sigprofiler_output_dir: Path) -> tuple[Path, Path]:
    activities_txt = sigprofiler_output_dir / "Assignment_Solution" / "Activities" / "Assignment_Solution_Activities.txt"
    stats_txt = sigprofiler_output_dir / "Assignment_Solution" / "Solution_Stats" / "Assignment_Solution_Samples_Stats.txt"
    activities_parquet = activities_txt.with_suffix(".parquet")
    stats_parquet = stats_txt.with_suffix(".parquet")

    pl.read_csv(activities_txt, separator="\t").write_parquet(activities_parquet)
    pl.read_csv(stats_txt, separator="\t").write_parquet(stats_parquet)
    return activities_parquet, stats_parquet


def sort_sample_stats_by_cosine(sigprofiler_output_dir: Path) -> tuple[Path, Path]:
    stats_txt = sigprofiler_output_dir / "Assignment_Solution" / "Solution_Stats" / "Assignment_Solution_Samples_Stats.txt"
    sorted_txt = stats_txt.with_name("Assignment_Solution_Samples_Stats.sorted_by_cosine_similarity_desc.txt")
    sorted_parquet = stats_txt.with_name("Assignment_Solution_Samples_Stats.sorted_by_cosine_similarity_desc.parquet")
    stats_df = pl.read_csv(stats_txt, separator="\t").sort("Cosine Similarity", descending=True)
    stats_df.write_csv(sorted_txt, separator="\t")
    stats_df.write_parquet(sorted_parquet)
    return sorted_txt, sorted_parquet


def dominant_signature_columns(activities_df: pl.DataFrame) -> pl.DataFrame:
    signature_columns = [column for column in activities_df.columns if column != "Samples"]
    rows: list[dict[str, object]] = []
    for row in activities_df.to_dicts():
        best_signature = max(signature_columns, key=lambda name: row[name])
        total = sum(float(row[name]) for name in signature_columns)
        best_value = float(row[best_signature])
        rows.append(
            {
                **row,
                "dominant_signature": best_signature,
                "dominant_count": best_value,
                "dominant_fraction": (best_value / total) if total else 0.0,
            }
        )
    return pl.DataFrame(rows)


def build_cluster_summary(
    cluster_stats: pl.DataFrame,
    sigprofiler_output_dir: Path,
    output_path: Path,
    output_json_path: Path,
) -> pl.DataFrame:
    activities_df = dominant_signature_columns(
        pl.read_csv(
            sigprofiler_output_dir / "Assignment_Solution" / "Activities" / "Assignment_Solution_Activities.txt",
            separator="\t",
        )
    ).rename({"Samples": "sample_name"})
    stats_df = pl.read_csv(
        sigprofiler_output_dir / "Assignment_Solution" / "Solution_Stats" / "Assignment_Solution_Samples_Stats.txt",
        separator="\t",
    ).rename({"Sample Names": "sample_name"})

    activities_df = activities_df.with_columns(
        pl.col("sample_name").str.replace("cluster_", "").cast(pl.Int64).alias("cluster_label")
    )
    stats_df = stats_df.with_columns(
        pl.col("sample_name").str.replace("cluster_", "").cast(pl.Int64).alias("cluster_label")
    )
    summary_df = (
        cluster_stats.join(activities_df, on="cluster_label", how="left")
        .join(stats_df, on=["cluster_label", "sample_name"], how="left")
        .sort("cluster_size", descending=True)
    )
    summary_df.write_parquet(output_path)
    output_json_path.write_text(json.dumps(summary_df.to_dicts(), indent=2))
    return summary_df


def run_sigprofiler_assignment(
    trinuc192_counts: pl.DataFrame,
    output_root: Path,
    genome_build: str,
    cosmic_version: str,
    cpu: int,
) -> dict[str, str]:
    sigprof_root = output_root / f"sigprofilerassignment_uv_only_{genome_build.lower()}_v{cosmic_version}"
    input_dir = sigprof_root / "input"
    output_dir = sigprof_root / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    signature_database_path = input_dir / f"uv_only_SBS_{genome_build}.tsv"
    matrix_path = input_dir / "cluster_sbs96_matrix.tsv"
    row_order = write_uv_only_signature_database(genome_build, cosmic_version, signature_database_path)
    matrix_df = write_cluster_sbs96_matrix(trinuc192_counts, matrix_path, row_order)
    if matrix_df.width <= 1:
        raise RuntimeError("HDBSCAN produced no non-noise clusters, so SigProfilerAssignment cannot run")

    Analyzer.cosmic_fit(
        samples=str(matrix_path),
        output=str(output_dir),
        signature_database=str(signature_database_path),
        genome_build=genome_build,
        cosmic_version=float(cosmic_version),
        make_plots=True,
        collapse_to_SBS96=True,
        connected_sigs=False,
        verbose=False,
        input_type="matrix",
        context_type="96",
        export_probabilities=True,
        sample_reconstruction_plots=False,
        cpu=cpu,
        add_background_signatures=False,
    )
    activities_parquet, stats_parquet = convert_sigprofiler_outputs_to_parquet(output_dir)
    sorted_txt, sorted_parquet = sort_sample_stats_by_cosine(output_dir)

    return {
        "root": str(sigprof_root),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "signature_database_path": str(signature_database_path),
        "matrix_path": str(matrix_path),
        "matrix_rows": str(matrix_df.height),
        "matrix_columns": str(matrix_df.width - 1),
        "activities_parquet": str(activities_parquet),
        "stats_parquet": str(stats_parquet),
        "sorted_stats_txt": str(sorted_txt),
        "sorted_stats_parquet": str(sorted_parquet),
    }


def write_pipeline_summary(output_path: Path, payload: dict) -> None:
    output_path.write_text(json.dumps(payload, indent=2))


def main() -> int:
    overall_start = perf_counter()
    args = parse_args()
    checkpoint_path = Path(args.checkpoint_path).resolve()
    parquet_path = Path(args.parquet_path).resolve()
    output_root = resolve_output_root(checkpoint_path, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    database_path = output_root / "pipeline.duckdb"
    temp_directory = output_root / "duckdb_tmp"

    checkpoint_payload = load_checkpoint_payload(checkpoint_path)
    feature_names = load_feature_names(checkpoint_payload)
    color_columns = parse_csv_list(args.color_columns)
    passthrough_columns = unique_columns(DEFAULT_PASSTHROUGH_COLUMNS + color_columns)
    selected_columns = unique_columns(feature_names + passthrough_columns)
    requested_sample_rows = None if args.use_all else (args.sample_rows or DEFAULT_SAMPLE_ROWS)
    requested_sample_label = "all" if args.use_all else f"{requested_sample_rows:,}"
    row_filter = args.row_filter
    log(
        "Starting variant clustering pipeline "
        f"checkpoint={checkpoint_path} parquet={parquet_path} output_root={output_root}"
    )
    log(
        f"Configured row_filter={row_filter!r}, sample_rows={requested_sample_label}, "
        f"umap_fit_rows={args.umap_fit_rows:,}, plot_rows={args.plot_rows:,}, "
        f"seed={args.seed}, threads={args.threads}, duckdb_memory_limit={args.duckdb_memory_limit}"
    )

    sampled_dedup_path = output_root / "sampled_deduplicated_variants.parquet"
    embedding_path = output_root / "sampled_deduplicated_variants_embeddings.parquet"
    umap_fit_path = output_root / "umap_fit_sample.parquet"
    analysis_path = output_root / "analysis.parquet"
    plot_sample_path = output_root / "plot_sample.parquet"
    plots_dir = output_root / "plots"
    cluster_profiles_dir = output_root / "cluster_profiles"
    predictor_state_dir = output_root / "predictor_state"
    plots_dir.mkdir(parents=True, exist_ok=True)
    cluster_profiles_dir.mkdir(parents=True, exist_ok=True)

    dedup_start = perf_counter()
    log("Deduplicating source parquet by CHROM/POS/REF/ALT with max-SNVQ tie-break ordering")
    dedup_population, sampled_rows = write_deduplicated_sample(
        parquet_path=parquet_path,
        output_path=sampled_dedup_path,
        row_filter=row_filter,
        selected_columns=selected_columns,
        sample_rows=requested_sample_rows,
        seed=args.seed,
        threads=args.threads,
        database_path=database_path,
        temp_directory=temp_directory,
        memory_limit=args.duckdb_memory_limit,
    )
    if requested_sample_rows is not None and sampled_rows < requested_sample_rows:
        log(
            f"WARNING: requested sample_rows={requested_sample_rows:,} exceeds deduplicated population={dedup_population:,}; "
            f"using sampled_rows={sampled_rows:,}"
        )
    log(
        f"Deduplicated population={dedup_population:,}; sampled {sampled_rows:,} rows to {sampled_dedup_path} "
        f"in {format_seconds(perf_counter() - dedup_start)}"
    )

    verify_start = perf_counter()
    log("Verifying sampled parquet uniqueness and spot-checking SNVQ maxima")
    verify_unique_variants(
        sampled_dedup_path,
        threads=args.threads,
        database_path=database_path,
        temp_directory=temp_directory,
        memory_limit=args.duckdb_memory_limit,
    )
    verify_max_snvq(
        sampled_dedup_path,
        parquet_path,
        row_filter,
        threads=args.threads,
        seed=args.seed,
        database_path=database_path,
        temp_directory=temp_directory,
        memory_limit=args.duckdb_memory_limit,
    )
    log(f"Verification completed in {format_seconds(perf_counter() - verify_start)}")

    embed_start = perf_counter()
    log("Loading checkpoint and embedding sampled variants")
    inference = LatentInference.from_checkpoint(
        checkpoint_path=checkpoint_path,
        parquet_path=sampled_dedup_path,
        device=args.device,
    )
    log(f"Embedding device resolved to {inference.device}")
    inference.embed_parquet(
        output_path=embedding_path,
        parquet_path=sampled_dedup_path,
        id_columns=passthrough_columns,
        batch_size=args.embed_batch_size,
        scan_batch_rows=args.scan_batch_rows,
        threads=args.threads,
        progress_callback=lambda rows_written, batch_index: log(
            f"Embedding batch {batch_index}: wrote {rows_written:,} / {sampled_rows:,} rows"
        ),
    )
    log(f"Embedding parquet written to {embedding_path} in {format_seconds(perf_counter() - embed_start)}")
    latent_columns = [name for name in pq.read_schema(embedding_path).names if name.startswith("latent_")]

    fit_rows = min(args.umap_fit_rows, sampled_rows)
    plot_rows = min(args.plot_rows, sampled_rows)
    fit_sample_start = perf_counter()
    log(f"Sampling {fit_rows:,} latent rows for UMAP fitting")
    sample_parquet_exact(
        embedding_path,
        umap_fit_path,
        fit_rows,
        args.seed,
        args.threads,
        selected_columns=unique_columns(["row_index", *latent_columns]),
        database_path=database_path,
        temp_directory=temp_directory,
        memory_limit=args.duckdb_memory_limit,
    )
    log(f"UMAP fit sample written in {format_seconds(perf_counter() - fit_sample_start)}")

    fit_start = perf_counter()
    log("Fitting UMAP and HDBSCAN on the latent fit sample")
    umap_model, clusterer, fit_df, fit_coords = fit_umap_and_hdbscan(
        fit_sample_path=umap_fit_path,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        min_cluster_size=args.hdbscan_min_cluster_size,
        min_samples=args.hdbscan_min_samples,
        seed=args.seed,
    )
    fit_cluster_labels = fit_df["cluster_label"].cast(pl.Int32)
    fit_cluster_count = len({int(value) for value in fit_cluster_labels.to_list() if int(value) >= 0})
    fit_noise_count = int((fit_cluster_labels == -1).sum())
    log(
        f"UMAP/HDBSCAN fit completed in {format_seconds(perf_counter() - fit_start)}; "
        f"clusters={fit_cluster_count}, noise={fit_noise_count:,}"
    )
    predictor_state = save_predictor_state(
        output_dir=predictor_state_dir,
        umap_model=umap_model,
        clusterer=clusterer,
        fit_df=fit_df,
        umap_latent_columns = fit_coords
    )
    log(f"Saved predictor state to {predictor_state_dir}")
    fit_assignment_df = fit_df.select(["row_index", "cluster_label", "cluster_probability"]).sort("row_index")
    fit_assignment_rows = fit_assignment_df["row_index"].to_numpy()
    fit_assignment_labels = fit_assignment_df["cluster_label"].cast(pl.Int32).to_numpy()
    fit_assignment_strengths = fit_assignment_df["cluster_probability"].cast(pl.Float32).to_numpy()

    transform_start = perf_counter()
    log("Transforming all embeddings into UMAP space and assigning HDBSCAN labels")
    rows_written = transform_embeddings_to_analysis(
        embedding_path=embedding_path,
        analysis_path=analysis_path,
        umap_model=umap_model,
        clusterer=clusterer,
        fit_assignment_rows=fit_assignment_rows,
        fit_assignment_labels=fit_assignment_labels,
        fit_assignment_strengths=fit_assignment_strengths,
        passthrough_columns=passthrough_columns,
        scan_batch_rows=args.scan_batch_rows,
        threads=args.threads,
        progress_callback=lambda rows_written, batch_index: log(
            f"Analysis batch {batch_index}: wrote {rows_written:,} / {sampled_rows:,} rows"
        ),
    )
    if rows_written != sampled_rows:
        raise RuntimeError(f"Analysis parquet row count mismatch: expected {sampled_rows}, got {rows_written}")
    log(f"Analysis parquet written to {analysis_path} in {format_seconds(perf_counter() - transform_start)}")

    plot_sample_start = perf_counter()
    log(f"Sampling {plot_rows:,} rows for plotting")
    sample_parquet_exact(
        analysis_path,
        plot_sample_path,
        plot_rows,
        args.seed + 1,
        args.threads,
        selected_columns=unique_columns(["umap_1", "umap_2", "cluster_label", "cluster_probability", *color_columns]),
        database_path=database_path,
        temp_directory=temp_directory,
        memory_limit=args.duckdb_memory_limit,
    )
    plot_df = pl.read_parquet(plot_sample_path)
    log(f"Plot sample written in {format_seconds(perf_counter() - plot_sample_start)}")

    stats_start = perf_counter()
    log("Aggregating cluster statistics and trinucleotide distributions")
    cluster_stats = query_cluster_stats(
        analysis_path=analysis_path,
        threads=args.threads,
        database_path=database_path,
        temp_directory=temp_directory,
        memory_limit=args.duckdb_memory_limit,
    )
    trinuc192_counts = query_trinuc192_counts(
        analysis_path=analysis_path,
        cluster_stats=cluster_stats,
        threads=args.threads,
        database_path=database_path,
        temp_directory=temp_directory,
        memory_limit=args.duckdb_memory_limit,
    )
    log(
        f"Aggregated {cluster_stats.height:,} non-noise clusters in {format_seconds(perf_counter() - stats_start)}"
    )
    cluster_sizes = {
        int(row["cluster_label"]): int(row["cluster_size"])
        for row in cluster_stats.select(["cluster_label", "cluster_size"]).to_dicts()
    }

    plots_start = perf_counter()
    log("Rendering global UMAP plots")
    plot_items: list[tuple[str, Path]] = []
    for column in color_columns:
        output_path = plots_dir / f"umap_{column}.png"
        numeric_umap_plot(plot_df, column, output_path)
        plot_items.append((column, output_path))

    cluster_plot_path = plots_dir / "umap_clusters.png"
    cluster_umap_plot(plot_df, cluster_sizes, cluster_plot_path)
    plot_items.append(("clusters", cluster_plot_path))

    probability_plot_path = plots_dir / "umap_cluster_probability.png"
    probability_umap_plot(plot_df, probability_plot_path)
    plot_items.append(("cluster_probability", probability_plot_path))

    contact_sheet_path = plots_dir / "umap_contact_sheet.png"
    make_contact_sheet(plot_items, contact_sheet_path, columns=2)
    log(f"Global UMAP plots written in {format_seconds(perf_counter() - plots_start)}")

    profiles_start = perf_counter()
    log(f"Rendering {cluster_stats.height:,} per-cluster profile plots")
    cluster_manifest_path, trinuc192_path = write_cluster_profiles(
        output_dir=cluster_profiles_dir,
        plot_df=plot_df,
        analysis_path=analysis_path,
        cluster_stats=cluster_stats,
        trinuc192_counts=trinuc192_counts,
        threads=args.threads,
        database_path=database_path,
        temp_directory=temp_directory,
        memory_limit=args.duckdb_memory_limit,
    )
    log(f"Cluster profiles written in {format_seconds(perf_counter() - profiles_start)}")

    sigprof_start = perf_counter()
    log("Running UV-only SigProfilerAssignment on non-noise clusters")
    sigprofiler_paths = run_sigprofiler_assignment(
        trinuc192_counts=trinuc192_counts,
        output_root=output_root,
        genome_build=args.genome_build,
        cosmic_version=args.cosmic_version,
        cpu=args.sigprofiler_cpu,
    )
    log(f"SigProfilerAssignment completed in {format_seconds(perf_counter() - sigprof_start)}")

    summary_start = perf_counter()
    log("Building cluster summary outputs")
    cluster_summary_path = output_root / "cluster_summary.parquet"
    cluster_summary_json_path = output_root / "cluster_summary.json"
    cluster_summary = build_cluster_summary(
        cluster_stats=cluster_stats,
        sigprofiler_output_dir=Path(sigprofiler_paths["output_dir"]),
        output_path=cluster_summary_path,
        output_json_path=cluster_summary_json_path,
    )
    log(f"Cluster summary written in {format_seconds(perf_counter() - summary_start)}")

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "parquet_path": str(parquet_path),
        "output_root": str(output_root),
        "row_filter": row_filter,
        "seed": args.seed,
        "deterministic_sampling_threads": 1,
        "hdbscan_core_dist_n_jobs": 1,
        "use_all": args.use_all,
        "requested_sample_rows": requested_sample_rows,
        "sample_rows": sampled_rows,
        "deduplicated_population": dedup_population,
        "fit_rows": fit_rows,
        "plot_rows": plot_rows,
        "rows_written": rows_written,
        "sampled_dedup_path": str(sampled_dedup_path),
        "embedding_path": str(embedding_path),
        "umap_fit_path": str(umap_fit_path),
        "analysis_path": str(analysis_path),
        "plot_sample_path": str(plot_sample_path),
        "cluster_profiles_dir": str(cluster_profiles_dir),
        "cluster_profiles_manifest": str(cluster_manifest_path),
        "cluster_trinuc192_path": str(trinuc192_path),
        "predictor_state": predictor_state,
        "cluster_summary_path": str(cluster_summary_path),
        "cluster_summary_json_path": str(cluster_summary_json_path),
        "cluster_count_excluding_noise": int(cluster_summary.height),
        "noise_count": query_noise_count(
            analysis_path=analysis_path,
            threads=args.threads,
            database_path=database_path,
            temp_directory=temp_directory,
            memory_limit=args.duckdb_memory_limit,
        ),
        "plot_contact_sheet": str(contact_sheet_path),
        "sigprofiler": sigprofiler_paths,
        "color_columns": color_columns,
    }
    write_pipeline_summary(output_root / "pipeline_summary.json", summary)
    log(f"Pipeline completed in {format_seconds(perf_counter() - overall_start)}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
