from __future__ import annotations

import argparse
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from uv_vae.data import connect_duckdb

CANONICAL_CHANGES = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
SBS96_CONTEXTS = [f"{left}{right}" for left in "ACGT" for right in "ACGT"]
CONTEXT_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#393b79",
    "#637939",
    "#8c6d31",
    "#843c39",
    "#7b4173",
    "#3182bd",
]


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


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def resolve_output_path(output_path: str | None) -> Path:
    if output_path is not None:
        target = Path(output_path).expanduser()
        return target if target.is_absolute() else (Path.cwd() / target).resolve()
    return (Path.cwd() / f"trinuc96_snvq_histograms_{utc_timestamp()}.html").resolve()


def complement_sql(expr: str) -> str:
    return f"""CASE {expr}
        WHEN 'A' THEN 'T'
        WHEN 'C' THEN 'G'
        WHEN 'G' THEN 'C'
        WHEN 'T' THEN 'A'
        ELSE NULL
    END"""


def build_base_ctes(
    parquet_path: Path,
    row_filter: str,
    limit: int | None,
    min_snvq: float | None = None,
    max_snvq: float | None = None,
) -> str:
    limit_clause = f"\n        LIMIT {int(limit)}" if limit is not None else ""
    ref_comp = complement_sql("ref_base")
    alt_comp = complement_sql("alt_base")
    prev_comp = complement_sql("prev_base")
    next_comp = complement_sql("next_base")
    change_list = ", ".join(sql_quote(change) for change in CANONICAL_CHANGES)
    windowed_cte = ""
    if min_snvq is not None and max_snvq is not None:
        windowed_cte = f""",
        windowed AS (
            SELECT *
            FROM canonical_only
            WHERE snvq >= {float(min_snvq):.17g}
              AND snvq <= {float(max_snvq):.17g}
        )"""
    return f"""
        WITH filtered AS (
            SELECT
                upper(CAST("REF" AS VARCHAR)) AS ref_base,
                upper(CAST("ALT" AS VARCHAR)) AS alt_base,
                upper(CAST("X_PREV1" AS VARCHAR)) AS prev_base,
                upper(CAST("X_NEXT1" AS VARCHAR)) AS next_base,
                CAST("SNVQ" AS DOUBLE) AS snvq
            FROM read_parquet({sql_quote(str(parquet_path))})
            WHERE {row_filter}{limit_clause}
        ),
        valid AS (
            SELECT *
            FROM filtered
            WHERE snvq IS NOT NULL
              AND ref_base IN ('A', 'C', 'G', 'T')
              AND alt_base IN ('A', 'C', 'G', 'T')
              AND prev_base IN ('A', 'C', 'G', 'T')
              AND next_base IN ('A', 'C', 'G', 'T')
              AND ref_base <> alt_base
        ),
        canonicalized AS (
            SELECT
                CASE WHEN ref_base IN ('C', 'T') THEN ref_base ELSE {ref_comp} END AS canonical_ref,
                CASE WHEN ref_base IN ('C', 'T') THEN alt_base ELSE {alt_comp} END AS canonical_alt,
                CASE WHEN ref_base IN ('C', 'T') THEN prev_base ELSE {next_comp} END AS canonical_prev,
                CASE WHEN ref_base IN ('C', 'T') THEN next_base ELSE {prev_comp} END AS canonical_next,
                snvq
            FROM valid
        ),
        canonical_only AS (
            SELECT
                canonical_ref || '>' || canonical_alt AS central_change,
                canonical_prev || canonical_next AS context16,
                canonical_prev || '[' || canonical_ref || '>' || canonical_alt || ']' || canonical_next AS sbs96,
                snvq
            FROM canonicalized
            WHERE canonical_ref || '>' || canonical_alt IN ({change_list})
        ){windowed_cte}
    """


def query_stats(
    parquet_path: Path,
    row_filter: str,
    limit: int | None,
    min_snvq: float,
    max_snvq: float,
    threads: int,
    memory_limit: str,
    temp_directory: Path,
) -> dict[str, float | int]:
    sql = f"""
        {build_base_ctes(parquet_path, row_filter, limit, min_snvq=min_snvq, max_snvq=max_snvq)}
        SELECT
            (SELECT COUNT(*) FROM filtered) AS filtered_rows,
            (SELECT COUNT(*) FROM valid) AS valid_rows,
            (SELECT COUNT(*) FROM canonical_only) AS canonical_rows,
            COUNT(*) AS plotted_rows,
            (SELECT MIN(snvq) FROM canonical_only) AS observed_min_snvq,
            (SELECT MAX(snvq) FROM canonical_only) AS observed_max_snvq
        FROM windowed
    """
    with connect_duckdb(threads=threads, temp_directory=temp_directory, memory_limit=memory_limit) as conn:
        row = conn.execute(sql).fetchone()
    if row is None or row[2] is None or int(row[2]) == 0:
        raise RuntimeError(f"No canonical SBS96 SNVQ rows matched the filter: {row_filter}")
    if row[3] is None or int(row[3]) == 0:
        raise RuntimeError(
            f"No canonical SBS96 SNVQ rows fell within the requested SNVQ window [{min_snvq}, {max_snvq}]"
        )
    return {
        "filtered_rows": int(row[0]),
        "valid_rows": int(row[1]),
        "canonical_rows": int(row[2]),
        "plotted_rows": int(row[3]),
        "observed_min_snvq": float(row[4]),
        "observed_max_snvq": float(row[5]),
    }


def build_bin_edges(min_snvq: float, max_snvq: float, num_bins: int) -> np.ndarray:
    if not math.isfinite(min_snvq) or not math.isfinite(max_snvq):
        raise RuntimeError("SNVQ range contains non-finite values")
    if max_snvq <= min_snvq:
        raise ValueError("--max-snvq must be greater than --min-snvq")
    return np.linspace(min_snvq, max_snvq, num_bins + 1, dtype=float)


def query_histogram_counts(
    parquet_path: Path,
    row_filter: str,
    limit: int | None,
    bin_edges: np.ndarray,
    threads: int,
    memory_limit: str,
    temp_directory: Path,
) -> pl.DataFrame:
    effective_bins = len(bin_edges) - 1
    min_snvq = float(bin_edges[0])
    max_snvq = float(bin_edges[-1])
    bin_width = float(bin_edges[1] - bin_edges[0])
    if effective_bins == 1:
        bin_expr = "0"
    else:
        bin_expr = (
            f"LEAST({effective_bins - 1}, GREATEST(0, "
            f"CAST(FLOOR((snvq - {min_snvq:.17g}) / {bin_width:.17g}) AS BIGINT)))"
        )
    sql = f"""
        {build_base_ctes(parquet_path, row_filter, limit, min_snvq=min_snvq, max_snvq=max_snvq)}
        SELECT
            central_change,
            context16,
            CAST({bin_expr} AS INTEGER) AS bin_index,
            COUNT(*) AS count
        FROM windowed
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, 3
    """
    with connect_duckdb(threads=threads, temp_directory=temp_directory, memory_limit=memory_limit) as conn:
        return conn.execute(sql).pl()


def build_figure(
    histogram_counts: pl.DataFrame,
    bin_edges: np.ndarray,
    parquet_path: Path,
    row_filter: str,
    stats: dict[str, float | int],
) -> go.Figure:
    effective_bins = len(bin_edges) - 1
    bin_starts = bin_edges[:-1]
    bin_ends = bin_edges[1:]
    bin_centers = (bin_starts + bin_ends) / 2.0
    count_lookup = {
        (row["central_change"], row["context16"], int(row["bin_index"])): int(row["count"])
        for row in histogram_counts.to_dicts()
    }
    figure = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=CANONICAL_CHANGES,
        horizontal_spacing=0.07,
        vertical_spacing=0.10,
    )
    for change_index, change in enumerate(CANONICAL_CHANGES):
        row = change_index // 2 + 1
        col = change_index % 2 + 1
        for context_index, context16 in enumerate(SBS96_CONTEXTS):
            y_values = [count_lookup.get((change, context16, bin_index), 0) for bin_index in range(effective_bins)]
            customdata = np.column_stack([bin_starts, bin_ends])
            figure.add_trace(
                go.Scatter(
                    x=bin_centers,
                    y=y_values,
                    mode="lines",
                    line={"color": CONTEXT_COLORS[context_index], "width": 1.6},
                    name=context16,
                    legendgroup=context16,
                    showlegend=change_index == 0,
                    customdata=customdata,
                    hovertemplate=(
                        f"Change={change}<br>"
                        f"Context={context16}<br>"
                        "SNVQ bin=%{customdata[0]:.3f} to %{customdata[1]:.3f}<br>"
                        "Count=%{y}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
        figure.update_xaxes(title_text="SNVQ", row=row, col=col)
        figure.update_yaxes(title_text="Count", row=row, col=col)
    dropped_rows = int(stats["filtered_rows"]) - int(stats["canonical_rows"])
    outside_window_rows = int(stats["canonical_rows"]) - int(stats["plotted_rows"])
    subtitle = (
        f"parquet={parquet_path.name} | filtered_rows={int(stats['filtered_rows']):,} | "
        f"canonical_rows={int(stats['canonical_rows']):,} | plotted_rows={int(stats['plotted_rows']):,} | "
        f"dropped_invalid_or_non_snv={dropped_rows:,} | outside_window={outside_window_rows:,}"
        f"<br>snvq_window=[{bin_edges[0]:.3f}, {bin_edges[-1]:.3f}] | row_filter={row_filter}"
    )
    figure.update_layout(
        template="plotly_white",
        title={"text": f"SNVQ Histograms by Canonical SBS96 Context<br><sup>{subtitle}</sup>", "x": 0.5},
        width=1600,
        height=1200,
        hovermode="closest",
        legend={
            "title": {"text": "Context16"},
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.10,
            "xanchor": "center",
            "x": 0.5,
        },
        margin={"t": 120, "b": 150, "l": 70, "r": 30},
    )
    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot SNVQ histograms for canonical SBS96 trinucleotide contexts without locus aggregation."
    )
    parser.add_argument("--parquet-path", required=True, help="Path to the input parquet feature map")
    parser.add_argument("--row-filter", required=True, help="DuckDB SQL filter applied before histogramming")
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output HTML path. Defaults to ./trinuc96_snvq_histograms_<timestamp>.html",
    )
    parser.add_argument("--num-bins", type=int, default=100, help="Number of SNVQ histogram bins")
    parser.add_argument("--min-snvq", type=float, default=40.0, help="Minimum SNVQ included in the histogram window")
    parser.add_argument("--max-snvq", type=float, default=65.0, help="Maximum SNVQ included in the histogram window")
    parser.add_argument("--threads", type=int, default=12, help="DuckDB thread count")
    parser.add_argument(
        "--duckdb-memory-limit",
        default="4GB",
        help="Memory limit passed to DuckDB so the grouped query can spill to disk",
    )
    parser.add_argument(
        "--duckdb-temp-dir",
        default=None,
        help="Optional DuckDB temp directory. Defaults to <output_dir>/duckdb_tmp",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit applied after filtering, intended for smoke testing only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parquet_path = Path(args.parquet_path).expanduser().resolve()
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    output_path = resolve_output_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_directory = (
        Path(args.duckdb_temp_dir).expanduser().resolve()
        if args.duckdb_temp_dir is not None
        else output_path.parent / "duckdb_tmp"
    )
    if args.num_bins <= 0:
        raise ValueError("--num-bins must be positive")
    if args.max_snvq <= args.min_snvq:
        raise ValueError("--max-snvq must be greater than --min-snvq")
    start = perf_counter()
    limit_label = "none" if args.limit is None else f"{args.limit:,}"
    log(
        f"Starting SNVQ histogram export parquet={parquet_path} output={output_path} "
        f"num_bins={args.num_bins} snvq_window=[{args.min_snvq:.3f}, {args.max_snvq:.3f}] limit={limit_label}"
    )
    stats_start = perf_counter()
    stats = query_stats(
        parquet_path=parquet_path,
        row_filter=args.row_filter,
        limit=args.limit,
        min_snvq=args.min_snvq,
        max_snvq=args.max_snvq,
        threads=args.threads,
        memory_limit=args.duckdb_memory_limit,
        temp_directory=temp_directory,
    )
    log(
        f"Resolved filtered_rows={int(stats['filtered_rows']):,}, canonical_rows={int(stats['canonical_rows']):,}, "
        f"plotted_rows={int(stats['plotted_rows']):,}, "
        f"observed_snvq_range=[{float(stats['observed_min_snvq']):.3f}, {float(stats['observed_max_snvq']):.3f}] "
        f"in {format_seconds(perf_counter() - stats_start)}"
    )
    bin_edges = build_bin_edges(args.min_snvq, args.max_snvq, args.num_bins)
    counts_start = perf_counter()
    histogram_counts = query_histogram_counts(
        parquet_path=parquet_path,
        row_filter=args.row_filter,
        limit=args.limit,
        bin_edges=bin_edges,
        threads=args.threads,
        memory_limit=args.duckdb_memory_limit,
        temp_directory=temp_directory,
    )
    log(
        f"Aggregated histogram counts to {histogram_counts.height:,} grouped rows "
        f"in {format_seconds(perf_counter() - counts_start)}"
    )
    figure = build_figure(
        histogram_counts=histogram_counts,
        bin_edges=bin_edges,
        parquet_path=parquet_path,
        row_filter=args.row_filter,
        stats=stats,
    )
    figure.write_html(output_path, include_plotlyjs=True, full_html=True)
    log(f"Wrote interactive HTML to {output_path} in {format_seconds(perf_counter() - start)}")


if __name__ == "__main__":
    main()
