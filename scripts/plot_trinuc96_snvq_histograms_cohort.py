"""Cohort-scale SNVQ histograms by canonical SBS96 trinucleotide context.

Same figure as ``plot_trinuc96_snvq_histograms.py`` (six subplots, one per canonical base
change, sixteen flanking-context lines each) but built to run over the whole 95-sample /
157 M-row cohort in one command instead of a single parquet.

What it adds over the single-file script:

* **Many inputs.** ``--parquet-path`` takes a file, a directory, a quoted glob, or may be
  repeated. Sources are de-duplicated and sorted.
* **Batched scanning with progress.** Files are processed in batches so a long run reports
  where it is, and so peak memory does not scale with cohort size.
* **Resumable.** With ``--cache-dir`` each batch's aggregated counts are written to disk
  and reused on a re-run, so an interrupted job restarts from the last finished batch.
* **Honest SNVQ windowing.** The window defaults to the observed data range instead of a
  hard-coded ``[40, 65]``. Pinning a window logs exactly how many rows it discards.
* **Preflight.** ``--stats-only`` runs just the counting pass and prints the observed
  range, so you can pin the window before paying for the histogram pass.

Typical use on miletus::

    # preflight: counts + observed SNVQ range, no plot
    python plot_trinuc96_snvq_histograms_cohort.py \
        --parquet-path '/data/lab/ppmseq_parquets/*.parquet' --threads 48 --stats-only

    # full cohort figure, window pinned from the preflight output
    python plot_trinuc96_snvq_histograms_cohort.py \
        --parquet-path '/data/lab/ppmseq_parquets/*.parquet' \
        --threads 48 --min-snvq 30 --max-snvq 80 \
        --cache-dir ./trinuc96_cache --output-path trinuc96_snvq_cohort.html

Quote globs so the shell does not expand them into separate arguments.

Performance note: this workload is a parquet scan feeding a tiny grouped aggregate (at
most 6 x 16 x num_bins groups). It is dominated by decoding parquet, not by arithmetic,
so it scales with DuckDB threads and storage bandwidth. See ``--threads``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

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

DEFAULT_ROW_FILTER = "st = 'MIXED' AND et = 'MIXED' AND FILT = 1"


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


# --------------------------------------------------------------------------- inputs


def resolve_sources(raw_paths: list[str]) -> list[Path]:
    """Expand files, directories and globs into a de-duplicated, sorted file list."""
    resolved: list[Path] = []
    seen: set[Path] = set()
    for raw in raw_paths:
        candidate = Path(raw).expanduser()
        if any(char in raw for char in "*?["):
            anchor = candidate.parent
            pattern = candidate.name
            while any(char in str(anchor) for char in "*?["):
                pattern = f"{anchor.name}/{pattern}"
                anchor = anchor.parent
            matches = sorted(p for p in anchor.glob(pattern) if p.is_file())
            if not matches:
                raise FileNotFoundError(f"Glob matched no files: {raw}")
        elif candidate.is_dir():
            matches = sorted(p for p in candidate.glob("*.parquet") if p.is_file())
            if not matches:
                raise FileNotFoundError(f"No *.parquet files in directory: {raw}")
        elif candidate.is_file():
            matches = [candidate]
        else:
            raise FileNotFoundError(f"Parquet path not found: {raw}")
        for match in matches:
            key = match.resolve()
            if key not in seen:
                seen.add(key)
                resolved.append(key)
    return resolved


def batched(items: list[Path], size: int) -> list[list[Path]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def source_sql(sources: list[Path]) -> str:
    if len(sources) == 1:
        return sql_quote(str(sources[0]))
    return "[" + ", ".join(sql_quote(str(path)) for path in sources) + "]"


def resolve_output_path(output_path: str | None) -> Path:
    if output_path is not None:
        target = Path(output_path).expanduser()
        return target if target.is_absolute() else (Path.cwd() / target).resolve()
    return (Path.cwd() / f"trinuc96_snvq_histograms_cohort_{utc_timestamp()}.html").resolve()


# --------------------------------------------------------------------------- sql


def complement_sql(expr: str) -> str:
    return f"""CASE {expr}
        WHEN 'A' THEN 'T'
        WHEN 'C' THEN 'G'
        WHEN 'G' THEN 'C'
        WHEN 'T' THEN 'A'
        ELSE NULL
    END"""


def build_base_ctes(sources: list[Path], row_filter: str, limit: int | None) -> str:
    """CTE chain ending in ``canonical_only``.

    Matches the single-file script's canonicalisation exactly: rows are normalised to the
    pyrimidine strand, and when the reference is a purine the flanking bases are both
    complemented *and* swapped (the base 5' of a purine ref is 3' of its pyrimidine
    equivalent).
    """
    limit_clause = f"\n        LIMIT {int(limit)}" if limit is not None else ""
    ref_comp = complement_sql("ref_base")
    alt_comp = complement_sql("alt_base")
    prev_comp = complement_sql("prev_base")
    next_comp = complement_sql("next_base")
    change_list = ", ".join(sql_quote(change) for change in CANONICAL_CHANGES)
    return f"""
        WITH filtered AS (
            SELECT
                upper(CAST("REF" AS VARCHAR)) AS ref_base,
                upper(CAST("ALT" AS VARCHAR)) AS alt_base,
                upper(CAST("X_PREV1" AS VARCHAR)) AS prev_base,
                upper(CAST("X_NEXT1" AS VARCHAR)) AS next_base,
                CAST("SNVQ" AS DOUBLE) AS snvq
            FROM read_parquet({source_sql(sources)})
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
                snvq
            FROM canonicalized
            WHERE canonical_ref || '>' || canonical_alt IN ({change_list})
        )
    """


def open_connection(threads: int, memory_limit: str, temp_directory: Path):
    # preserve_insertion_order is off deliberately: every query here is an order-independent
    # aggregate, and disabling it is a large-scan memory optimisation.
    return connect_duckdb(
        threads=threads,
        temp_directory=temp_directory,
        memory_limit=memory_limit,
        preserve_insertion_order=False,
    )


def query_batch_stats(
    sources: list[Path],
    row_filter: str,
    limit: int | None,
    threads: int,
    memory_limit: str,
    temp_directory: Path,
) -> dict[str, float | int | None]:
    """Stage counts and observed SNVQ range for one batch, in a single scan."""
    sql = f"""
        {build_base_ctes(sources, row_filter, limit)}
        SELECT
            (SELECT COUNT(*) FROM filtered) AS filtered_rows,
            (SELECT COUNT(*) FROM valid) AS valid_rows,
            COUNT(*) AS canonical_rows,
            MIN(snvq) AS observed_min_snvq,
            MAX(snvq) AS observed_max_snvq
        FROM canonical_only
    """
    with open_connection(threads, memory_limit, temp_directory) as conn:
        row = conn.execute(sql).fetchone()
    return {
        "filtered_rows": int(row[0] or 0),
        "valid_rows": int(row[1] or 0),
        "canonical_rows": int(row[2] or 0),
        "observed_min_snvq": None if row[3] is None else float(row[3]),
        "observed_max_snvq": None if row[4] is None else float(row[4]),
    }


def build_bin_edges(min_snvq: float, max_snvq: float, num_bins: int) -> np.ndarray:
    if not math.isfinite(min_snvq) or not math.isfinite(max_snvq):
        raise RuntimeError("SNVQ range contains non-finite values")
    if max_snvq <= min_snvq:
        raise ValueError(
            f"SNVQ window is empty: max ({max_snvq}) must be greater than min ({min_snvq})"
        )
    return np.linspace(min_snvq, max_snvq, num_bins + 1, dtype=float)


def query_batch_histogram(
    sources: list[Path],
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
        {build_base_ctes(sources, row_filter, limit)}
        SELECT
            central_change,
            context16,
            CAST({bin_expr} AS INTEGER) AS bin_index,
            COUNT(*) AS count
        FROM canonical_only
        WHERE snvq >= {min_snvq:.17g}
          AND snvq <= {max_snvq:.17g}
        GROUP BY 1, 2, 3
    """
    with open_connection(threads, memory_limit, temp_directory) as conn:
        return conn.execute(sql).pl()


# --------------------------------------------------------------------------- driver


def batch_key(
    batch: list[Path],
    row_filter: str,
    bin_edges: np.ndarray,
    limit: int | None,
) -> str:
    """Stable short identifier for a cached batch result.

    The key covers everything that changes the counts -- the files, the row filter, the
    bin edges and the row limit -- so that changing ``--num-bins``, the SNVQ window or the
    filter cannot silently reuse a stale cache entry.
    """
    import hashlib

    payload = "\n".join(
        [
            *(str(p) for p in batch),
            f"row_filter={row_filter}",
            f"bins={len(bin_edges) - 1}",
            f"min={float(bin_edges[0]):.17g}",
            f"max={float(bin_edges[-1]):.17g}",
            f"limit={limit}",
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def run_stats_pass(
    batches: list[list[Path]],
    row_filter: str,
    limit: int | None,
    threads: int,
    memory_limit: str,
    temp_directory: Path,
) -> dict[str, float | int]:
    totals = {"filtered_rows": 0, "valid_rows": 0, "canonical_rows": 0}
    observed_min: float | None = None
    observed_max: float | None = None
    start = perf_counter()
    for index, batch in enumerate(batches, start=1):
        batch_start = perf_counter()
        stats = query_batch_stats(
            batch, row_filter, limit, threads, memory_limit, temp_directory
        )
        for key in totals:
            totals[key] += int(stats[key])
        if stats["observed_min_snvq"] is not None:
            observed_min = (
                stats["observed_min_snvq"]
                if observed_min is None
                else min(observed_min, stats["observed_min_snvq"])
            )
            observed_max = (
                stats["observed_max_snvq"]
                if observed_max is None
                else max(observed_max, stats["observed_max_snvq"])
            )
        log(
            f"  [stats {index}/{len(batches)}] {len(batch)} file(s), "
            f"canonical={int(stats['canonical_rows']):,}, "
            f"running_total={totals['canonical_rows']:,} "
            f"({format_seconds(perf_counter() - batch_start)})"
        )
    if totals["filtered_rows"] == 0:
        raise RuntimeError(f"No rows matched the row filter: {row_filter}")
    if totals["canonical_rows"] == 0:
        raise RuntimeError(
            "No canonical SBS96 SNVQ rows survived validation. "
            f"filtered_rows={totals['filtered_rows']:,}, valid_rows={totals['valid_rows']:,}"
        )
    log(f"Stats pass finished in {format_seconds(perf_counter() - start)}")
    return {
        **totals,
        "observed_min_snvq": float(observed_min),
        "observed_max_snvq": float(observed_max),
    }


def run_histogram_pass(
    batches: list[list[Path]],
    row_filter: str,
    limit: int | None,
    bin_edges: np.ndarray,
    threads: int,
    memory_limit: str,
    temp_directory: Path,
    cache_dir: Path | None,
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    start = perf_counter()
    for index, batch in enumerate(batches, start=1):
        cache_path = (
            cache_dir / f"batch_{batch_key(batch, row_filter, bin_edges, limit)}.parquet"
            if cache_dir
            else None
        )
        if cache_path is not None and cache_path.is_file():
            frame = pl.read_parquet(cache_path)
            log(f"  [hist {index}/{len(batches)}] reused cache {cache_path.name}")
        else:
            batch_start = perf_counter()
            frame = query_batch_histogram(
                batch, row_filter, limit, bin_edges, threads, memory_limit, temp_directory
            )
            if cache_path is not None:
                frame.write_parquet(cache_path)
            log(
                f"  [hist {index}/{len(batches)}] {len(batch)} file(s), "
                f"{frame.height:,} groups, {int(frame['count'].sum()) if frame.height else 0:,} rows "
                f"({format_seconds(perf_counter() - batch_start)})"
            )
        frames.append(frame)
    combined = pl.concat(frames) if frames else pl.DataFrame()
    if combined.height:
        combined = (
            combined.group_by(["central_change", "context16", "bin_index"])
            .agg(pl.col("count").sum())
            .sort(["central_change", "context16", "bin_index"])
        )
    log(f"Histogram pass finished in {format_seconds(perf_counter() - start)}")
    return combined


# --------------------------------------------------------------------------- figure


def build_figure(
    histogram_counts: pl.DataFrame,
    bin_edges: np.ndarray,
    sources: list[Path],
    row_filter: str,
    stats: dict[str, float | int] | None,
    plotted_rows: int,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

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
            y_values = [
                count_lookup.get((change, context16, bin_index), 0)
                for bin_index in range(effective_bins)
            ]
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

    source_label = (
        sources[0].name
        if len(sources) == 1
        else f"{len(sources)} parquet files ({sources[0].parent})"
    )
    if stats is None:
        # Counting pass was skipped (window pinned). Say so rather than printing numbers
        # derived from the plotted rows as though they were cohort-wide counts.
        accounting = (
            f"plotted_rows={plotted_rows:,} | "
            "filtered/canonical/outside_window not measured (counting pass skipped; "
            "re-run with --stats-only for cohort counts)"
        )
        observed = "observed_snvq_range=not measured"
    else:
        dropped_rows = int(stats["filtered_rows"]) - int(stats["canonical_rows"])
        outside_window_rows = int(stats["canonical_rows"]) - plotted_rows
        accounting = (
            f"filtered_rows={int(stats['filtered_rows']):,} | "
            f"canonical_rows={int(stats['canonical_rows']):,} | "
            f"plotted_rows={plotted_rows:,} | "
            f"dropped_invalid_or_non_snv={dropped_rows:,} | "
            f"outside_window={outside_window_rows:,}"
        )
        observed = (
            f"observed_snvq_range=[{float(stats['observed_min_snvq']):.3f}, "
            f"{float(stats['observed_max_snvq']):.3f}]"
        )
    subtitle = (
        f"source={source_label} | {accounting}"
        f"<br>snvq_window=[{bin_edges[0]:.3f}, {bin_edges[-1]:.3f}] | {observed} | "
        f"row_filter={row_filter}"
    )
    figure.update_layout(
        template="plotly_white",
        title={
            "text": f"SNVQ Histograms by Canonical SBS96 Context<br><sup>{subtitle}</sup>",
            "x": 0.5,
        },
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


# --------------------------------------------------------------------------- cli


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cohort-scale SNVQ histograms by canonical SBS96 trinucleotide context. "
            "Accepts many parquets and scans them in batches."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "--parquet-path accepts a file, a directory, a quoted glob, or may be repeated.\n"
            "Quote globs so the shell does not expand them first."
        ),
    )
    parser.add_argument(
        "--parquet-path",
        required=True,
        action="append",
        dest="parquet_paths",
        metavar="PATH",
        help="Parquet file, directory, or quoted glob. Repeat to combine sources.",
    )
    parser.add_argument(
        "--row-filter",
        default=DEFAULT_ROW_FILTER,
        help=f"DuckDB SQL filter applied before histogramming (default: {DEFAULT_ROW_FILTER!r})",
    )
    parser.add_argument("--output-path", default=None, help="Output HTML path")
    parser.add_argument("--num-bins", type=int, default=100, help="Number of SNVQ histogram bins")
    parser.add_argument(
        "--min-snvq",
        type=float,
        default=None,
        help="Minimum SNVQ included. Defaults to the observed minimum.",
    )
    parser.add_argument(
        "--max-snvq",
        type=float,
        default=None,
        help="Maximum SNVQ included. Defaults to the observed maximum.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=48,
        help="DuckDB thread count (default 48; this workload is scan-bound so more helps)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Parquet files scanned per query. Larger = better parallelism, coarser progress.",
    )
    parser.add_argument(
        "--duckdb-memory-limit",
        default="16GB",
        help="Memory limit passed to DuckDB so the grouped query can spill to disk",
    )
    parser.add_argument(
        "--duckdb-temp-dir",
        default=None,
        help="DuckDB temp directory. Defaults to <output_dir>/duckdb_tmp",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory for per-batch histogram counts. Makes an interrupted run resumable.",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Run only the counting pass; print counts and observed SNVQ range, write no plot.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Row limit applied per batch after filtering. Smoke testing only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.stats_only:
        try:
            import plotly  # noqa: F401
        except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
            raise SystemExit(
                "plotly is required to write the figure but is not installed in the active "
                "environment. Install it (e.g. `micromamba install -n uv_vae plotly`), or "
                "re-run with --stats-only to get the counts without a plot."
            ) from exc

    sources = resolve_sources(args.parquet_paths)
    output_path = resolve_output_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_directory = (
        Path(args.duckdb_temp_dir).expanduser().resolve()
        if args.duckdb_temp_dir is not None
        else output_path.parent / "duckdb_tmp"
    )
    cache_dir = None
    if args.cache_dir is not None:
        cache_dir = Path(args.cache_dir).expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)

    if args.num_bins <= 0:
        raise ValueError("--num-bins must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.min_snvq is not None and args.max_snvq is not None and args.max_snvq <= args.min_snvq:
        raise ValueError("--max-snvq must be greater than --min-snvq")

    batches = batched(sources, args.batch_size)
    start = perf_counter()
    log(
        f"Resolved {len(sources)} parquet file(s) into {len(batches)} batch(es) "
        f"of up to {args.batch_size}; threads={args.threads}"
    )
    if len(sources) <= 5:
        for path in sources:
            log(f"  {path}")
    else:
        log(f"  first: {sources[0]}")
        log(f"  last:  {sources[-1]}")

    pinned = args.min_snvq is not None and args.max_snvq is not None
    stats: dict[str, float | int] | None = None
    if not pinned or args.stats_only:
        log("Counting pass (stage counts + observed SNVQ range) ...")
        stats = run_stats_pass(
            batches,
            args.row_filter,
            args.limit,
            args.threads,
            args.duckdb_memory_limit,
            temp_directory,
        )
        log(
            f"filtered_rows={int(stats['filtered_rows']):,}, "
            f"valid_rows={int(stats['valid_rows']):,}, "
            f"canonical_rows={int(stats['canonical_rows']):,}, "
            f"observed_snvq_range=[{float(stats['observed_min_snvq']):.3f}, "
            f"{float(stats['observed_max_snvq']):.3f}]"
        )
        if args.stats_only:
            summary = {
                "sources": len(sources),
                "row_filter": args.row_filter,
                **{k: (float(v) if isinstance(v, float) else int(v)) for k, v in stats.items()},
            }
            print(json.dumps(summary, indent=2))
            log(
                "Pin the window on the next run with "
                f"--min-snvq {float(stats['observed_min_snvq']):.3f} "
                f"--max-snvq {float(stats['observed_max_snvq']):.3f} "
                "to skip this pass."
            )
            return

    if pinned:
        min_snvq, max_snvq = float(args.min_snvq), float(args.max_snvq)
        log(f"Using pinned SNVQ window [{min_snvq:.3f}, {max_snvq:.3f}] (counting pass skipped)")
    else:
        assert stats is not None
        min_snvq = (
            float(args.min_snvq) if args.min_snvq is not None else float(stats["observed_min_snvq"])
        )
        max_snvq = (
            float(args.max_snvq) if args.max_snvq is not None else float(stats["observed_max_snvq"])
        )
        log(f"Using SNVQ window [{min_snvq:.3f}, {max_snvq:.3f}]")

    bin_edges = build_bin_edges(min_snvq, max_snvq, args.num_bins)

    log("Histogram pass ...")
    histogram_counts = run_histogram_pass(
        batches,
        args.row_filter,
        args.limit,
        bin_edges,
        args.threads,
        args.duckdb_memory_limit,
        temp_directory,
        cache_dir,
    )
    plotted_rows = int(histogram_counts["count"].sum()) if histogram_counts.height else 0
    if plotted_rows == 0:
        raise RuntimeError(
            f"No canonical SBS96 SNVQ rows fell within the SNVQ window "
            f"[{min_snvq:.3f}, {max_snvq:.3f}]."
        )
    log(f"Combined to {histogram_counts.height:,} groups covering {plotted_rows:,} rows")

    if stats is None:
        log(
            "NOTE: counting pass was skipped because the window was pinned, so cohort "
            "row-accounting is reported as 'not measured'. Run --stats-only for true counts."
        )
    else:
        outside_window = int(stats["canonical_rows"]) - plotted_rows
        if outside_window > 0:
            pct = outside_window / int(stats["canonical_rows"]) * 100.0
            log(
                f"WARNING: {outside_window:,} canonical rows ({pct:.1f}%) fall outside the "
                f"SNVQ window [{min_snvq:.3f}, {max_snvq:.3f}] and are not plotted."
            )

    figure = build_figure(
        histogram_counts=histogram_counts,
        bin_edges=bin_edges,
        sources=sources,
        row_filter=args.row_filter,
        stats=stats,
        plotted_rows=plotted_rows,
    )
    figure.write_html(output_path, include_plotlyjs=True, full_html=True)
    log(f"Wrote interactive HTML to {output_path} in {format_seconds(perf_counter() - start)}")


if __name__ == "__main__":
    main()
