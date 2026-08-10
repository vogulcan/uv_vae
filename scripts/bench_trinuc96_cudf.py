"""Benchmark: DuckDB vs pandas vs cuDF for the SBS96 x SNVQ histogram aggregate.

Answers two questions at once for the workload in
``plot_trinuc96_snvq_histograms_cohort.py``:

1. **Is it correct?** The dataframe path must reproduce DuckDB's per-group counts
   *exactly*. Any mismatch is printed as a diff, not summarised away.
2. **Is it faster?** Read time and compute time are timed separately, because the
   claim under test is that this workload is bound by parquet decode rather than by
   arithmetic. If that is true, the GPU wins little even when its compute is ~free.

Three backends:

* ``duckdb`` -- the reference. Reuses the cohort script's SQL verbatim.
* ``pandas`` -- CPU control. Lets you validate the dataframe logic on a laptop with no
  GPU before running anything on miletus.
* ``cudf``   -- RAPIDS GPU. Same code path as pandas (cuDF mirrors the pandas API for
  every operation used here).

Typical use::

    # local, no GPU: prove the dataframe logic matches DuckDB
    python bench_trinuc96_cudf.py --parquet-path some.parquet \\
        --backends duckdb,pandas --limit 2000000

    # on miletus: the real comparison
    python bench_trinuc96_cudf.py \\
        --parquet-path '/data/lab/ppmseq_parquets/*.parquet' --files 4 \\
        --backends duckdb,cudf --threads 48

**The row-filter caveat, which is the actual point of the experiment.** DuckDB accepts
arbitrary SQL in ``--row-filter``. cuDF has no SQL layer, so this script ships a small
translator that handles ``AND``-joined ``column <op> literal`` clauses -- enough for the
project default ``st = 'MIXED' AND et = 'MIXED' AND FILT = 1``. Anything more complex
(``OR``, ``IN``, ``LIKE``, arithmetic, subqueries) is *refused* rather than silently
mistranslated. That refusal is the finding: a GPU path either constrains the filter
language or reintroduces DuckDB to do the filtering, in which case the scan cost -- the
thing that dominates -- is paid anyway.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from plot_trinuc96_snvq_histograms_cohort import (  # noqa: E402
    CANONICAL_CHANGES,
    DEFAULT_ROW_FILTER,
    build_bin_edges,
    log,
    query_batch_histogram,
    query_batch_stats,
    resolve_sources,
)

COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}
BASES = ["A", "C", "G", "T"]
NEEDED_COLUMNS = ["REF", "ALT", "X_PREV1", "X_NEXT1", "SNVQ"]

# --------------------------------------------------------------------------- row filter


class UnsupportedFilter(RuntimeError):
    """Raised when a SQL row filter cannot be expressed as dataframe masks."""


_CLAUSE = re.compile(
    r"""^\s*"?(?P<col>[A-Za-z_][A-Za-z0-9_]*)"?\s*
        (?P<op><>|!=|>=|<=|==|=|<|>)\s*
        (?P<val>'(?:[^']|'')*'|[-+]?[0-9]*\.?[0-9]+)\s*$""",
    re.VERBOSE,
)


def parse_simple_filter(expr: str) -> list[tuple[str, str, object]]:
    """Translate a conjunction of simple comparisons into (column, op, value) triples.

    Deliberately narrow. Refuses anything it cannot translate *exactly* -- a filter that
    silently means something different on the GPU path would corrupt the comparison this
    script exists to make.
    """
    expr = expr.strip()
    if not expr:
        return []
    if re.search(r"\b(OR|NOT|IN|LIKE|BETWEEN|IS|CASE|SELECT)\b", expr, re.IGNORECASE):
        raise UnsupportedFilter(
            f"Row filter uses SQL this translator does not support: {expr!r}. "
            "Only AND-joined `column <op> literal` comparisons can be expressed as "
            "dataframe masks. This is the GPU path's real limitation, not a bug."
        )
    if "(" in expr or ")" in expr:
        raise UnsupportedFilter(f"Parenthesised row filters are not supported: {expr!r}")

    clauses: list[tuple[str, str, object]] = []
    for raw in re.split(r"\bAND\b", expr, flags=re.IGNORECASE):
        match = _CLAUSE.match(raw)
        if match is None:
            raise UnsupportedFilter(
                f"Could not translate row-filter clause {raw.strip()!r} into a mask."
            )
        col, op, raw_value = match.group("col"), match.group("op"), match.group("val")
        if raw_value.startswith("'"):
            value: object = raw_value[1:-1].replace("''", "'")
        elif "." in raw_value:
            value = float(raw_value)
        else:
            value = int(raw_value)
        clauses.append((col, "!=" if op == "<>" else ("==" if op == "=" else op), value))
    return clauses


def filter_columns(clauses: list[tuple[str, str, object]]) -> list[str]:
    return sorted({col for col, _, _ in clauses})


def apply_clauses(frame, clauses: list[tuple[str, str, object]]):
    """Apply parsed clauses as boolean masks. Works on pandas and cuDF alike."""
    if not clauses:
        return frame
    mask = None
    for col, op, value in clauses:
        if col not in frame.columns:
            raise UnsupportedFilter(f"Row filter references missing column {col!r}")
        column = frame[col]
        if op == "==":
            clause_mask = column == value
        elif op == "!=":
            clause_mask = column != value
        elif op == ">":
            clause_mask = column > value
        elif op == ">=":
            clause_mask = column >= value
        elif op == "<":
            clause_mask = column < value
        else:
            clause_mask = column <= value
        # SQL three-valued logic: NULL never satisfies a comparison.
        clause_mask = clause_mask.fillna(False)
        mask = clause_mask if mask is None else (mask & clause_mask)
    return frame[mask]


# --------------------------------------------------------------------------- dataframe path


def _complement(series):
    """Map A<->T, C<->G. Inputs are pre-validated to ACGT, so no NULL branch is needed."""
    return series.replace(COMPLEMENT)


def canonical_counts_dataframe(frame, bin_edges: np.ndarray, xp) -> pl.DataFrame:
    """The SBS96 x SNVQ-bin aggregate, in pandas/cuDF, matching the DuckDB SQL exactly.

    Canonicalisation normalises to the pyrimidine strand. For a purine reference the
    flanking bases are complemented **and swapped**: the base 5' of a purine reference
    sits 3' of its pyrimidine equivalent. Getting only the complement and not the swap is
    the classic way to produce a plausible-looking but wrong SBS96 matrix.
    """
    work = frame.rename(
        columns={
            "REF": "ref",
            "ALT": "alt",
            "X_PREV1": "prev",
            "X_NEXT1": "next",
            "SNVQ": "snvq",
        }
    )[["ref", "alt", "prev", "next", "snvq"]]

    for column in ("ref", "alt", "prev", "next"):
        work[column] = work[column].astype("str").str.upper()
    work["snvq"] = work["snvq"].astype("float64")

    valid = (
        work["snvq"].notna()
        & work["ref"].isin(BASES)
        & work["alt"].isin(BASES)
        & work["prev"].isin(BASES)
        & work["next"].isin(BASES)
        & (work["ref"] != work["alt"])
    )
    work = work[valid]

    # Split rather than .where(): string-typed .where support differs across cuDF
    # versions, while boolean-mask selection and concat are stable in both libraries.
    is_pyrimidine = work["ref"].isin(["C", "T"])
    pyr = work[is_pyrimidine]
    pur = work[~is_pyrimidine]

    pieces = []
    if len(pyr):
        pieces.append(
            xp.DataFrame(
                {
                    "central_change": pyr["ref"].str.cat(pyr["alt"], sep=">"),
                    "context16": pyr["prev"].str.cat(pyr["next"]),
                    "snvq": pyr["snvq"],
                }
            )
        )
    if len(pur):
        pieces.append(
            xp.DataFrame(
                {
                    "central_change": _complement(pur["ref"]).str.cat(
                        _complement(pur["alt"]), sep=">"
                    ),
                    # swapped: complement(next) becomes the 5' base
                    "context16": _complement(pur["next"]).str.cat(_complement(pur["prev"])),
                    "snvq": pur["snvq"],
                }
            )
        )
    if not pieces:
        return _empty_counts()
    canonical = xp.concat(pieces, ignore_index=True) if len(pieces) > 1 else pieces[0]
    canonical = canonical[canonical["central_change"].isin(CANONICAL_CHANGES)]

    lo = float(bin_edges[0])
    hi = float(bin_edges[-1])
    width = float(bin_edges[1] - bin_edges[0])
    n_bins = len(bin_edges) - 1

    canonical = canonical[(canonical["snvq"] >= lo) & (canonical["snvq"] <= hi)]
    if not len(canonical):
        return _empty_counts()

    if n_bins == 1:
        bin_index = (canonical["snvq"] * 0).astype("int64")
    else:
        # np.floor on the quotient, not floor-division: this is what the SQL does
        # (FLOOR(x / w)), and the two can differ by one ulp at exact bin edges.
        bin_index = np.floor((canonical["snvq"] - lo) / width).astype("int64")
        bin_index = bin_index.clip(0, n_bins - 1)
    canonical["bin_index"] = bin_index

    grouped = (
        canonical.groupby(["central_change", "context16", "bin_index"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    if hasattr(grouped, "to_pandas"):  # cuDF -> host
        grouped = grouped.to_pandas()
    return pl.from_pandas(grouped[["central_change", "context16", "bin_index", "count"]])


def _empty_counts() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "central_change": pl.Utf8,
            "context16": pl.Utf8,
            "bin_index": pl.Int64,
            "count": pl.Int64,
        }
    )


# --------------------------------------------------------------------------- runners


def normalise(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.height == 0:
        return _empty_counts()
    return (
        frame.select(
            pl.col("central_change").cast(pl.Utf8),
            pl.col("context16").cast(pl.Utf8),
            pl.col("bin_index").cast(pl.Int64),
            pl.col("count").cast(pl.Int64),
        )
        .group_by(["central_change", "context16", "bin_index"])
        .agg(pl.col("count").sum())
        .sort(["central_change", "context16", "bin_index"])
    )


def run_duckdb(sources, row_filter, limit, bin_edges, threads, memory_limit, temp_dir):
    start = perf_counter()
    frame = query_batch_histogram(
        sources, row_filter, limit, bin_edges, threads, memory_limit, temp_dir
    )
    elapsed = perf_counter() - start
    return normalise(frame), {"read+compute": elapsed, "total": elapsed}


def run_dataframe(backend: str, sources, row_filter, limit, bin_edges):
    clauses = parse_simple_filter(row_filter)
    columns = sorted(set(NEEDED_COLUMNS) | set(filter_columns(clauses)))

    if backend == "cudf":
        import cudf as xp
    else:
        import pandas as xp

    read_start = perf_counter()
    frames = []
    for source in sources:
        chunk = xp.read_parquet(str(source), columns=columns)
        chunk = apply_clauses(chunk, clauses)
        if limit is not None:
            chunk = chunk.head(int(limit))
        frames.append(chunk)
    frame = xp.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    read_seconds = perf_counter() - read_start

    compute_start = perf_counter()
    counts = canonical_counts_dataframe(frame, bin_edges, xp)
    if backend == "cudf":  # groupby is lazy-ish on device; force completion before timing
        import cudf  # noqa: F401
    compute_seconds = perf_counter() - compute_start

    return normalise(counts), {
        "read+filter": read_seconds,
        "compute": compute_seconds,
        "total": read_seconds + compute_seconds,
    }


def compare(reference: pl.DataFrame, other: pl.DataFrame, name: str) -> bool:
    if reference.equals(other):
        total = int(reference["count"].sum()) if reference.height else 0
        log(f"  {name}: MATCH ({reference.height:,} groups, {total:,} rows)")
        return True

    log(f"  {name}: MISMATCH")
    joined = reference.join(
        other,
        on=["central_change", "context16", "bin_index"],
        how="full",
        suffix="_other",
        coalesce=True,
    ).with_columns(
        pl.col("count").fill_null(0),
        pl.col("count_other").fill_null(0),
    )
    diff = joined.filter(pl.col("count") != pl.col("count_other"))
    log(
        f"    groups: reference={reference.height:,} {name}={other.height:,}; "
        f"differing groups={diff.height:,}"
    )
    log(
        f"    row totals: reference={int(reference['count'].sum()):,} "
        f"{name}={int(other['count'].sum()):,}"
    )
    with pl.Config(tbl_rows=20):
        print(diff.head(20), file=sys.stderr)
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark and cross-validate DuckDB / pandas / cuDF for the "
        "SBS96 x SNVQ histogram aggregate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        "--backends",
        default="duckdb,pandas",
        help="Comma-separated subset of duckdb,pandas,cudf. First listed is the "
        "correctness reference (default: duckdb,pandas).",
    )
    parser.add_argument(
        "--files",
        type=int,
        default=1,
        help="Use only the first N resolved files. Keep small: the dataframe backends "
        "materialise every row in RAM/VRAM (default 1).",
    )
    parser.add_argument("--row-filter", default=DEFAULT_ROW_FILTER)
    parser.add_argument("--num-bins", type=int, default=100)
    parser.add_argument("--min-snvq", type=float, default=None)
    parser.add_argument("--max-snvq", type=float, default=None)
    parser.add_argument("--threads", type=int, default=48, help="DuckDB threads")
    parser.add_argument("--duckdb-memory-limit", default="16GB")
    parser.add_argument("--duckdb-temp-dir", default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Row cap applied per file after filtering. Smoke testing only; note DuckDB "
        "and the dataframe path may pick different rows, so counts can differ legitimately.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Timed repetitions per backend. Report the minimum; the first pass pays "
        "cold page-cache and CUDA-context costs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    unknown = set(backends) - {"duckdb", "pandas", "cudf"}
    if unknown:
        raise SystemExit(f"Unknown backend(s): {', '.join(sorted(unknown))}")
    if not backends:
        raise SystemExit("No backends selected")

    sources = resolve_sources(args.parquet_paths)[: max(1, args.files)]
    temp_dir = (
        Path(args.duckdb_temp_dir).expanduser().resolve()
        if args.duckdb_temp_dir
        else Path.cwd() / "duckdb_tmp"
    )
    log(f"Benchmarking {len(sources)} file(s); backends={','.join(backends)}")
    for source in sources:
        log(f"  {source}")

    if args.limit is not None:
        log(
            "WARNING: --limit is set. DuckDB and the dataframe backends may retain "
            "different rows, so a MISMATCH here is not necessarily a real defect."
        )

    if args.min_snvq is None or args.max_snvq is None:
        log("Measuring SNVQ range (DuckDB counting pass) ...")
        stats = query_batch_stats(
            sources, args.row_filter, args.limit, args.threads, args.duckdb_memory_limit, temp_dir
        )
        if not stats["canonical_rows"]:
            raise SystemExit("No canonical SBS96 rows survived the filter; nothing to benchmark.")
        min_snvq = args.min_snvq if args.min_snvq is not None else stats["observed_min_snvq"]
        max_snvq = args.max_snvq if args.max_snvq is not None else stats["observed_max_snvq"]
        log(
            f"filtered_rows={stats['filtered_rows']:,}, "
            f"canonical_rows={stats['canonical_rows']:,}, "
            f"snvq_range=[{min_snvq:.3f}, {max_snvq:.3f}]"
        )
    else:
        min_snvq, max_snvq = args.min_snvq, args.max_snvq

    bin_edges = build_bin_edges(float(min_snvq), float(max_snvq), args.num_bins)

    results: dict[str, pl.DataFrame] = {}
    timings: dict[str, dict[str, float]] = {}

    for backend in backends:
        log(f"Running {backend} ...")
        best: dict[str, float] | None = None
        counts = None
        try:
            for attempt in range(max(1, args.repeat)):
                if backend == "duckdb":
                    counts, timing = run_duckdb(
                        sources,
                        args.row_filter,
                        args.limit,
                        bin_edges,
                        args.threads,
                        args.duckdb_memory_limit,
                        temp_dir,
                    )
                else:
                    counts, timing = run_dataframe(
                        backend, sources, args.row_filter, args.limit, bin_edges
                    )
                log(
                    f"  pass {attempt + 1}/{max(1, args.repeat)}: "
                    + ", ".join(f"{k}={v:.2f}s" for k, v in timing.items())
                )
                if best is None or timing["total"] < best["total"]:
                    best = timing
        except UnsupportedFilter as exc:
            log(f"  {backend} SKIPPED -- {exc}")
            continue
        except ImportError as exc:
            log(f"  {backend} SKIPPED -- not installed in this environment ({exc})")
            continue
        except Exception as exc:  # noqa: BLE001 - a failing backend must not hide the rest
            log(f"  {backend} FAILED -- {type(exc).__name__}: {exc}")
            continue
        assert counts is not None and best is not None
        results[backend] = counts
        timings[backend] = best

    if not results:
        raise SystemExit("Every backend failed or was skipped; nothing to report.")

    reference_name = next(iter(results))
    reference = results[reference_name]
    log(f"Correctness (reference = {reference_name}):")
    all_match = True
    for name, frame in results.items():
        if name == reference_name:
            continue
        all_match &= compare(reference, frame, name)
    if len(results) == 1:
        log(f"  only {reference_name} ran; nothing to cross-check against")

    log("Timings (best of %d):" % max(1, args.repeat))
    baseline = timings[reference_name]["total"]
    for name, timing in timings.items():
        parts = ", ".join(f"{k}={v:.2f}s" for k, v in timing.items() if k != "total")
        speedup = baseline / timing["total"] if timing["total"] > 0 else float("inf")
        log(f"  {name:7s} total={timing['total']:7.2f}s  ({parts})  vs {reference_name}: {speedup:.2f}x")

    if len(results) > 1:
        log("VERDICT: outputs identical" if all_match else "VERDICT: OUTPUTS DIFFER -- see diff above")


if __name__ == "__main__":
    main()
