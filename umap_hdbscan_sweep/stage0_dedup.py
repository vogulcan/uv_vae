"""Stage 0 -- deduplicate the 95-file cohort to one row per (CHROM, POS, REF, ALT).

``pipeline_vae.write_deduplicated_sample`` does this for ONE parquet and materialises the
whole result as a temp table. Neither survives the cohort: it calls ``pq.read_schema`` on a
single file, and a global window over 5,078,201,907 rows would need a sort far larger than
host RAM.

This module fixes both:

* **All 95 files at once** via ``read_parquet([...])`` -- DuckDB unions them, and the schema
  is read from the first file (they are produced by the same pipeline, so it is shared).
* **One chromosome at a time.** ``(CHROM, POS, REF, ALT)`` partitions cleanly on CHROM, so
  deduplicating per chromosome gives the identical answer while keeping each sort ~1/24th
  the size. The files are sorted by ``(CHROM, POS)``, so the row-group statistics prune
  almost perfectly and each pass reads only its own chromosome's row groups.

The ranking that picks the surviving row per locus: QUAL, MAPQ, DP, RAW_VAF DESC, then a
stable tie-break on available columns. SNVQ is excluded because it is a derived score that
conflates multiple quality signals already captured individually by the other columns.

``locus_reads`` -- how many reads collapsed into each surviving row -- is computed in the
same window pass. It costs nothing extra and is what lets stage 3 report read-weighted
signature counts without a second scan of the cohort.
"""
from __future__ import annotations

import argparse
import glob
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
for candidate in (REPO_ROOT / "uv_vae", REPO_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import pyarrow.parquet as pq

from uv_vae.data import connect_duckdb, quote_ident
from uv_vae.pipeline_vae import (
    build_selected_columns,
    load_feature_names,
    sql_quote,
    stable_order_by,
)

DEFAULT_ROW_FILTER = "st = 'MIXED' AND et = 'MIXED' AND FILT = 1"
IDENTITY_COLUMNS = ["CHROM", "POS", "REF", "ALT"]
# Context columns SigProfiler needs to build the SBS96 trinucleotide matrix downstream.
CONTEXT_COLUMNS = ["X_PREV1", "X_NEXT1"]
# Carried through because the ORDER BY below already reads them, so selecting them is free,
# and because they are four of the eight columns run_variant_cluster_pipeline colours its
# UMAP by. Leaving them out made those plots unreproducible for the cohort without a second
# pass over all 5.08B source rows. SNVQ rides along for the same reason even though it is
# deliberately not part of the ranking.
QUALITY_COLUMNS = ["QUAL", "MAPQ", "DP", "RAW_VAF", "SNVQ"]


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def resolve_parquet_paths(patterns: list[str]) -> list[Path]:
    """Expand globs and literal paths into a sorted, de-duplicated list.

    Sorted by filename so the union -- and therefore which row wins a locus tie -- does not
    depend on shell glob ordering. Mirrors ``multi_streaming.resolve_parquet_paths`` so the
    cohort is enumerated identically here and in training, but is duplicated rather than
    imported: that module pulls in torch, and this stage is pure DuckDB.
    """
    resolved: list[Path] = []
    for pattern in patterns:
        candidate = Path(pattern)
        if candidate.exists():
            resolved.append(candidate)
            continue
        # glob.glob handles absolute patterns; Path.glob does not.
        matches = sorted(Path(match) for match in glob.glob(pattern))
        if not matches:
            raise SystemExit(f"no parquet files matched {pattern!r}")
        resolved.extend(matches)

    unique: list[Path] = []
    seen: set[str] = set()
    for path in sorted(resolved, key=lambda item: item.name):
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path.resolve())
    return unique


def parquet_list_sql(paths: list[Path]) -> str:
    return "[" + ", ".join(sql_quote(str(path)) for path in paths) + "]"


def discover_chromosomes(connection, source: str, row_filter: str) -> list[str]:
    """List the distinct CHROM values present after filtering.

    Only the CHROM column is projected, so this scans a single column across the cohort
    rather than the full row width.
    """
    sql = f"SELECT DISTINCT \"CHROM\" AS chrom FROM read_parquet({source}) WHERE {row_filter} ORDER BY 1"
    return [row[0] for row in connection.execute(sql).fetchall() if row[0] is not None]


def dedup_one_chromosome(
    connection,
    source: str,
    chromosome: str,
    row_filter: str,
    select_columns: list[str],
    output_path: Path,
    tie_break_columns: list[str] | None = None,
) -> tuple[int, int]:
    """Write the deduplicated rows for one chromosome; return (reads_in, loci_out)."""
    select_list = ", ".join(quote_ident(name) for name in select_columns)
    # Deliberately NOT select_columns: the quality columns were added to the SELECT list
    # after the fact, and stable_order_by folds whatever it is given into the tie-break, so
    # passing them here would change which row survives a tie and silently make this dedup
    # disagree with every earlier one. The tie-break stays on the original column set.
    tie_break = stable_order_by(tie_break_columns or select_columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # COUNT(*) OVER the same partition rides along on the window already being built for
    # ROW_NUMBER, so locus_reads is effectively free.
    sql = f"""
        COPY (
            WITH ranked AS (
                SELECT
                    {select_list},
                    ROW_NUMBER() OVER (
                        PARTITION BY "CHROM", "POS", "REF", "ALT"
                        ORDER BY
                            "QUAL" DESC NULLS LAST,
                            "MAPQ" DESC NULLS LAST,
                            "DP" DESC NULLS LAST,
                            "RAW_VAF" DESC NULLS LAST,
                            {tie_break}
                    ) AS rn,
                    COUNT(*) OVER (PARTITION BY "CHROM", "POS", "REF", "ALT") AS locus_reads
                FROM read_parquet({source})
                WHERE ({row_filter}) AND "CHROM" = {sql_quote(chromosome)}
            )
            SELECT {select_list}, CAST(locus_reads AS INTEGER) AS locus_reads
            FROM ranked
            WHERE rn = 1
        ) TO {sql_quote(str(output_path))} (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    connection.execute(sql)

    loci = connection.execute(
        f"SELECT COUNT(*), COALESCE(SUM(locus_reads), 0) FROM read_parquet({sql_quote(str(output_path))})"
    ).fetchone()
    return int(loci[1]), int(loci[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deduplicate the cohort to one row per locus")
    parser.add_argument("--parquet-paths", nargs="+", required=True, help="paths or globs (95 files)")
    parser.add_argument("--checkpoint-path", required=True, help="supplies the feature column list")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--row-filter", default=DEFAULT_ROW_FILTER)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--memory-limit", default="200GB")
    parser.add_argument("--temp-directory", default=None,
                        help="spill directory; needs room for roughly the cohort's on-disk size")
    parser.add_argument("--chromosomes", default=None,
                        help="comma-separated subset (default: every CHROM present)")
    return parser.parse_args()


def main() -> int:
    started = perf_counter()
    args = parse_args()

    paths = resolve_parquet_paths(args.parquet_paths)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_directory = Path(args.temp_directory) if args.temp_directory else output_dir / "_duckdb_tmp"
    temp_directory.mkdir(parents=True, exist_ok=True)

    log(f"Cohort: {len(paths)} parquet files")

    feature_names = load_feature_names(args.checkpoint_path)
    wanted = build_selected_columns(feature_names)
    available = set(pq.read_schema(paths[0]).names)
    missing_identity = [name for name in IDENTITY_COLUMNS if name not in available]
    if missing_identity:
        raise SystemExit(f"cannot deduplicate: {missing_identity} absent from {paths[0]}")

    # dict.fromkeys preserves order while dropping the duplicates the concatenation created.
    tie_break_columns = list(dict.fromkeys(
        name for name in [*wanted, *IDENTITY_COLUMNS, *CONTEXT_COLUMNS] if name in available
    ))
    select_columns = list(dict.fromkeys([
        *tie_break_columns,
        *(name for name in QUALITY_COLUMNS if name in available),
    ]))
    extra = [name for name in select_columns if name not in tie_break_columns]
    log(f"Selecting {len(select_columns)} columns"
        + (f" (+{', '.join(extra)} carried for downstream plots)" if extra else ""))

    source = parquet_list_sql(paths)
    connection = connect_duckdb(
        threads=args.threads,
        temp_directory=temp_directory,
        memory_limit=args.memory_limit,
    )

    chromosomes = (
        [part.strip() for part in args.chromosomes.split(",") if part.strip()]
        if args.chromosomes
        else discover_chromosomes(connection, source, args.row_filter)
    )
    log(f"Deduplicating {len(chromosomes)} chromosomes: {', '.join(chromosomes)}")

    parts_dir = output_dir / "dedup"
    parts_dir.mkdir(parents=True, exist_ok=True)
    per_chromosome: list[dict] = []
    total_reads = total_loci = 0

    for index, chromosome in enumerate(chromosomes, start=1):
        chromosome_started = perf_counter()
        part_path = parts_dir / f"chrom={chromosome}.parquet"
        reads, loci = dedup_one_chromosome(
            connection, source, chromosome, args.row_filter, select_columns, part_path,
            tie_break_columns=tie_break_columns,
        )
        total_reads += reads
        total_loci += loci
        elapsed = perf_counter() - chromosome_started
        per_chromosome.append(
            {
                "chromosome": chromosome,
                "reads": reads,
                "loci": loci,
                "collapse_ratio": round(reads / loci, 2) if loci else None,
                "path": str(part_path),
                "seconds": round(elapsed, 1),
            }
        )
        log(
            f"  [{index}/{len(chromosomes)}] {chromosome}: {reads:,} reads -> {loci:,} loci "
            f"({reads / loci:.1f}x) in {elapsed:.0f}s"
        )

    manifest = {
        "parquet_files": [str(path) for path in paths],
        "row_filter": args.row_filter,
        "select_columns": select_columns,
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "total_reads": total_reads,
        "total_loci": total_loci,
        "collapse_ratio": round(total_reads / total_loci, 2) if total_loci else None,
        "per_chromosome": per_chromosome,
        "dedup_dir": str(parts_dir),
        "seconds": round(perf_counter() - started, 1),
    }
    (output_dir / "dedup_manifest.json").write_text(json.dumps(manifest, indent=2))

    log("")
    log(f"TOTAL: {total_reads:,} reads -> {total_loci:,} unique loci "
        f"({manifest['collapse_ratio']}x collapse) in {manifest['seconds']}s")
    log(f"Manifest -> {output_dir / 'dedup_manifest.json'}")
    log("")
    log(f"{total_loci:,} loci is what stage 2 fits UMAP/HDBSCAN on. "
        "Above ~20M expect the GPU fit to need chunking or a capped fit set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
