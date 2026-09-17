#!/usr/bin/env python
"""Collect the per-sample UMAP coordinates and cluster labels, and optionally attach them.

``per_parquet_inference.py`` already writes the joinable artefact for each sample --
``row_assignments.parquet``: one row per surviving read, keyed on ``file_row_number``, with
``umap_1``, ``umap_2`` and ``cluster_label``. What it does not do is give the 95 of them a
single point of entry, or perform the join. This does both:

``--manifest`` (default)
    Walk the results dir, read every ``row_assignments.parquet`` and
    ``source_fingerprint.json``, and write ``assignments_manifest.csv`` / ``.json``: which
    source file each sample keys against, how many rows carry a label, the per-sample cluster
    count and noise fraction, and the ``file_row_number`` range covered. Source fingerprints
    are re-checked in the same pass, because a manifest that lists a drifted file as usable
    is worse than no manifest.

``--attach``
    Actually join, writing ``<out-dir>/<sample>.enriched.parquet`` = every source column plus
    the three new ones. Streamed through duckdb's ``COPY``, so peak memory is a row group
    rather than a file.

The join key deserves care. ``file_row_number`` is *positional* -- duckdb derives it from
physical layout rather than reading it from the file -- so it identifies a read only while
the parquet is byte-identical to the one that was labelled. Every mode here verifies the
recorded fingerprint before trusting it, and ``--attach`` refuses to write a sample whose
source has drifted unless ``--allow-drift`` says otherwise.

Rows the inference filter dropped (``st``/``et``/``FILT``) have no assignment. The default
LEFT join keeps them with NULL coordinates and a NULL label, so the enriched file stays
row-for-row comparable with its source; ``--filtered-only`` emits just the labelled rows.

Usage::

    # what have I got, and does it still key correctly?
    python umap_hdbscan_sweep/export_assignments.py \\
        --results-dir $HOME/pure-internship/umap_hdbscan_sweep/per_parquet_inference_cuml

    # attach to the sources
    python umap_hdbscan_sweep/export_assignments.py \\
        --results-dir $HOME/pure-internship/umap_hdbscan_sweep/per_parquet_inference_cuml \\
        --attach --out-dir /data/lab/ppmseq_enriched
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ASSIGNMENT_COLUMNS = ("umap_1", "umap_2", "cluster_label")


def log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}", flush=True)


def footer_sha256(path: str, tail_bytes: int = 1_000_000) -> str:
    """Hash of the parquet footer -- schema, row-group offsets, row counts.

    The footer rather than the whole file because it is what any rewrite, re-sort or
    recompaction disturbs, and hashing several GB per sample to learn the same thing would
    make this check expensive enough to skip.
    """
    size = os.stat(path).st_size
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        handle.seek(max(0, size - tail_bytes))
        digest.update(handle.read())
    return digest.hexdigest()


def check_fingerprint(record: dict) -> tuple[str, str]:
    """(status, detail) where status is OK / MISSING / DRIFTED / NOFILE."""
    path = record.get("path")
    if not path:
        return "NOFILE", "fingerprint records no source path"
    if not os.path.exists(path):
        return "MISSING", f"source no longer at {path}"
    try:
        size = os.stat(path).st_size
        if record.get("size_bytes") is not None and size != record["size_bytes"]:
            return "DRIFTED", f"size {record['size_bytes']:,} -> {size:,}"
        recorded = record.get("footer_sha256")
        if recorded:
            current = footer_sha256(path)
            if current != recorded:
                return "DRIFTED", f"footer {recorded[:12]}… -> {current[:12]}…"
        return "OK", ""
    except Exception as exc:  # noqa: BLE001
        return "MISSING", f"{type(exc).__name__}: {exc}"


def scan_sample(sample_dir: Path) -> dict | None:
    """Summarise one sample dir, or None if it holds no assignments."""
    assignments = sample_dir / "row_assignments.parquet"
    if not assignments.exists():
        return None

    import polars as pl

    record: dict = {"sample": sample_dir.name,
                    "assignments": str(assignments),
                    "assignments_mb": round(assignments.stat().st_size / 1e6, 1)}

    fingerprint_path = sample_dir / "source_fingerprint.json"
    fingerprint: dict = {}
    if fingerprint_path.exists():
        try:
            fingerprint = json.loads(fingerprint_path.read_text())
        except Exception as exc:  # noqa: BLE001
            record["fingerprint_error"] = f"{type(exc).__name__}: {exc}"
    record["source"] = fingerprint.get("path", "")
    record["row_filter"] = fingerprint.get("row_filter", "")

    status, detail = check_fingerprint(fingerprint) if fingerprint else ("NOFILE", "no source_fingerprint.json")
    record["source_status"] = status
    record["source_detail"] = detail

    # Aggregate in one lazy pass -- these files are ~40M rows each and the numbers wanted
    # are all reductions, so nothing needs to be brought into memory whole.
    frame = pl.scan_parquet(assignments)
    stats = frame.select(
        pl.len().alias("rows"),
        pl.col("file_row_number").min().alias("frn_min"),
        pl.col("file_row_number").max().alias("frn_max"),
        pl.col("cluster_label").max().alias("label_max"),
        (pl.col("cluster_label") < 0).sum().alias("noise_rows"),
        pl.col("cluster_label").n_unique().alias("distinct_labels"),
    ).collect().to_dicts()[0]

    rows = int(stats["rows"])
    noise_rows = int(stats["noise_rows"])
    label_max = int(stats["label_max"]) if stats["label_max"] is not None else -1
    record.update({
        "rows": rows,
        "file_row_number_min": int(stats["frn_min"]),
        "file_row_number_max": int(stats["frn_max"]),
        # Clusters PRESENT IN THIS SAMPLE, not the cohort total: a sample need not visit
        # every cohort cluster, and distinct_labels counts noise as one of its values.
        "clusters_present": int(stats["distinct_labels"]) - (1 if noise_rows else 0),
        "max_cluster_id": label_max,
        "noise_rows": noise_rows,
        "noise_pct": round(noise_rows / rows * 100, 4) if rows else None,
    })
    return record


def write_manifest(records: list[dict], out_dir: Path, results_dir: Path) -> None:
    columns = ["sample", "source", "source_status", "source_detail", "rows",
               "clusters_present", "max_cluster_id", "noise_rows", "noise_pct",
               "file_row_number_min", "file_row_number_max", "assignments",
               "assignments_mb", "row_filter"]
    csv_path = out_dir / "assignments_manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    total_rows = sum(r["rows"] for r in records)
    total_noise = sum(r["noise_rows"] for r in records)
    cohort_max = max((r["max_cluster_id"] for r in records), default=-1)
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "results_dir": str(results_dir),
        "n_samples": len(records),
        "total_rows": total_rows,
        "total_noise_rows": total_noise,
        "noise_pct": round(total_noise / total_rows * 100, 4) if total_rows else None,
        # The largest label seen anywhere, +1. Equals the cohort cluster count only if every
        # cluster appears in at least one sample -- stated as an observation, not asserted.
        "max_cluster_id_across_samples": cohort_max,
        "implied_cluster_count": cohort_max + 1 if cohort_max >= 0 else 0,
        "source_status_counts": {
            status: sum(1 for r in records if r["source_status"] == status)
            for status in sorted({r["source_status"] for r in records})
        },
        "columns": {
            "file_row_number": "int64, positional row id in the source parquet (join key)",
            "umap_1": "float32, parametric UMAP dimension 1",
            "umap_2": "float32, parametric UMAP dimension 2",
            "cluster_label": "int32, HDBSCAN cluster id; -1 = noise",
        },
        "samples": records,
    }
    (out_dir / "assignments_manifest.json").write_text(json.dumps(payload, indent=2))
    log(f"manifest → {csv_path.name}, assignments_manifest.json")
    return payload


def attach(record: dict, out_dir: Path, args) -> dict:
    """Join one sample's assignments onto its source parquet."""
    import duckdb

    source = record["source"]
    out_path = out_dir / f"{record['sample']}.enriched.parquet"
    result = {"sample": record["sample"], "out": str(out_path)}

    if args.skip_existing and out_path.exists():
        result["status"] = "skipped (exists)"
        return result
    if not source or not os.path.exists(source):
        result["status"] = f"FAILED: source missing ({source or 'unrecorded'})"
        return result

    connection = duckdb.connect()
    connection.execute(f"SET memory_limit='{args.memory_limit}'")
    connection.execute(f"SET threads={args.threads}")
    if args.temp_dir:
        connection.execute(f"SET temp_directory='{args.temp_dir}'")

    # file_row_number is generated by the reader, so a source that already has a column of
    # that name would produce two and make the join ambiguous. Rename ours out of the way
    # rather than guessing which one the assignment file meant.
    existing = {row[0] for row in connection.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{source}') LIMIT 0").fetchall()}
    # See build_enriched_views for why this is refused rather than worked around: duckdb
    # rejects file_row_number=true on a file that already has that column, and the option
    # takes no alternative name.
    if "file_row_number" in existing:
        result["status"] = ("FAILED: source already has a file_row_number column, so the "
                            "positional join key cannot be generated. The file must have "
                            "changed since inference ran -- re-run inference for this sample.")
        connection.close()
        return result
    key = "file_row_number"

    join_type = "INNER" if args.filtered_only else "LEFT"
    select_columns = ", ".join(f'src."{name}"' for name in sorted(existing)) or "src.*"
    query = f"""
        COPY (
            SELECT {select_columns},
                   asg.umap_1        AS umap_1,
                   asg.umap_2        AS umap_2,
                   asg.cluster_label AS cluster_label
            FROM read_parquet('{source}', file_row_number=true) AS src
            {join_type} JOIN read_parquet('{record["assignments"]}') AS asg
                   ON src.file_row_number = asg.file_row_number
        ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION {args.compression})
    """
    started = perf_counter()
    try:
        connection.execute(query)
    except Exception as exc:  # noqa: BLE001
        result["status"] = f"FAILED: {type(exc).__name__}: {exc}"
        connection.close()
        return result

    # Did the join find the rows it should have? A LEFT join that matched nothing writes a
    # perfectly valid file in which every new column is null, and only a count catches it.
    matched = connection.execute(
        f"SELECT count(*) FROM read_parquet('{out_path}') WHERE cluster_label IS NOT NULL"
    ).fetchone()[0]
    written = connection.execute(
        f"SELECT count(*) FROM read_parquet('{out_path}')").fetchone()[0]
    connection.close()

    result.update({
        "seconds": round(perf_counter() - started, 1),
        "rows_written": int(written),
        "rows_labelled": int(matched),
        "rows_expected": record["rows"],
        "size_mb": round(out_path.stat().st_size / 1e6, 1),
    })
    if matched != record["rows"]:
        result["status"] = (f"FAILED: {matched:,} rows joined but the assignment file has "
                            f"{record['rows']:,} -- key mismatch, output is not trustworthy")
    else:
        result["status"] = "ok"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", required=True, type=Path,
                        help="per_parquet_inference output dir (one subdir per sample)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="where to write the manifest / enriched parquets "
                             "(default: --results-dir)")
    parser.add_argument("--attach", action="store_true",
                        help="also write <sample>.enriched.parquet = source + 3 columns")
    parser.add_argument("--filtered-only", action="store_true",
                        help="INNER join: emit only rows that carry an assignment")
    parser.add_argument("--allow-drift", action="store_true",
                        help="attach even where the source fingerprint no longer matches")
    parser.add_argument("--skip-existing", action="store_true",
                        help="leave enriched files that are already present")
    parser.add_argument("--samples", default=None,
                        help="comma-separated sample names; default is all of them")
    parser.add_argument("--memory-limit", default="32GB")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--temp-dir", default=None,
                        help="duckdb spill directory for the join")
    parser.add_argument("--compression", default="ZSTD",
                        choices=["ZSTD", "SNAPPY", "GZIP", "UNCOMPRESSED"])
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    if not results_dir.is_dir():
        log(f"ERROR: no such results dir: {results_dir}")
        return 1
    out_dir: Path = args.out_dir or results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = set(args.samples.split(",")) if args.samples else None
    sample_dirs = sorted(p for p in results_dir.iterdir() if p.is_dir())
    if wanted:
        sample_dirs = [p for p in sample_dirs if p.name in wanted]

    log(f"scanning {len(sample_dirs)} candidate sample dirs under {results_dir}")
    records, skipped = [], []
    for sample_dir in sample_dirs:
        record = scan_sample(sample_dir)
        if record is None:
            skipped.append(sample_dir.name)
            continue
        records.append(record)
        flag = "" if record["source_status"] == "OK" else f"  [{record['source_status']}: {record['source_detail']}]"
        log(f"  {record['sample']:<34} {record['rows']:>12,} rows  "
            f"{record['clusters_present']:>4} clusters  "
            f"{record['noise_pct']:>6.2f}% noise{flag}")

    if not records:
        log("ERROR: no row_assignments.parquet found under any sample dir.")
        log("  per_parquet_inference writes it; older runs may need attach_row_ids.py.")
        return 1
    if skipped:
        log(f"{len(skipped)} dirs had no row_assignments.parquet: {', '.join(skipped[:8])}"
            + (" …" if len(skipped) > 8 else ""))

    payload = write_manifest(records, out_dir, results_dir)
    log(f"{payload['n_samples']} samples, {payload['total_rows']:,} labelled rows, "
        f"{payload['noise_pct']}% noise, "
        f"{payload['implied_cluster_count']} clusters implied by the largest label")
    for status, count in payload["source_status_counts"].items():
        log(f"  source {status}: {count}")

    if not args.attach:
        log("manifest only; pass --attach to write enriched parquets")
        return 0 if payload["source_status_counts"].get("OK", 0) == len(records) else 1

    drifted = [r for r in records if r["source_status"] != "OK"]
    if drifted and not args.allow_drift:
        log(f"REFUSING to attach: {len(drifted)} sample(s) no longer key against their "
            f"recorded source.")
        for record in drifted:
            log(f"  {record['sample']}: {record['source_status']} {record['source_detail']}")
        log("  file_row_number is positional -- a drifted source means the join would "
            "attach every read to the wrong row. Re-run inference for these samples, or "
            "pass --allow-drift if the change is known to be irrelevant.")
        return 1

    log(f"attaching → {out_dir}")
    results = []
    for record in records:
        result = attach(record, out_dir, args)
        results.append(result)
        detail = ""
        if result.get("rows_written") is not None:
            detail = (f"  {result['rows_written']:,} rows "
                      f"({result['rows_labelled']:,} labelled)  "
                      f"{result['size_mb']:,} MB  {result['seconds']}s")
        log(f"  {result['sample']:<34} {result['status']}{detail}")

    (out_dir / "attach_summary.json").write_text(json.dumps(results, indent=2))
    failed = [r for r in results if r["status"].startswith("FAILED")]
    log(f"attached {len(results) - len(failed)}/{len(results)}; summary → attach_summary.json")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
