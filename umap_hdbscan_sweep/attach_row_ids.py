#!/usr/bin/env python3
"""
attach_row_ids.py — backfill file_row_number into per_parquet_inference results.

For each sample that has labels.npy + umap_coords.npy, re-scans the source parquet
with the same filter used during inference (single-threaded, preserve insertion order)
to get file_row_number for each filtered row, then writes:

  <out_dir>/<sample>/row_assignments.parquet
      columns: file_row_number (i64), umap_1 (f32), umap_2 (f32), cluster_label (i32)

  <out_dir>/<sample>/source_fingerprint.json
      size_bytes, mtime_ns, footer_sha256, path

The positional pairing assumes DuckDB returns rows in file order under a
single-threaded scan (preserve_insertion_order=True, threads=1). The row-count
check is the integrity guard: if the counts don't match the pair is wrong and
we abort that sample rather than write a silently corrupt file.

Usage (miletus):
  python umap_hdbscan_sweep/attach_row_ids.py \\
      --results-dir $HOME/pure-internship/umap_hdbscan_sweep/per_parquet_inference \\
      --parquet-glob '/data/lab/ppmseq_parquets/*.parquet'
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import duckdb
import numpy as np
import polars as pl

ROW_FILTER = "st = 'MIXED' AND et = 'MIXED' AND FILT = 1"


def footer_sha256(path: str, tail_bytes: int = 1_000_000) -> str:
    size = os.stat(path).st_size
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        fh.seek(max(0, size - tail_bytes))
        h.update(fh.read())
    return h.hexdigest()


def source_fingerprint(path: str) -> dict:
    st = os.stat(path)
    return {
        "path": path,
        "size_bytes": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "footer_sha256": footer_sha256(path),
    }


def get_filtered_row_numbers(parquet_path: str) -> np.ndarray:
    """Single-threaded DuckDB scan returning file_row_number for every filtered row."""
    con = duckdb.connect(config={"threads": 1, "preserve_insertion_order": True})
    result = con.execute(
        f"SELECT file_row_number() AS frn "
        f"FROM read_parquet('{parquet_path}') "
        f"WHERE {ROW_FILTER} "
        f"ORDER BY frn"          # explicit order matches physical file order
    ).fetchnumpy()
    con.close()
    return result["frn"].astype(np.int64)


def log(msg: str) -> None:
    print(msg, flush=True)


def process_sample(sample_dir: Path, parquet_path: str, skip_existing: bool) -> str:
    """Returns 'ok', 'skipped', or 'error:<reason>'."""
    out_assign = sample_dir / "row_assignments.parquet"
    out_fp = sample_dir / "source_fingerprint.json"

    if skip_existing and out_assign.exists() and out_fp.exists():
        return "skipped"

    labels_path = sample_dir / "labels.npy"
    coords_path = sample_dir / "umap_coords.npy"

    if not labels_path.exists():
        return "error:labels.npy missing"
    if not coords_path.exists():
        return "error:umap_coords.npy missing"

    labels = np.load(labels_path)          # shape (N,)  int
    coords = np.load(coords_path)          # shape (N,2) float32
    n_rows = len(labels)

    t0 = perf_counter()
    frn = get_filtered_row_numbers(parquet_path)
    scan_s = perf_counter() - t0

    if len(frn) != n_rows:
        return (f"error:row count mismatch — parquet filtered to {len(frn):,} rows "
                f"but labels.npy has {n_rows:,}")

    # write sidecar parquet
    df = pl.DataFrame({
        "file_row_number": frn,
        "umap_1":          coords[:, 0].astype(np.float32),
        "umap_2":          coords[:, 1].astype(np.float32),
        "cluster_label":   labels.astype(np.int32),
    })
    df.write_parquet(str(out_assign), compression="zstd")

    # write fingerprint
    fp = source_fingerprint(parquet_path)
    out_fp.write_text(json.dumps(fp, indent=2))

    log(f"  {n_rows:,} rows  scan {scan_s:.1f}s  → {out_assign.name}")
    return "ok"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True,
                   help="per_parquet_inference output dir")
    p.add_argument("--parquet-glob", required=True,
                   help="glob matching the 95 source parquet files")
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="skip samples that already have row_assignments.parquet")
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = p.parse_args()

    results_root = Path(args.results_dir)
    parquet_paths = {Path(p).stem: p for p in glob.glob(args.parquet_glob)}
    log(f"Found {len(parquet_paths)} parquet files")

    sample_dirs = sorted(d for d in results_root.iterdir()
                         if d.is_dir() and d.name != "signature_db")
    log(f"Found {len(sample_dirs)} sample dirs")

    counts = {"ok": 0, "skipped": 0, "error": 0, "no_parquet": 0}
    errors: list[str] = []

    for i, sample_dir in enumerate(sample_dirs, 1):
        sample = sample_dir.name  # e.g. csb0-1-ppm0058.featuremap
        stem   = sample           # parquet file has same stem

        if stem not in parquet_paths:
            # try without .featuremap suffix
            stem2 = sample.replace(".featuremap", "")
            if stem2 in parquet_paths:
                stem = stem2
            else:
                log(f"[{i}/{len(sample_dirs)}] {sample}  -- NO SOURCE PARQUET FOUND")
                counts["no_parquet"] += 1
                continue

        pq_path = parquet_paths[stem]
        log(f"[{i}/{len(sample_dirs)}] {sample}")
        result = process_sample(sample_dir, pq_path, args.skip_existing)

        if result == "ok":
            counts["ok"] += 1
        elif result == "skipped":
            log("  skipped (already done)")
            counts["skipped"] += 1
        else:
            msg = result.removeprefix("error:")
            log(f"  ERROR: {msg}")
            errors.append(f"{sample}: {msg}")
            counts["error"] += 1

    log(f"\nDone: {counts}")
    if errors:
        log("\nErrors:")
        for e in errors:
            log(f"  {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
