#!/usr/bin/env python
"""Check that per-sample assignments still key correctly against their source parquets.

Why this is needed
------------------
``per_parquet_inference.py`` joins its results back on ``file_row_number``, which duckdb
derives from a row's physical position rather than reading it from the file. That makes it
a correct join key only while the source parquet is byte-identical to the one that was
labelled. If a file is rewritten, re-sorted or recompacted, every id silently repoints at a
different read -- the join still succeeds, it is just wrong, and nothing in the output
would look unusual.

So each sample dir carries ``source_fingerprint.json`` recording the file's size, mtime and
a hash of its footer (schema + row-group offsets + row counts, which any rewrite disturbs).
This script re-reads the sources and reports drift before a join is trusted.

Run it before joining, and again before publishing anything derived from a join::

    python umap_hdbscan_sweep/verify_source_fingerprint.py \\
        --results-dir $HOME/pure-internship/umap_hdbscan_sweep/per_parquet_inference

Exit status is 0 when every sample matches, 1 when any drifted or could not be checked, so
it can gate a downstream step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def footer_sha256(path: str, tail_bytes: int = 1_000_000) -> str:
    size = os.stat(path).st_size
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        fh.seek(max(0, size - tail_bytes))
        h.update(fh.read())
    return h.hexdigest()


def check(record: dict) -> tuple[str, str]:
    """Returns (status, detail). status is OK / MISSING / DRIFTED."""
    path = record["path"]
    if not os.path.exists(path):
        return "MISSING", f"source no longer exists: {path}"

    st = os.stat(path)
    if st.st_size != record["size_bytes"]:
        return "DRIFTED", (f"size {st.st_size:,} != recorded {record['size_bytes']:,}")

    actual = footer_sha256(path)
    if actual != record["footer_sha256"]:
        return "DRIFTED", (f"footer hash {actual[:16]}… != recorded "
                           f"{record['footer_sha256'][:16]}…")

    # mtime alone is not proof of change -- a touch or a restore-in-place moves it while
    # the bytes stay identical -- so it is reported, never fatal.
    if st.st_mtime_ns != record["mtime_ns"]:
        return "OK", "mtime moved but size and footer hash match (content unchanged)"
    return "OK", ""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True,
                   help="per_parquet_inference output dir (contains one dir per sample)")
    p.add_argument("--quiet", action="store_true", help="only print problems")
    args = p.parse_args()

    root = Path(args.results_dir)
    fingerprints = sorted(root.glob("*/source_fingerprint.json"))
    if not fingerprints:
        print(f"No source_fingerprint.json found under {root}", file=sys.stderr)
        return 1

    counts = {"OK": 0, "DRIFTED": 0, "MISSING": 0, "UNREADABLE": 0}
    problems: list[str] = []

    for fp in fingerprints:
        sample = fp.parent.name
        try:
            record = json.loads(fp.read_text())
            status, detail = check(record)
        except Exception as exc:  # noqa: BLE001
            status, detail = "UNREADABLE", f"{type(exc).__name__}: {exc}"
        counts[status] += 1
        line = f"  {status:10s} {sample}" + (f"  -- {detail}" if detail else "")
        if status != "OK" or detail:
            problems.append(line)
        if not args.quiet:
            print(line)

    print(f"\n{len(fingerprints)} samples checked: "
          + ", ".join(f"{k}={v}" for k, v in counts.items() if v))

    bad = counts["DRIFTED"] + counts["MISSING"] + counts["UNREADABLE"]
    if bad:
        if args.quiet:
            print("\n".join(problems))
        print(f"\n{bad} sample(s) cannot be safely joined on file_row_number.\n"
              f"A drifted source must be re-run through per_parquet_inference.py; its\n"
              f"existing row_assignments.parquet keys into the OLD file layout.",
              file=sys.stderr)
        return 1
    print("All sources unchanged -- file_row_number joins are safe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
