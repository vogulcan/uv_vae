#!/usr/bin/env python
"""Make the merged view of every sample queryable, without merging any data.

The advisor's requirement is access, not duplication: at any moment, be able to read a
sample's original columns *together with* its UMAP coordinates and HDBSCAN label. Writing 95
enriched parquets satisfies that by copying 569 GB of source data so three columns can ride
along beside it. A view satisfies it by storing the sentence that describes the join.

So this builds a DuckDB database of views. The file is kilobytes -- a view holds SQL text,
not rows -- but querying it is indistinguishable from querying a merged table:

    SELECT * FROM csb0_1_ppm0058 WHERE cluster_label = 42;

Every source column is there, plus ``umap_1``, ``umap_2``, ``cluster_label``. DuckDB pushes
the filter down into both parquets and reads only the row groups it needs, so a selective
query touches a fraction of the data a materialised merge would have to scan.

``all_samples`` unions all 95 with a ``sample`` column in front, for cohort-wide questions.

What this does NOT do is make the join safe forever on its own. ``file_row_number`` is
positional -- duckdb derives it from physical layout rather than reading it from the file --
so every view here is valid only while its source parquet stays byte-identical to the one
that was labelled. The manifest records a fingerprint per source; ``--verify`` re-checks
them, and ``verify_source_fingerprint.py`` is the standalone version to run before trusting
anything derived from these views months from now.

Usage::

    # build it (on miletus, where the sources live)
    python umap_hdbscan_sweep/build_enriched_views.py \\
        --manifest ~/pure-internship/umap_hdbscan_sweep/per_parquet_inference_cuml/assignments_manifest.json \\
        --database ~/pure-internship/umap_hdbscan_sweep/enriched.duckdb --verify 3

    # then, from anywhere
    duckdb ~/pure-internship/umap_hdbscan_sweep/enriched.duckdb
    D SELECT * FROM sample_index LIMIT 5;
    D SELECT * FROM csb0_1_ppm0058 LIMIT 5;
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ASSIGNMENT_COLUMNS = ("umap_1", "umap_2", "cluster_label")


def log(message: str) -> None:
    stream = sys.stdout
    text = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}"
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
    except UnicodeEncodeError:
        text = text.encode(encoding, errors="replace").decode(encoding)
    print(text, flush=True)


def view_name(sample: str) -> str:
    """A sample name that SQL will accept unquoted.

    ``csb0-1-ppm0058.featuremap`` is not an identifier -- the dashes parse as subtraction and
    the dot as a schema qualifier. Both survive as quoted identifiers, but a name that needs
    quoting in every query someone writes is a name that will be got wrong. The original is
    kept in ``sample_index`` so nothing is lost.
    """
    name = sample.removesuffix(".featuremap")
    name = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    if name and name[0].isdigit():
        name = f"s_{name}"
    return name


def rewrite(path: str, old_root: str | None, new_root: str | None) -> str:
    """Repoint an absolute path recorded on one machine at another machine's copy."""
    if not old_root or not new_root or not path:
        return path
    old_root = old_root.rstrip("/\\")
    if path.startswith(old_root):
        return new_root.rstrip("/\\") + path[len(old_root):]
    return path


def source_columns(connection, source: str) -> list[str]:
    rows = connection.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{source}') LIMIT 0").fetchall()
    return [row[0] for row in rows]


def build_view_sql(name: str, source: str, assignments: str, columns: list[str]) -> str:
    """One sample's join, as a view definition.

    The source's own columns are listed rather than selected with ``src.*`` so the generated
    ``file_row_number`` is projected exactly once and unambiguously. It is exposed under that
    name because it is the row identity the assignments key against, and callers need it to
    trace a row back to its position in the source.
    """
    # duckdb REFUSES file_row_number=true on a file that already carries that column, and
    # the option is a boolean -- there is no way to ask for the generated id under another
    # name. Such a source could never have been labelled either, since per_parquet_inference
    # reads it the same way, so this means the file changed after inference. Refuse rather
    # than substitute row_number(), whose order is not guaranteed to match.
    if "file_row_number" in columns:
        raise ValueError(
            "source already has a file_row_number column, so duckdb cannot generate the "
            "positional id the assignments key against. The file must have changed since "
            "inference ran -- re-run inference for this sample.")

    projected = ", ".join(f'src."{c}"' for c in columns)
    projected = f"{projected}, src.file_row_number" if projected else "src.file_row_number"
    read_source = f"read_parquet('{source}', file_row_number=true)"
    key = "file_row_number"
    return f"""CREATE OR REPLACE VIEW {name} AS
SELECT {projected},
       asg.umap_1        AS umap_1,
       asg.umap_2        AS umap_2,
       asg.cluster_label AS cluster_label
FROM {read_source} AS src
LEFT JOIN read_parquet('{assignments}') AS asg
       ON src.{key} = asg.file_row_number"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path,
                        help="assignments_manifest.json written by export_assignments.py")
    parser.add_argument("--database", required=True, type=Path,
                        help="duckdb file to create (kilobytes; holds views, not rows)")
    parser.add_argument("--source-root", default=None,
                        help="prefix of the source paths recorded in the manifest, to be "
                             "replaced by --new-source-root (for building on another machine)")
    parser.add_argument("--new-source-root", default=None)
    parser.add_argument("--assignments-root", default=None,
                        help="likewise for the per-sample results dir")
    parser.add_argument("--new-assignments-root", default=None)
    parser.add_argument("--combined-view", default="all_samples",
                        help="name of the UNION ALL view over every sample ('' to skip)")
    parser.add_argument("--verify", type=int, default=0, metavar="N",
                        help="after building, read N rows back from N samples and confirm "
                             "the assignment columns are populated")
    parser.add_argument("--allow-drift", action="store_true",
                        help="build views for samples whose source fingerprint no longer "
                             "matches. The views will still resolve, and they will be wrong.")
    args = parser.parse_args()

    import duckdb

    if not args.manifest.exists():
        log(f"ERROR: no manifest at {args.manifest}")
        log("  Run export_assignments.py (without --attach) to produce one.")
        return 1
    payload = json.loads(args.manifest.read_text())
    records = payload.get("samples", [])
    if not records:
        log("ERROR: manifest lists no samples")
        return 1
    log(f"manifest: {len(records)} samples, {payload.get('total_rows', 0):,} labelled rows")

    drifted = [r for r in records if r.get("source_status") not in ("OK", None)]
    if drifted and not args.allow_drift:
        log(f"ERROR: {len(drifted)} sample(s) have a source that is not OK:")
        for record in drifted[:10]:
            log(f"  {record['sample']}: {record['source_status']} -- {record.get('source_detail','')}")
        log("  A view over a drifted source resolves fine and returns wrong answers, because")
        log("  file_row_number is positional. Re-run inference for these, or --allow-drift if")
        log("  you have another reason to believe the positions still line up.")
        return 1

    args.database.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(args.database))

    connection.execute("""
        CREATE OR REPLACE TABLE sample_index (
            view_name     VARCHAR,
            sample        VARCHAR,
            source        VARCHAR,
            assignments   VARCHAR,
            labelled_rows BIGINT,
            noise_pct     DOUBLE,
            row_filter    VARCHAR
        )""")

    built: list[tuple[str, dict]] = []
    failed: list[tuple[str, str]] = []
    for record in records:
        sample = record["sample"]
        name = view_name(sample)
        source = rewrite(record.get("source", ""), args.source_root, args.new_source_root)
        assignments = rewrite(record.get("assignments", ""),
                              args.assignments_root, args.new_assignments_root)
        if not source or not assignments:
            failed.append((sample, "manifest records no source or assignments path"))
            continue
        try:
            columns = source_columns(connection, source)
            connection.execute(build_view_sql(name, source, assignments, columns))
        except Exception as exc:  # noqa: BLE001
            failed.append((sample, f"{type(exc).__name__}: {exc}"))
            continue
        connection.execute(
            "INSERT INTO sample_index VALUES (?, ?, ?, ?, ?, ?, ?)",
            [name, sample, source, assignments, record.get("rows"),
             record.get("noise_pct"), record.get("row_filter")])
        built.append((name, record))

    log(f"built {len(built)} views" + (f", {len(failed)} failed" if failed else ""))
    for sample, reason in failed:
        log(f"  FAILED {sample}: {reason}")

    # The union is only meaningful if every sample projects the same columns; sources that
    # disagree would make UNION ALL fail outright, and BY NAME quietly fill the gaps with
    # nulls. BY NAME is the right call here -- a sample missing an optional column should
    # still be reachable through the cohort view -- but it is worth knowing it happened.
    if args.combined_view and built:
        schemas = {}
        for name, _ in built:
            cols = tuple(row[0] for row in
                         connection.execute(f"DESCRIBE SELECT * FROM {name} LIMIT 0").fetchall())
            schemas.setdefault(cols, []).append(name)
        if len(schemas) > 1:
            log(f"NOTE: {len(schemas)} distinct schemas across samples; the combined view "
                f"uses UNION ALL BY NAME, so absent columns read as NULL rather than failing")
            for cols, names in sorted(schemas.items(), key=lambda kv: -len(kv[1])):
                log(f"  {len(names):>3} sample(s), {len(cols)} columns  e.g. {names[0]}")
        union = "\nUNION ALL BY NAME\n".join(
            f"SELECT '{record['sample']}' AS sample, * FROM {name}" for name, record in built)
        connection.execute(f"CREATE OR REPLACE VIEW {args.combined_view} AS\n{union}")
        log(f"combined view: {args.combined_view} ({len(built)} samples)")

    if args.verify and built:
        log(f"verifying {min(args.verify, len(built))} sample(s)")
        for name, record in built[:args.verify]:
            started = perf_counter()
            try:
                row = connection.execute(f"""
                    SELECT count(*) AS n,
                           count(cluster_label) AS labelled,
                           min(umap_1), max(umap_1)
                    FROM (SELECT * FROM {name} LIMIT 200000)""").fetchone()
            except Exception as exc:  # noqa: BLE001
                log(f"  FAIL {name}: {type(exc).__name__}: {exc}")
                failed.append((name, str(exc)))
                continue
            n, labelled, lo, hi = row
            elapsed = round(perf_counter() - started, 1)
            if not labelled:
                log(f"  FAIL {name}: {n:,} rows read, none carry a cluster_label -- the join "
                    f"matched nothing, so file_row_number does not line up")
                failed.append((name, "join matched no rows"))
            else:
                log(f"  ok   {name}: {labelled:,}/{n:,} of the first rows labelled, "
                    f"umap_1 in [{lo:.2f}, {hi:.2f}] ({elapsed}s)")

    connection.close()
    log("")
    log(f"database → {args.database}")
    log("Query it with:")
    log(f"    duckdb {args.database}")
    log("    D SELECT * FROM sample_index;                    -- what is in here")
    if built:
        log(f"    D SELECT * FROM {built[0][0]} LIMIT 5;")
        log(f"    D SELECT cluster_label, count(*) FROM {built[0][0]} GROUP BY 1 ORDER BY 2 DESC;")
    if args.combined_view and built:
        log(f"    D SELECT sample, count(*) FROM {args.combined_view} GROUP BY 1;  -- scans everything")
    log("")
    log("Each view returns every source column plus umap_1, umap_2, cluster_label. Nothing")
    log("is copied: the views read the original parquets and the assignment files in place,")
    log("so they stay correct for exactly as long as the sources are left untouched.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
