"""Concatenate several featuremap parquet files into one.

Used to merge per-sample featuremaps into a single training input. Rows are copied
verbatim — no filtering, no deduplication, no column selection — so the merged file is
just the union of its inputs and the downstream row filter behaves exactly as it would on
a single sample.

    python scripts/combine_parquets.py \
        --inputs /path/sample_a.parquet /path/sample_b.parquet \
        --output /path/combined.featuremap.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine parquet feature maps into one file")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input parquet paths, in the order they should be concatenated",
    )
    parser.add_argument("--output", required=True, help="Path for the combined parquet")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--memory-limit",
        default="8GB",
        help="DuckDB memory cap so large copies spill to disk instead of exhausting RAM",
    )
    parser.add_argument(
        "--temp-directory",
        default=None,
        help="Optional DuckDB spill directory (defaults to alongside the output file)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_paths: list[Path] = []
    for raw in args.inputs:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input parquet not found: {path}")
        input_paths.append(path)

    if len(input_paths) < 2:
        print(
            f"NOTE: only {len(input_paths)} input given; the output will be a copy.",
            file=sys.stderr,
            flush=True,
        )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_directory = Path(args.temp_directory).expanduser().resolve() if args.temp_directory else (
        output_path.parent / "duckdb_tmp"
    )
    temp_directory.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect()
    conn.execute(f"SET threads = {int(args.threads)}")
    conn.execute(f"SET memory_limit = {sql_quote(args.memory_limit)}")
    conn.execute(f"SET temp_directory = {sql_quote(str(temp_directory))}")

    expected_rows = 0
    for path in input_paths:
        rows = conn.execute("SELECT COUNT(*) FROM read_parquet(?)", [str(path)]).fetchone()[0]
        columns = len(conn.execute("SELECT * FROM read_parquet(?) LIMIT 0", [str(path)]).description)
        print(f"  input: {path}  rows={int(rows):,}  columns={columns}", file=sys.stderr, flush=True)
        expected_rows += int(rows)

    file_list = ", ".join(sql_quote(str(path)) for path in input_paths)
    print(f"Writing combined parquet to {output_path}", file=sys.stderr, flush=True)
    conn.execute(
        f"""
        COPY (
            SELECT * FROM read_parquet([{file_list}], union_by_name = true)
        ) TO {sql_quote(str(output_path))} (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    combined_rows = int(
        conn.execute("SELECT COUNT(*) FROM read_parquet(?)", [str(output_path)]).fetchone()[0]
    )
    conn.close()

    print(
        f"Combined {len(input_paths)} files -> {output_path}\n"
        f"  expected rows: {expected_rows:,}\n"
        f"  combined rows: {combined_rows:,}",
        file=sys.stderr,
        flush=True,
    )
    if combined_rows != expected_rows:
        raise RuntimeError(
            f"Row count mismatch: inputs sum to {expected_rows:,} but combined file has "
            f"{combined_rows:,}. Refusing to report success."
        )

    print(str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
