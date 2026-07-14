from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from uv_vae.data import connect_duckdb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Connect to a parquet file with DuckDB and inspect/read it")
    parser.add_argument("--parquet-path", required=True, help="Path to the parquet file")
    parser.add_argument("--row-filter", default=None, help="Optional DuckDB SQL WHERE clause, e.g. \"SNVQ > 30\"")
    parser.add_argument("--limit", type=int, default=10, help="Number of rows to preview")
    parser.add_argument("--threads", type=int, default=4, help="DuckDB thread count")
    parser.add_argument("--memory-limit", default=None, help="e.g. 16GB. Caps DuckDB memory so it spills to disk on an HPC node")
    parser.add_argument("--temp-directory", default=None, help="Scratch dir for spill files, e.g. $TMPDIR on the HPC node")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with connect_duckdb(
        threads=args.threads,
        memory_limit=args.memory_limit,
        temp_directory=args.temp_directory,
    ) as conn:
        schema = conn.execute(
            "SELECT * FROM read_parquet(?) LIMIT 0", [args.parquet_path]
        ).pl().schema
        print("Columns:")
        for name, dtype in schema.items():
            print(f"  {name}: {dtype}")

        count_sql = "SELECT COUNT(*) FROM read_parquet(?)"
        if args.row_filter:
            count_sql = f"SELECT COUNT(*) FROM read_parquet(?) WHERE {args.row_filter}"
        row_count = conn.execute(count_sql, [args.parquet_path]).fetchone()[0]
        print(f"\nRow count: {row_count}")

        preview_sql = "SELECT * FROM read_parquet(?)"
        if args.row_filter:
            preview_sql += f" WHERE {args.row_filter}"
        preview_sql += f" LIMIT {int(args.limit)}"

        df = conn.execute(preview_sql, [args.parquet_path]).pl()
        print(f"\nPreview ({len(df)} rows):")
        print(df)


if __name__ == "__main__":
    main()
