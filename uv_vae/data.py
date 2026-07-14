from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl
import pyarrow.parquet as pq

from uv_vae.features import FeatureSpec


def quote_ident(name: str) -> str:
    return f'"{name.replace("\"", "\"\"")}"'


def _quote_sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def connect_duckdb(
    threads: int | None = None,
    database: str | Path = ":memory:",
    temp_directory: str | Path | None = None,
    memory_limit: str | None = None,
) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(database=str(database))
    if threads:
        conn.execute(f"SET threads = {int(threads)}")
    if temp_directory is not None:
        temp_path = Path(temp_directory)
        temp_path.mkdir(parents=True, exist_ok=True)
        conn.execute(f"SET temp_directory = {_quote_sql_string(str(temp_path))}")
    if memory_limit:
        conn.execute(f"SET memory_limit = {_quote_sql_string(memory_limit)}")
    return conn


def build_parquet_select_sql(
    select_sql: str,
    where: str | None = None,
    limit: int | None = None,
) -> str:
    sql = f"{select_sql} FROM read_parquet(?)"
    if where:
        sql += f" WHERE {where}"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return sql


def get_row_count(
    conn: duckdb.DuckDBPyConnection,
    parquet_path: str | Path,
    where: str | None = None,
) -> int:
    if where is None:
        result = conn.execute(
            "SELECT num_rows FROM parquet_file_metadata(?)",
            [str(parquet_path)],
        ).fetchone()
    else:
        result = conn.execute(
            build_parquet_select_sql("SELECT count(*)", where=where),
            [str(parquet_path)],
        ).fetchone()
    if result is None:
        raise RuntimeError(f"Unable to read parquet metadata from {parquet_path}")
    return int(result[0])


def get_non_null_counts(
    conn: duckdb.DuckDBPyConnection,
    parquet_path: str | Path,
    feature_names: list[str],
    where: str | None = None,
) -> dict[str, int]:
    select_list = ", ".join(
        f"count({quote_ident(name)}) AS {quote_ident(name)}" for name in feature_names
    )
    row = conn.execute(
        build_parquet_select_sql(f"SELECT {select_list}", where=where),
        [str(parquet_path)],
    ).fetchone()
    if row is None:
        raise RuntimeError("DuckDB returned no row for the non-null feature count query")
    return dict(zip(feature_names, map(int, row), strict=True))


def sample_frame(
    conn: duckdb.DuckDBPyConnection,
    parquet_path: str | Path,
    feature_names: list[str],
    sample_rows: int,
    seed: int,
    where: str | None = None,
) -> pl.DataFrame:
    select_list = ", ".join(quote_ident(name) for name in feature_names)
    if where:
        sql = f"""
            SELECT {select_list}
            FROM (
                SELECT {select_list}
                FROM read_parquet(?)
                WHERE {where}
            ) AS filtered_rows
            USING SAMPLE reservoir({int(sample_rows)} ROWS)
            REPEATABLE ({int(seed)})
        """
    else:
        sql = f"""
            SELECT {select_list}
            FROM read_parquet(?)
            USING SAMPLE reservoir({int(sample_rows)} ROWS)
            REPEATABLE ({int(seed)})
        """
    return conn.execute(sql, [str(parquet_path)]).pl()


def stream_parquet_batches(
    conn: duckdb.DuckDBPyConnection,
    parquet_path: str | Path,
    select_columns: list[str],
    rows_per_batch: int,
    where: str | None = None,
    limit: int | None = None,
):
    available_columns = set(pq.read_schema(parquet_path).names)
    matched_columns = [name for name in select_columns if name in available_columns]
    if not matched_columns:
        raise RuntimeError(
            f"None of the requested columns {select_columns} exist in {parquet_path}. "
            f"Available columns: {sorted(available_columns)}"
        )
    select_list = ", ".join(quote_ident(name) for name in matched_columns)
    sql = build_parquet_select_sql(
        f"SELECT {select_list}",
        where=where,
        limit=limit,
    )
    return conn.execute(sql, [str(parquet_path)]).fetch_record_batch(rows_per_batch=rows_per_batch)


def split_specs(
    specs: list[FeatureSpec],
    non_null_counts: dict[str, int],
) -> tuple[list[FeatureSpec], list[FeatureSpec], list[str]]:
    active_specs = [spec for spec in specs if non_null_counts.get(spec.name, 0) > 0]
    dropped_specs = [spec.name for spec in specs if non_null_counts.get(spec.name, 0) == 0]
    categorical_specs = [spec for spec in active_specs if spec.is_categorical]
    numeric_specs = [spec for spec in active_specs if spec.is_numeric]
    return categorical_specs, numeric_specs, dropped_specs
