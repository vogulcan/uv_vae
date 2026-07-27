from __future__ import annotations

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from uv_vae.data import stream_parquet_batches


def test_stream_parquet_batches_skips_missing_columns(tmp_path):
    parquet_path = tmp_path / "input.parquet"
    table = pa.table({"feature_a": [1, 2], "feature_b": [3, 4]})
    pq.write_table(table, parquet_path)

    with duckdb.connect() as conn:
        reader = stream_parquet_batches(
            conn=conn,
            parquet_path=parquet_path,
            select_columns=["feature_a", "CHROM"],
            rows_per_batch=2,
        )
        batches = list(reader)

    assert len(batches) == 1
    assert batches[0].column_names == ["feature_a"]
