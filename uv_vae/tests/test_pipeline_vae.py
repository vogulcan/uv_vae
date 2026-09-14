from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from uv_vae.pipeline_vae import write_deduplicated_sample


def test_write_deduplicated_sample_skips_dedup_when_identity_columns_missing(tmp_path: Path) -> None:
    parquet_path = tmp_path / "demo.parquet"
    output_path = tmp_path / "sampled.parquet"
    table = pa.table({"feature_a": [1, 2, 3], "feature_b": [0.1, 0.2, 0.3]})
    pq.write_table(table, parquet_path)

    dedup_population, sampled_rows = write_deduplicated_sample(
        parquet_path=parquet_path,
        output_path=output_path,
        row_filter="1=1",
        selected_columns=["feature_a", "CHROM", "POS", "REF", "ALT"],
        sample_rows=None,
        seed=7,
        threads=1,
    )

    assert dedup_population == 3
    assert sampled_rows == 3
    written = pq.read_table(output_path)
    assert written.column_names == ["feature_a"]
