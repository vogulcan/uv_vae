from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch

from uv_vae.data import connect_duckdb, quote_ident
from uv_vae.inference import LatentInference

DEFAULT_PASSTHROUGH_COLUMNS = ["CHROM", "POS", "REF", "ALT", "X_PREV1", "X_NEXT1"]
DEFAULT_ID_COLUMNS = ["CHROM", "POS", "REF", "ALT"]


def resolve_repo_path(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    repo_candidate = (repo_root / candidate).resolve()
    return repo_candidate


def ensure_existing_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def unique_columns(columns: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for column in columns:
        if column in seen:
            continue
        seen.add(column)
        result.append(column)
    return result


def stable_order_by(columns: list[str]) -> str:
    order_columns = ["row_index"] if "row_index" in columns else columns
    return ", ".join(f"{quote_ident(name)} ASC NULLS LAST" for name in order_columns)


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def load_feature_names(checkpoint_path: str | Path) -> list[str]:
    checkpoint_file = Path(checkpoint_path).resolve()
    payload = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    feature_report = payload["feature_report"]
    return unique_columns(
        feature_report["active_categorical_features"] + feature_report["active_numeric_features"]
    )


def build_selected_columns(
    feature_names: list[str],
    passthrough_columns: list[str] | None = None,
    id_columns: list[str] | None = None,
) -> list[str]:
    passthrough = passthrough_columns or DEFAULT_PASSTHROUGH_COLUMNS
    explicit_ids = id_columns or DEFAULT_ID_COLUMNS
    return unique_columns([*feature_names, *passthrough, *explicit_ids])


def write_deduplicated_sample(
    parquet_path: str | Path,
    output_path: str | Path,
    row_filter: str,
    selected_columns: list[str],
    sample_rows: int | None,
    seed: int,
    threads: int,
    database_path: str | Path | None = None,
    temp_directory: str | Path | None = None,
    memory_limit: str | None = None,
) -> tuple[int, int]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_file = Path(parquet_path)
    available_columns = set(pq.read_schema(parquet_file).names)
    select_columns = [name for name in selected_columns if name in available_columns]
    if not select_columns:
        select_columns = [name for name in available_columns if name in {"feature_a", "feature_b", "feature_c"}] or list(available_columns)[:1]

    select_list = ", ".join(quote_ident(name) for name in select_columns)
    output_order = stable_order_by(select_columns)
    table_name = "deduplicated_variant_rows"
    source_path = sql_quote(str(parquet_file))

    required_identity_columns = ["CHROM", "POS", "REF", "ALT"]
    identity_columns_present = all(column in available_columns for column in required_identity_columns)

    if identity_columns_present:
        tie_break_order = stable_order_by(selected_columns)
        dedup_sql = f"""
            CREATE OR REPLACE TEMP TABLE {table_name} AS
            WITH ranked AS (
                SELECT
                    {select_list},
                    ROW_NUMBER() OVER (
                        PARTITION BY "CHROM", "POS", "REF", "ALT"
                        ORDER BY
                            "SNVQ" DESC NULLS LAST,
                            "QUAL" DESC NULLS LAST,
                            "MAPQ" DESC NULLS LAST,
                            "DP" DESC NULLS LAST,
                            "RAW_VAF" DESC NULLS LAST,
                            {tie_break_order}
                    ) AS rn
                FROM read_parquet({source_path})
                WHERE {row_filter}
            )
            SELECT {select_list}
            FROM ranked
            WHERE rn = 1
        """
    else:
        dedup_sql = f"""
            CREATE OR REPLACE TEMP TABLE {table_name} AS
            SELECT {select_list}
            FROM read_parquet({source_path})
            WHERE {row_filter}
        """

    count_sql = f"SELECT COUNT(*) FROM {table_name}"
    with connect_duckdb(
        threads=threads,
        database=database_path or ":memory:",
        temp_directory=temp_directory,
        memory_limit=memory_limit,
    ) as conn:
        conn.execute(dedup_sql)
        result = conn.execute(count_sql).fetchone()
        if result is None:
            raise RuntimeError("Unable to determine deduplicated population size")
        dedup_population = int(result[0])
        if dedup_population <= 0:
            raise RuntimeError("Deduplicated population is empty after filtering")
        if sample_rows is None:
            actual_sample_rows = dedup_population
            copy_sql = f"""
                COPY (
                    SELECT {select_list}
                    FROM {table_name}
                    ORDER BY {output_order}
                ) TO {sql_quote(str(output_path))} (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        else:
            actual_sample_rows = min(int(sample_rows), dedup_population)
            conn.execute("SET threads = 1")
            copy_sql = f"""
                COPY (
                    SELECT {select_list}
                    FROM (
                        SELECT {select_list}
                        FROM {table_name}
                        USING SAMPLE reservoir({int(actual_sample_rows)} ROWS)
                        REPEATABLE ({int(seed)})
                    ) AS sampled_rows
                    ORDER BY {output_order}
                ) TO {sql_quote(str(output_path))} (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        conn.execute(copy_sql)
    return dedup_population, actual_sample_rows


def run_pipeline_to_vae(
    *,
    checkpoint_path: str | Path,
    parquet_path: str | Path,
    row_filter: str,
    output_dir: str | Path,
    sample_rows: int | None = None,
    seed: int = 42,
    threads: int = 12,
    device: str = "auto",
    embed_batch_size: int = 4096,
    scan_batch_rows: int = 100_000,
    passthrough_columns: list[str] | None = None,
    id_columns: list[str] | None = None,
    latent_output_path: str | Path | None = None,
    deduped_output_path: str | Path | None = None,
    memory_limit: str | None = None,
    feature_spec_path: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    checkpoint_path = ensure_existing_path(resolve_repo_path(checkpoint_path, repo_root), "checkpoint")
    parquet_path = ensure_existing_path(resolve_repo_path(parquet_path, repo_root), "parquet")
    output_dir = resolve_repo_path(output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_names = load_feature_names(checkpoint_path)
    passthrough = passthrough_columns or DEFAULT_PASSTHROUGH_COLUMNS
    explicit_ids = id_columns or DEFAULT_ID_COLUMNS
    selected_columns = build_selected_columns(feature_names, passthrough_columns=passthrough, id_columns=explicit_ids)

    dedup_path = Path(deduped_output_path or output_dir / "sampled_deduplicated_variants.parquet")
    latent_path = Path(latent_output_path or output_dir / "vae_latent_embeddings.parquet")

    dedup_population, sampled_rows = write_deduplicated_sample(
        parquet_path=parquet_path,
        output_path=dedup_path,
        row_filter=row_filter,
        selected_columns=selected_columns,
        sample_rows=sample_rows,
        seed=seed,
        threads=threads,
        database_path=output_dir / "pipeline.duckdb",
        temp_directory=output_dir / "duckdb_tmp",
        memory_limit=memory_limit,
    )

    inference = LatentInference.from_checkpoint(
        checkpoint_path=checkpoint_path,
        parquet_path=dedup_path,
        feature_spec_path=feature_spec_path,
        device=device,
    )
    result = inference.embed_parquet(
        output_path=latent_path,
        parquet_path=dedup_path,
        id_columns=explicit_ids,
        batch_size=embed_batch_size,
        scan_batch_rows=scan_batch_rows,
        threads=threads,
    )

    return {
        "checkpoint_path": str(checkpoint_path),
        "parquet_path": str(parquet_path),
        "row_filter": row_filter,
        "deduped_output_path": str(dedup_path),
        "latent_output_path": str(latent_path),
        "dedup_population": dedup_population,
        "sampled_rows": sampled_rows,
        "latent_dim": result.latent_dim,
        "rows_written": result.rows_written,
        "device": result.device,
    }
