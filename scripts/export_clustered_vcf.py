from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from uv_vae.data import connect_duckdb, quote_ident

LOCUS_JOIN_KEYS = ["CHROM", "POS", "REF", "ALT"]


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remainder:.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes}m {remainder:.1f}s"


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def resolve_output_path(output_path: str | None) -> Path:
    if output_path is not None:
        target = Path(output_path).expanduser()
        return target if target.is_absolute() else (Path.cwd() / target).resolve()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return (Path.cwd() / f"clustered_variants_{timestamp}.vcf").resolve()


def parquet_columns(path: Path) -> set[str]:
    return set(pq.read_schema(path).names)


def detect_join_keys(source_columns: set[str], label_columns: set[str], join_on: str) -> list[str]:
    if join_on == "row_index":
        if "row_index" not in source_columns or "row_index" not in label_columns:
            raise RuntimeError("join-on=row_index requires row_index in both parquet files")
        return ["row_index"]
    if join_on == "locus":
        if not all(key in source_columns and key in label_columns for key in LOCUS_JOIN_KEYS):
            raise RuntimeError("join-on=locus requires CHROM, POS, REF, ALT in both parquet files")
        return LOCUS_JOIN_KEYS
    if "row_index" in source_columns and "row_index" in label_columns:
        return ["row_index"]
    if all(key in source_columns and key in label_columns for key in LOCUS_JOIN_KEYS):
        return LOCUS_JOIN_KEYS
    raise RuntimeError("Unable to auto-detect join keys; provide --join-on row_index or --join-on locus")


def parse_cluster_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError("--use-clusters must contain at least one integer cluster label")
    clusters: list[int] = []
    seen: set[int] = set()
    for part in parts:
        label = int(part)
        if label in seen:
            continue
        seen.add(label)
        clusters.append(label)
    return clusters


def maybe_cast_expr(columns: set[str], table_alias: str, column: str, sql_type: str, output_alias: str) -> str:
    if column in columns:
        return f'CAST({table_alias}.{quote_ident(column)} AS {sql_type}) AS {output_alias}'
    return f"CAST(NULL AS {sql_type}) AS {output_alias}"


def build_cluster_filter(
    cluster_label_column: str,
    table_alias: str | None,
    exclude_noise: bool,
    use_clusters: list[int] | None,
) -> str:
    qualified = quote_ident(cluster_label_column)
    if table_alias is not None:
        qualified = f"{table_alias}.{qualified}"
    predicates = [f"{qualified} IS NOT NULL"]
    if use_clusters is not None:
        cluster_values = ", ".join(str(int(label)) for label in use_clusters)
        predicates.append(f"{qualified} IN ({cluster_values})")
    elif exclude_noise:
        predicates.append(f"{qualified} >= 0")
    return " AND ".join(predicates)


def header_lines(reference: str | None) -> list[str]:
    lines = [
        "##fileformat=VCFv4.3",
        f"##fileDate={datetime.now(UTC).strftime('%Y%m%d')}",
        "##source=uv_vae_export_clustered_vcf",
        '##INFO=<ID=CLUSTER,Number=1,Type=Integer,Description="HDBSCAN cluster label">',
        '##INFO=<ID=CLPROB,Number=1,Type=Float,Description="HDBSCAN cluster membership probability">',
        '##INFO=<ID=UMAP1,Number=1,Type=Float,Description="UMAP component 1">',
        '##INFO=<ID=UMAP2,Number=1,Type=Float,Description="UMAP component 2">',
        '##INFO=<ID=SRC_FILT,Number=1,Type=Integer,Description="Original FILT column from source parquet">',
        '##FILTER=<ID=PASS,Description="Passed source filter or FILT=1">',
    ]
    if reference:
        lines.append(f"##reference={reference}")
    lines.append("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO")
    return lines


def build_info(record: dict[str, object], has_source_filt: bool) -> str:
    info_parts = [f"CLUSTER={int(record['cluster_label'])}"]
    cluster_probability = record.get("cluster_probability")
    if cluster_probability is not None:
        info_parts.append(f"CLPROB={float(cluster_probability):.6g}")
    umap_1 = record.get("umap_1")
    if umap_1 is not None:
        info_parts.append(f"UMAP1={float(umap_1):.6g}")
    umap_2 = record.get("umap_2")
    if umap_2 is not None:
        info_parts.append(f"UMAP2={float(umap_2):.6g}")
    if has_source_filt and record.get("src_filt") is not None:
        info_parts.append(f"SRC_FILT={int(record['src_filt'])}")
    return ";".join(info_parts)


def choose_filter(record: dict[str, object], has_source_filt: bool) -> str:
    if has_source_filt and record.get("src_filt") is not None:
        return "PASS" if int(record["src_filt"]) == 1 else "."
    return "PASS"


def choose_qual(record: dict[str, object], has_qual: bool) -> str:
    if not has_qual or record.get("qual") is None:
        return "."
    return f"{float(record['qual']):.6g}"


def count_rows(
    source_path: Path,
    labels_path: Path | None,
    row_filter: str | None,
    join_keys: list[str] | None,
    cluster_label_column: str,
    exclude_noise: bool,
    use_clusters: list[int] | None,
    threads: int,
    temp_directory: Path,
    memory_limit: str,
) -> int:
    source_where_clause = f"WHERE {row_filter}" if row_filter else ""
    if labels_path is None:
        label_filter = build_cluster_filter(cluster_label_column, None, exclude_noise, use_clusters)
        sql = f"""
            SELECT COUNT(*)
            FROM (
                SELECT *
                FROM read_parquet({sql_quote(str(source_path))})
                {source_where_clause}
            ) AS source
            WHERE {label_filter}
        """
    else:
        join_clause = " AND ".join(f'source.{quote_ident(key)} = labels.{quote_ident(key)}' for key in join_keys or [])
        label_filter = build_cluster_filter(cluster_label_column, "labels", exclude_noise, use_clusters)
        sql = f"""
            SELECT COUNT(*)
            FROM (
                SELECT *
                FROM read_parquet({sql_quote(str(source_path))})
                {source_where_clause}
            ) AS source
            INNER JOIN read_parquet({sql_quote(str(labels_path))}) AS labels
                ON {join_clause}
            WHERE {label_filter}
        """
    with connect_duckdb(threads=threads, temp_directory=temp_directory, memory_limit=memory_limit) as conn:
        result = conn.execute(sql).fetchone()
    return int(result[0]) if result is not None else 0


def stream_joined_records(
    source_path: Path,
    labels_path: Path | None,
    row_filter: str | None,
    join_keys: list[str] | None,
    cluster_label_column: str,
    exclude_noise: bool,
    use_clusters: list[int] | None,
    source_columns: set[str],
    label_columns: set[str] | None,
    threads: int,
    rows_per_batch: int,
    temp_directory: Path,
    memory_limit: str,
):
    source_where_clause = f"WHERE {row_filter}" if row_filter else ""
    if labels_path is None:
        source_cluster_probability = maybe_cast_expr(source_columns, "source", "cluster_probability", "FLOAT", "cluster_probability")
        source_umap_1 = maybe_cast_expr(source_columns, "source", "umap_1", "FLOAT", "umap_1")
        source_umap_2 = maybe_cast_expr(source_columns, "source", "umap_2", "FLOAT", "umap_2")
        source_qual = maybe_cast_expr(source_columns, "source", "QUAL", "DOUBLE", "qual")
        source_filt = maybe_cast_expr(source_columns, "source", "FILT", "INTEGER", "src_filt")
        label_filter = build_cluster_filter(cluster_label_column, None, exclude_noise, use_clusters)
        sql = f"""
            SELECT
                CAST(source."CHROM" AS VARCHAR) AS chrom,
                CAST(source."POS" AS BIGINT) AS pos,
                CAST(source."REF" AS VARCHAR) AS ref,
                CAST(source."ALT" AS VARCHAR) AS alt,
                {source_qual},
                {source_filt},
                CAST(source.{quote_ident(cluster_label_column)} AS INTEGER) AS cluster_label,
                {source_cluster_probability},
                {source_umap_1},
                {source_umap_2}
            FROM (
                SELECT *
                FROM read_parquet({sql_quote(str(source_path))})
                {source_where_clause}
            ) AS source
            WHERE {label_filter}
            ORDER BY chrom, pos, ref, alt
        """
    else:
        labels_columns = label_columns or set()
        source_qual = maybe_cast_expr(source_columns, "source", "QUAL", "DOUBLE", "qual")
        source_filt = maybe_cast_expr(source_columns, "source", "FILT", "INTEGER", "src_filt")
        labels_cluster_probability = maybe_cast_expr(
            labels_columns, "labels", "cluster_probability", "FLOAT", "cluster_probability"
        )
        labels_umap_1 = maybe_cast_expr(labels_columns, "labels", "umap_1", "FLOAT", "umap_1")
        labels_umap_2 = maybe_cast_expr(labels_columns, "labels", "umap_2", "FLOAT", "umap_2")
        join_clause = " AND ".join(f'source.{quote_ident(key)} = labels.{quote_ident(key)}' for key in join_keys or [])
        label_filter = build_cluster_filter(cluster_label_column, "labels", exclude_noise, use_clusters)
        sql = f"""
            SELECT
                CAST(source."CHROM" AS VARCHAR) AS chrom,
                CAST(source."POS" AS BIGINT) AS pos,
                CAST(source."REF" AS VARCHAR) AS ref,
                CAST(source."ALT" AS VARCHAR) AS alt,
                {source_qual},
                {source_filt},
                CAST(labels.{quote_ident(cluster_label_column)} AS INTEGER) AS cluster_label,
                {labels_cluster_probability},
                {labels_umap_1},
                {labels_umap_2}
            FROM (
                SELECT *
                FROM read_parquet({sql_quote(str(source_path))})
                {source_where_clause}
            ) AS source
            INNER JOIN read_parquet({sql_quote(str(labels_path))}) AS labels
                ON {join_clause}
            WHERE {label_filter}
            ORDER BY chrom, pos, ref, alt
        """
    with connect_duckdb(threads=threads, temp_directory=temp_directory, memory_limit=memory_limit) as conn:
        yield from conn.execute(sql).arrow(rows_per_batch)


def export_vcf(
    source_path: Path,
    labels_path: Path | None,
    output_path: Path,
    row_filter: str | None,
    join_keys: list[str] | None,
    cluster_label_column: str,
    exclude_noise: bool,
    use_clusters: list[int] | None,
    reference: str | None,
    threads: int,
    rows_per_batch: int,
    temp_directory: Path,
    memory_limit: str,
) -> int:
    has_source_filt = "FILT" in parquet_columns(source_path)
    has_qual = "QUAL" in parquet_columns(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for line in header_lines(reference):
            handle.write(line)
            handle.write("\n")
        for record_batch in stream_joined_records(
            source_path=source_path,
            labels_path=labels_path,
            row_filter=row_filter,
            join_keys=join_keys,
            cluster_label_column=cluster_label_column,
            exclude_noise=exclude_noise,
            use_clusters=use_clusters,
            source_columns=parquet_columns(source_path),
            label_columns=parquet_columns(labels_path) if labels_path is not None else None,
            threads=threads,
            rows_per_batch=rows_per_batch,
            temp_directory=temp_directory,
            memory_limit=memory_limit,
        ):
            table = record_batch.to_pydict()
            batch_size = len(table["chrom"])
            for index in range(batch_size):
                record = {key: value[index] for key, value in table.items()}
                handle.write(
                    "\t".join(
                        [
                            str(record["chrom"]),
                            str(int(record["pos"])),
                            ".",
                            str(record["ref"]),
                            str(record["alt"]),
                            choose_qual(record, has_qual),
                            choose_filter(record, has_source_filt),
                            build_info(record, has_source_filt),
                        ]
                    )
                )
                handle.write("\n")
                rows_written += 1
    return rows_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export clustered variants from parquet to a VCF file.")
    parser.add_argument("--parquet-path", required=True, help="Source parquet with CHROM, POS, REF, ALT")
    parser.add_argument(
        "--cluster-labels-path",
        default=None,
        help="Optional separate parquet containing cluster labels. If omitted, cluster labels are read from --parquet-path",
    )
    parser.add_argument("--output-path", default=None, help="Output .vcf path")
    parser.add_argument("--row-filter", default=None, help="Optional DuckDB SQL filter applied to the source parquet")
    parser.add_argument(
        "--join-on",
        choices=["auto", "row_index", "locus"],
        default="auto",
        help="How to join --cluster-labels-path onto --parquet-path when a separate labels parquet is provided",
    )
    parser.add_argument(
        "--cluster-label-column",
        default="cluster_label",
        help="Cluster-label column name in the source or labels parquet",
    )
    parser.add_argument(
        "--exclude-noise",
        action="store_true",
        help="Drop rows whose cluster label is -1",
    )
    parser.add_argument(
        "--use-clusters",
        default=None,
        help="Comma-separated cluster labels to export, for example 0,123,89",
    )
    parser.add_argument("--reference", default=None, help="Optional reference/genome-build string for the VCF header")
    parser.add_argument("--threads", type=int, default=12, help="DuckDB thread count")
    parser.add_argument("--rows-per-batch", type=int, default=100_000, help="Streaming batch size")
    parser.add_argument(
        "--duckdb-memory-limit",
        default="4GB",
        help="Memory limit passed to DuckDB so large joins can spill to disk",
    )
    parser.add_argument(
        "--duckdb-temp-dir",
        default=None,
        help="Optional DuckDB temp directory. Defaults to <output_dir>/duckdb_tmp",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.parquet_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Source parquet not found: {source_path}")
    labels_path = None
    if args.cluster_labels_path is not None:
        labels_path = Path(args.cluster_labels_path).expanduser().resolve()
        if not labels_path.is_file():
            raise FileNotFoundError(f"Cluster labels parquet not found: {labels_path}")
    use_clusters = parse_cluster_list(args.use_clusters)
    output_path = resolve_output_path(args.output_path)
    temp_directory = (
        Path(args.duckdb_temp_dir).expanduser().resolve()
        if args.duckdb_temp_dir is not None
        else output_path.parent / "duckdb_tmp"
    )
    source_columns = parquet_columns(source_path)
    required_source = {"CHROM", "POS", "REF", "ALT"}
    if not required_source.issubset(source_columns):
        raise RuntimeError(f"Source parquet must contain {sorted(required_source)}")
    join_keys = None
    if labels_path is None:
        if args.cluster_label_column not in source_columns:
            raise RuntimeError(
                f"Cluster label column {args.cluster_label_column!r} not found in source parquet and no --cluster-labels-path was given"
            )
    else:
        label_columns = parquet_columns(labels_path)
        if args.cluster_label_column not in label_columns:
            raise RuntimeError(f"Cluster label column {args.cluster_label_column!r} not found in labels parquet")
        join_keys = detect_join_keys(source_columns, label_columns, args.join_on)
    start = perf_counter()
    join_mode_label = "embedded" if labels_path is None else ",".join(join_keys or [])
    cluster_filter_label = "all non-null clusters" if use_clusters is None else ",".join(str(value) for value in use_clusters)
    log(
        f"Starting clustered VCF export source={source_path} labels={labels_path or source_path} "
        f"output={output_path} join={join_mode_label} clusters={cluster_filter_label}"
    )
    row_count = count_rows(
        source_path=source_path,
        labels_path=labels_path,
        row_filter=args.row_filter,
        join_keys=join_keys,
        cluster_label_column=args.cluster_label_column,
        exclude_noise=args.exclude_noise,
        use_clusters=use_clusters,
        threads=args.threads,
        temp_directory=temp_directory,
        memory_limit=args.duckdb_memory_limit,
    )
    if row_count == 0:
        raise RuntimeError("No rows matched the requested parquet/filter/cluster-label configuration")
    log(f"Matched {row_count:,} rows for VCF export")
    rows_written = export_vcf(
        source_path=source_path,
        labels_path=labels_path,
        output_path=output_path,
        row_filter=args.row_filter,
        join_keys=join_keys,
        cluster_label_column=args.cluster_label_column,
        exclude_noise=args.exclude_noise,
        use_clusters=use_clusters,
        reference=args.reference,
        threads=args.threads,
        rows_per_batch=args.rows_per_batch,
        temp_directory=temp_directory,
        memory_limit=args.duckdb_memory_limit,
    )
    log(f"Wrote {rows_written:,} VCF records to {output_path} in {format_seconds(perf_counter() - start)}")


if __name__ == "__main__":
    main()
