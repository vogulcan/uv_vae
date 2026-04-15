from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from uv_vae.inference import LatentInference


@dataclass(frozen=True)
class EmbeddingConfig:
    checkpoint_path: str
    parquet_path: str | None
    output_path: str
    batch_size: int
    scan_batch_rows: int
    threads: int | None
    limit: int | None
    where: str | None
    id_columns: list[str]
    feature_spec_path: str | None
    device: str | None


def add_embed_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Path to the saved model.pt checkpoint",
    )
    parser.add_argument(
        "--parquet-path",
        default=None,
        help="Optional parquet path to embed. Defaults to the training parquet path stored in the checkpoint",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path for the output parquet containing latent embeddings",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Model batch size used while encoding parquet rows",
    )
    parser.add_argument(
        "--scan-batch-rows",
        type=int,
        default=100_000,
        help="Number of parquet rows to stream from DuckDB per batch",
    )
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for embedding generation",
    )
    parser.add_argument(
        "--where",
        default=None,
        help="Optional DuckDB SQL filter expression applied before embedding",
    )
    parser.add_argument(
        "--id-columns",
        default="CHROM,POS,REF,ALT",
        help="Comma-separated columns to copy through to the output parquet",
    )
    parser.add_argument(
        "--feature-spec-path",
        default=None,
        help="Optional override for the feature spec file path stored in the checkpoint",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for latent inference",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate latent embeddings into a new parquet")
    add_embed_args(parser)

    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] == "embed":
        args_list = args_list[1:]
    return parser.parse_args(args_list)


def parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = EmbeddingConfig(
        checkpoint_path=args.checkpoint_path,
        parquet_path=args.parquet_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        scan_batch_rows=args.scan_batch_rows,
        threads=args.threads,
        limit=args.limit,
        where=args.where,
        id_columns=parse_csv_list(args.id_columns),
        feature_spec_path=args.feature_spec_path,
        device=args.device,
    )
    inference = LatentInference.from_checkpoint(
        checkpoint_path=config.checkpoint_path,
        feature_spec_path=config.feature_spec_path,
        parquet_path=config.parquet_path,
        device=config.device,
    )
    result = inference.embed_parquet(
        output_path=config.output_path,
        id_columns=config.id_columns,
        batch_size=config.batch_size,
        scan_batch_rows=config.scan_batch_rows,
        threads=config.threads,
        where=config.where,
        limit=config.limit,
    )
    print(json.dumps(asdict(result), indent=2))
    return 0
