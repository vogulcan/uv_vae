from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from uv_vae.data import connect_duckdb, get_row_count
from uv_vae.training import (
    DEFAULT_TRAINING_FEATURE_SPEC_PATH,
    DEFAULT_TRAINING_ROW_FILTER,
    DEFAULT_TRAINING_SAMPLE_ROWS,
    TrainingConfig,
    train,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the VAE")
    parser.add_argument(
        "--parquet-path",
        default="data/ddbR9-b2-ppm0029.featuremap.parquet",
        help="Path to the parquet feature map",
    )
    parser.add_argument(
        "--feature-spec-path",
        default=DEFAULT_TRAINING_FEATURE_SPEC_PATH,
        help="Path to the ML feature spec file",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where run artifacts will be written",
    )
    parser.add_argument(
        "--row-filter",
        default=DEFAULT_TRAINING_ROW_FILTER,
        help="DuckDB SQL filter used to select parquet rows for training",
    )
    sample_group = parser.add_mutually_exclusive_group()
    sample_group.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help=f"Number of filtered rows to sample. Defaults to {DEFAULT_TRAINING_SAMPLE_ROWS:,} when omitted.",
    )
    sample_group.add_argument(
        "--use-all",
        action="store_true",
        help="Use all filtered rows instead of sampling a subset.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument(
        "--hidden-dims",
        default="256,128",
        help="Comma-separated hidden dimensions",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--kl-weight", type=float, default=0.05)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=None)

    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] == "train":
        args_list = args_list[1:]
    return parser.parse_args(args_list)


def resolve_requested_sample_rows(args: argparse.Namespace) -> int | None:
    if args.use_all:
        return None
    return args.sample_rows or DEFAULT_TRAINING_SAMPLE_ROWS


def resolve_training_sample_rows(
    parquet_path: str | Path,
    row_filter: str,
    requested_sample_rows: int | None,
    threads: int | None,
) -> tuple[int, int]:
    with connect_duckdb(threads=threads) as conn:
        eligible_rows = get_row_count(
            conn,
            parquet_path,
            where=row_filter,
        )
    if eligible_rows <= 0:
        raise RuntimeError(f"No rows matched the training filter: {row_filter}")
    if requested_sample_rows is None:
        return eligible_rows, eligible_rows
    return eligible_rows, min(requested_sample_rows, eligible_rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    requested_sample_rows = resolve_requested_sample_rows(args)
    eligible_rows, actual_sample_rows = resolve_training_sample_rows(
        parquet_path=args.parquet_path,
        row_filter=args.row_filter,
        requested_sample_rows=requested_sample_rows,
        threads=args.threads,
    )
    if requested_sample_rows is not None and actual_sample_rows < requested_sample_rows:
        print(
            (
                f"WARNING: requested sample_rows={requested_sample_rows:,} exceeds eligible_rows={eligible_rows:,}; "
                f"using sample_rows={actual_sample_rows:,}"
            ),
            file=sys.stderr,
            flush=True,
        )

    config = TrainingConfig(
        parquet_path=args.parquet_path,
        feature_spec_path=args.feature_spec_path,
        output_dir=args.output_dir,
        row_filter=args.row_filter,
        sample_rows=actual_sample_rows,
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        hidden_dims=[int(part) for part in args.hidden_dims.split(",") if part],
        learning_rate=args.learning_rate,
        kl_weight=args.kl_weight,
        train_fraction=args.train_fraction,
        seed=args.seed,
        threads=args.threads,
    )
    run_dir = train(config)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "eligible_rows": eligible_rows,
                "sample_rows": actual_sample_rows,
                "requested_sample_rows": requested_sample_rows,
                "row_filter": args.row_filter,
            },
            indent=2,
        )
    )
    return 0
