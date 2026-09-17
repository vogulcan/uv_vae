from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from uv_vae.data import connect_duckdb, get_row_count
from uv_vae.training import (
    DEFAULT_TRAINING_FEATURE_SPEC_PATH,
    DEFAULT_TRAINING_ROW_FILTER,
    DEFAULT_TRAINING_SAMPLE_ROWS,
    TrainingConfig,
    train,
)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the standalone VAE model training pipeline")
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
        help="Directory where timestamped training artifacts will be written",
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
        help=(
            "Number of filtered rows to sample for training. "
            f"Defaults to {DEFAULT_TRAINING_SAMPLE_ROWS:,} when omitted."
        ),
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
    parser.add_argument("--threads", type=int, default=12)
    return parser.parse_args()


def resolve_requested_sample_rows(args: argparse.Namespace) -> int | None:
    if args.use_all:
        return None
    return args.sample_rows or DEFAULT_TRAINING_SAMPLE_ROWS


def resolve_training_sample_rows(
    parquet_path: Path,
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


def main() -> int:
    overall_start = perf_counter()
    args = parse_args()
    parquet_path = Path(args.parquet_path).resolve()
    feature_spec_path = Path(args.feature_spec_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    requested_sample_rows = resolve_requested_sample_rows(args)
    requested_sample_label = "all" if requested_sample_rows is None else f"{requested_sample_rows:,}"

    log(
        "Starting model training pipeline "
        f"parquet={parquet_path} feature_spec={feature_spec_path} output_dir={output_dir}"
    )
    log(
        f"Configured row_filter={args.row_filter!r}, sample_rows={requested_sample_label}, "
        f"epochs={args.epochs}, batch_size={args.batch_size}, latent_dim={args.latent_dim}, threads={args.threads}"
    )

    count_start = perf_counter()
    eligible_rows, actual_sample_rows = resolve_training_sample_rows(
        parquet_path=parquet_path,
        row_filter=args.row_filter,
        requested_sample_rows=requested_sample_rows,
        threads=args.threads,
    )
    if requested_sample_rows is not None and actual_sample_rows < requested_sample_rows:
        log(
            f"WARNING: requested sample_rows={requested_sample_rows:,} exceeds eligible_rows={eligible_rows:,}; "
            f"using sample_rows={actual_sample_rows:,}"
        )
    log(
        f"Eligible rows={eligible_rows:,}; training sample_rows={actual_sample_rows:,} "
        f"(resolved in {format_seconds(perf_counter() - count_start)})"
    )

    config = TrainingConfig(
        parquet_path=str(parquet_path),
        feature_spec_path=str(feature_spec_path),
        output_dir=str(output_dir),
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

    train_start = perf_counter()
    log("Launching VAE training")
    run_dir = train(config)
    log(f"Training finished in {format_seconds(perf_counter() - train_start)}")

    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    result = {
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "requested_sample_rows": requested_sample_rows,
        "sample_rows": actual_sample_rows,
        "eligible_rows": eligible_rows,
        "row_filter": args.row_filter,
        "final_epoch": summary.get("final_epoch", {}),
    }
    log(f"Model training pipeline completed in {format_seconds(perf_counter() - overall_start)}")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
