"""Train the VAE with ELBO + active-unit early stopping.

Additive entry point: the stock trainer (`train.py`) is untouched. This script takes the
same arguments plus `--patience`, `--min-delta` and `--active-unit-threshold`, and writes
the same run artifacts plus `diagnostics_report.json`.

Set `--epochs` to a generous ceiling and let `--patience` decide when to stop:

    python scripts/train_with_early_stopping.py \
        --parquet-path <parquet> --use-all \
        --epochs 100 --patience 8 --output-dir artifacts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

if torch.cuda.is_available():
    try:
        import cuml.accel
        cuml.accel.install()
    except ImportError:
        pass

from uv_vae.early_stopping import (
    DEFAULT_ACTIVE_UNIT_THRESHOLD,
    EarlyStoppingConfig,
    train_with_early_stopping,
)
from uv_vae.train_cli import resolve_requested_sample_rows, resolve_training_sample_rows
from uv_vae.training import (
    DEFAULT_TRAINING_FEATURE_SPEC_PATH,
    DEFAULT_TRAINING_ROW_FILTER,
    DEFAULT_TRAINING_SAMPLE_ROWS,
    TrainingConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the VAE with early stopping")
    parser.add_argument(
        "--parquet-path",
        required=True,
        help="Path to the parquet feature map (no default: always pass this explicitly)",
    )
    parser.add_argument("--feature-spec-path", default=DEFAULT_TRAINING_FEATURE_SPEC_PATH)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--row-filter", default=DEFAULT_TRAINING_ROW_FILTER)

    sample_group = parser.add_mutually_exclusive_group()
    sample_group.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help=f"Rows to sample. Defaults to {DEFAULT_TRAINING_SAMPLE_ROWS:,} when omitted.",
    )
    sample_group.add_argument(
        "--use-all",
        action="store_true",
        help="Train on all filtered rows instead of sampling a subset (loads into RAM).",
    )
    sample_group.add_argument(
        "--streaming",
        action="store_true",
        help="Stream all rows from parquet — flat memory, no OOM on large datasets.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Upper bound on epochs. With early stopping this is a ceiling, not a target.",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dims", default="256,128")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--kl-weight", type=float, default=0.05)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=None)

    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Consecutive stagnant epochs before stopping. 0 disables early stopping "
        "(diagnostics are still recorded).",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-3,
        help="Relative validation-loss improvement below which an epoch counts as stagnant.",
    )
    parser.add_argument(
        "--active-unit-threshold",
        type=float,
        default=DEFAULT_ACTIVE_UNIT_THRESHOLD,
        help="Posterior-mean variance above which a latent dim counts as active (Burda 2016).",
    )
    parser.add_argument(
        "--input-dropout",
        type=float,
        default=0.0,
        help="Dropout rate on the concatenated input (before the encoder). Streaming only.",
    )
    parser.add_argument(
        "--hidden-dropout",
        type=float,
        default=0.0,
        help="Dropout rate after each hidden-layer ReLU. Streaming only.",
    )
    parser.add_argument(
        "--test-parquet-path",
        default=None,
        help="Path to a held-out test parquet for per-epoch convergence tracking "
        "(Procrustes, CKA, trustworthiness between consecutive epochs).",
    )
    parser.add_argument(
        "--convergence-rows",
        type=int,
        default=5000,
        help="Rows to sample from the test parquet for convergence metrics.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    early_stopping_config = EarlyStoppingConfig(
        patience=args.patience,
        min_delta=args.min_delta,
        active_unit_threshold=args.active_unit_threshold,
    )

    if args.streaming:
        from uv_vae.streaming import train_with_early_stopping_streaming

        config = TrainingConfig(
            parquet_path=args.parquet_path,
            feature_spec_path=args.feature_spec_path,
            output_dir=args.output_dir,
            row_filter=args.row_filter,
            sample_rows=0,
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
        run_dir = train_with_early_stopping_streaming(
            config,
            early_stopping_config,
            input_dropout=args.input_dropout,
            hidden_dropout=args.hidden_dropout,
            test_parquet_path=args.test_parquet_path,
            convergence_rows=args.convergence_rows,
        )
        summary = json.loads((run_dir / "summary.json").read_text())
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "checkpoint_path": str(run_dir / "model.pt"),
                    "eligible_rows": summary.get("eligible_rows_in_parquet", 0),
                    "sample_rows": summary.get("sample_rows", 0),
                    "streaming": True,
                    "row_filter": args.row_filter,
                    "early_stopping": summary.get("early_stopping", {}),
                    "final_epoch": summary.get("final_epoch", {}),
                },
                indent=2,
            )
        )
        return 0

    requested_sample_rows = resolve_requested_sample_rows(args)
    eligible_rows, actual_sample_rows = resolve_training_sample_rows(
        parquet_path=args.parquet_path,
        row_filter=args.row_filter,
        requested_sample_rows=requested_sample_rows,
        threads=args.threads,
    )
    if requested_sample_rows is not None and actual_sample_rows < requested_sample_rows:
        print(
            f"WARNING: requested sample_rows={requested_sample_rows:,} exceeds "
            f"eligible_rows={eligible_rows:,}; using sample_rows={actual_sample_rows:,}",
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

    run_dir = train_with_early_stopping(config, early_stopping_config)

    summary = json.loads((run_dir / "summary.json").read_text())
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "checkpoint_path": str(run_dir / "model.pt"),
                "eligible_rows": eligible_rows,
                "sample_rows": actual_sample_rows,
                "row_filter": args.row_filter,
                "early_stopping": summary.get("early_stopping", {}),
                "final_epoch": summary.get("final_epoch", {}),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
