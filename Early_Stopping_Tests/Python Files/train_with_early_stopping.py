"""Train the VAE with ELBO + active-unit early stopping.

Additive entry point: the stock trainer (`train.py`) is untouched. This script takes the
same arguments plus `--patience`, `--min-delta` and `--active-unit-threshold`, and writes
the same run artifacts plus `diagnostics_report.json`.

Set `--epochs` to a generous ceiling and let `--patience` decide when to stop:

    python "Early_Stopping_Tests/Python Files/train_with_early_stopping.py" \
        --parquet-path <parquet> --use-all \
        --epochs 100 --patience 8 --output-dir artifacts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Locate the uv_vae package: this file lives in Early_Stopping_Tests/Python Files/,
# so parents[2] is the PURE Files root, and uv_vae/ is one level below that.
REPO_ROOT = Path(__file__).resolve().parents[2] / "uv_vae"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os

import torch

# cuML accelerates sklearn/UMAP/HDBSCAN -- none of which the training path uses.
# Installing it here costs a CUDA context plus an RMM pool that competes with the
# trainer for the same card, so it is off unless explicitly asked for. Set
# UV_VAE_ENABLE_CUML=1 only when a downstream clustering stage runs in-process.
if torch.cuda.is_available() and os.environ.get("UV_VAE_ENABLE_CUML", "").strip() in {"1", "true", "yes"}:
    try:
        import cuml.accel
        cuml.accel.install()
    except ImportError:
        pass

from uv_vae.early_stopping import (
    DEFAULT_ACTIVE_UNIT_THRESHOLD,
    DEFAULT_KL_COLLAPSE_THRESHOLD,
    EarlyStoppingConfig,
    train_with_early_stopping,
)
from uv_vae.multi_parquet import DEFAULT_SHUFFLE_BUFFER_ROWS
from uv_vae.multi_streaming import DEFAULT_VAL_MAX_ROWS
from uv_vae.splitting import GLOBAL_SITE_HASH, STRATEGIES
from uv_vae.train_cli import resolve_requested_sample_rows, resolve_training_sample_rows
from uv_vae.training import (
    DEFAULT_TRAINING_FEATURE_SPEC_PATH,
    DEFAULT_TRAINING_ROW_FILTER,
    DEFAULT_TRAINING_SAMPLE_ROWS,
    TrainingConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the VAE with early stopping")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--parquet-path",
        help="Path to a single parquet feature map (no default: always pass this explicitly)",
    )
    source_group.add_argument(
        "--parquet-paths",
        nargs="+",
        metavar="PATH_OR_GLOB",
        help="Two or more per-sample parquet feature maps, or a glob matching them. "
        "Selects the interleaved multi-file trainer: every batch contains every "
        "sample in proportion to its share of the total filtered rows, and each "
        "file is read in shuffled row-group order so a batch is not one genomic "
        "window. Implies streaming; incompatible with --streaming/--use-all/"
        "--sample-rows.",
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
        "--kl-collapse-threshold",
        type=float,
        default=DEFAULT_KL_COLLAPSE_THRESHOLD,
        help="Per-dim KL below which a latent dim counts as collapsed (Lucas 2019). "
        "Diagnostics only -- does not affect the stopping decision.",
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
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Steps to linearly ramp lr from ~0 to target. Streaming only. "
        "Recommended when lr > 1e-3 or batch_size > 65536.",
    )

    multi = parser.add_argument_group(
        "multi-file interleaving", "Ignored unless --parquet-paths is given."
    )
    multi.add_argument(
        "--split-strategy",
        default=GLOBAL_SITE_HASH,
        choices=list(STRATEGIES),
        help="How to assign rows to train/val. global_site_hash puts a locus on the "
        "same side in EVERY sample -- the only clean holdout across a cohort of "
        "human genomes that share loci. per_sample_site_hash keeps a locus intact "
        "within a sample but still leaks across samples. per_sample_row_hash is "
        "cheapest (no CHROM/POS read) but leaks at the locus level and makes early "
        "stopping fire late. Default: %(default)s.",
    )
    multi.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help="Validation share for the interleaved trainer. Defaults to "
        "1 - --train-fraction.",
    )
    multi.add_argument(
        "--epoch-shards",
        type=int,
        default=1,
        help="Split each file's row groups into E disjoint strata and give epoch e "
        "stratum e. Every row is then seen exactly once per E epochs, each epoch "
        "costs 1/E of a pass, and each epoch is still a proportional mix of all "
        "samples. E=20 makes a full pass over ~4.75B rows cost 20 epochs.",
    )
    multi.add_argument(
        "--stats-cache-path",
        default=None,
        help="JSON file for per-file row counts and normalisation statistics. "
        "Statistics are cached per file, so adding a sample later costs one scan "
        "rather than rescanning everything.",
    )
    multi.add_argument(
        "--shuffle-buffer-rows",
        type=int,
        default=DEFAULT_SHUFFLE_BUFFER_ROWS,
        help="Post-filter rows each reader holds decoded and shuffled at once. "
        "Costs ~320 B/row per sample. Default: %(default)s.",
    )
    multi.add_argument(
        "--decode-workers",
        type=int,
        default=1,
        help="Threads decoding row groups concurrently across the per-sample readers. "
        "1 (default) keeps the original sequential path bit-for-bit. Higher values "
        "overlap the single-threaded encode/split stages, measured at ~47%% of decode "
        "time, with the parallel read stage. Batch composition is unchanged: readers "
        "share no state and results are reassembled in reader order.",
    )
    multi.add_argument(
        "--val-max-rows",
        type=int,
        default=DEFAULT_VAL_MAX_ROWS,
        help="Cap on validation rows read per epoch (0 = uncapped). A site-keyed "
        "split scatters validation rows through every row group, so an uncapped "
        "validation pass touches the whole dataset and costs more than a sharded "
        "training epoch. Default: %(default)s.",
    )

    args = parser.parse_args()

    # argparse cannot express "mutually exclusive with a whole other group", and
    # silently ignoring a flag the user passed is worse than refusing to start.
    if args.parquet_paths:
        conflicts = [
            name
            for name, value in (
                ("--streaming", args.streaming),
                ("--use-all", args.use_all),
                ("--sample-rows", args.sample_rows is not None),
            )
            if value
        ]
        if conflicts:
            parser.error(
                f"--parquet-paths already implies streaming and cannot be combined "
                f"with {', '.join(conflicts)}"
            )
    return args


def main() -> int:
    args = parse_args()

    early_stopping_config = EarlyStoppingConfig(
        patience=args.patience,
        min_delta=args.min_delta,
        active_unit_threshold=args.active_unit_threshold,
        kl_collapse_threshold=args.kl_collapse_threshold,
    )

    if args.parquet_paths:
        from uv_vae.multi_streaming import resolve_parquet_paths, train_interleaved
        from uv_vae.splitting import SplitConfig

        parquet_paths = resolve_parquet_paths(args.parquet_paths)
        val_fraction = (
            args.val_fraction
            if args.val_fraction is not None
            else round(1.0 - args.train_fraction, 10)
        )
        split_config = SplitConfig(
            strategy=args.split_strategy, val_fraction=val_fraction, seed=args.seed
        )
        print(
            f"Interleaving {len(parquet_paths)} parquet files "
            f"(split={split_config.strategy}, val_fraction={split_config.val_fraction}, "
            f"epoch_shards={args.epoch_shards})",
            file=sys.stderr,
            flush=True,
        )

        config = TrainingConfig(
            # TrainingConfig is deliberately not modified: it carries a single
            # parquet_path, and inference.py reads that field as a fallback when no
            # parquet is passed explicitly. Setting it to the first sample keeps
            # that fallback resolvable; the full list is recorded under
            # parquet_paths in every report.
            parquet_path=parquet_paths[0],
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
        run_dir = train_interleaved(
            config,
            parquet_paths=parquet_paths,
            early_stopping=early_stopping_config,
            split_config=split_config,
            epoch_shards=args.epoch_shards,
            stats_cache_path=args.stats_cache_path,
            shuffle_buffer_rows=args.shuffle_buffer_rows,
            decode_workers=args.decode_workers,
            val_max_rows=args.val_max_rows,
            input_dropout=args.input_dropout,
            hidden_dropout=args.hidden_dropout,
            test_parquet_path=args.test_parquet_path,
            convergence_rows=args.convergence_rows,
            warmup_steps=args.warmup_steps,
        )
        summary = json.loads((run_dir / "summary.json").read_text())
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "checkpoint_path": str(run_dir / "model.pt"),
                    "eligible_rows": summary.get("eligible_rows_in_parquet", 0),
                    "sample_count": summary.get("sample_count", 0),
                    "interleaved": True,
                    "row_filter": args.row_filter,
                    "sampling": summary.get("sampling", {}),
                    "early_stopping": summary.get("early_stopping", {}),
                    "final_epoch": summary.get("final_epoch", {}),
                    "wall_clock": summary.get("wall_clock", {}),
                },
                indent=2,
            )
        )
        return 0

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
            warmup_steps=args.warmup_steps,
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
                    "wall_clock": summary.get("wall_clock", {}),
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
                "wall_clock": summary.get("wall_clock", {}),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
