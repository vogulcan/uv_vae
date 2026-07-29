"""VAE training across many per-sample parquet featuremaps, proportionally interleaved.

This is assembly, not new mathematics.  ``_run_training_epoch``,
``run_val_epoch_with_diagnostics`` and ``_build_model`` are **imported** from
``streaming.py``, ``EarlyStoppingMonitor`` from ``early_stopping.py``, and
``seed_everything``/``write_json`` from ``training.py``.  Nothing is
reimplemented, so a 95-file run is numerically comparable to a single-file
streaming run by construction rather than by inspection, and ``streaming.py``
stays byte-identical so every prior sweep result remains valid.

What this module adds is where the batches come from -- see ``multi_parquet.py``
for the interleave and ``splitting.py`` for the train/val predicate.

It writes the same run artifacts in the same formats as the other three
trainers, so ``inference.LatentInference.from_checkpoint`` and the clustering
pipeline consume the output unchanged, plus a ``sampling`` block in
``training_report.json`` recording per-file row counts, interleave weights,
per-file batch draws, the split strategy and the validation plan.
"""

from __future__ import annotations

import glob
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from uv_vae import gpu_budget
from uv_vae.early_stopping import EarlyStoppingConfig, EarlyStoppingMonitor
from uv_vae.features import load_feature_specs
from uv_vae.multi_parquet import (
    DEFAULT_SHUFFLE_BUFFER_ROWS,
    InterleavedRowSource,
    RowEncoder,
    SampleSource,
)
from uv_vae.splitting import SplitConfig
from uv_vae.stats_cache import load_or_compute_stats
from uv_vae.streaming import (
    _build_model,
    _run_training_epoch,
    run_val_epoch_with_diagnostics,
)
from uv_vae.training import TrainingConfig, seed_everything, write_json

# Cap on validation rows read per epoch. See InterleavedRowSource.cap_rows: an
# uncapped site-keyed validation split costs a full pass over every file.
DEFAULT_VAL_MAX_ROWS = 5_000_000


class InterleavedDataset(IterableDataset):
    """Adapts :class:`InterleavedRowSource` to the torch DataLoader protocol.

    Use ``DataLoader(dataset, batch_size=None)``: the source already emits whole
    batches, so the loader must pass them through rather than re-batch them.
    """

    def __init__(self, source: InterleavedRowSource) -> None:
        self.source = source

    def set_epoch(self, epoch: int) -> None:
        self.source.set_epoch(epoch)

    def __iter__(self):
        for categorical, numeric, mask in self.source:
            yield (
                torch.from_numpy(categorical).long(),
                torch.from_numpy(numeric).float(),
                torch.from_numpy(mask).float(),
            )


def resolve_parquet_paths(patterns: list[str]) -> list[str]:
    """Expand globs and literal paths into a sorted, de-duplicated list.

    Sorted so the sample order -- and therefore the reported weights and the
    per-file draw allocation -- does not depend on shell glob ordering.
    """
    resolved: list[Path] = []
    for pattern in patterns:
        candidate = Path(pattern)
        if candidate.exists():
            resolved.append(candidate)
            continue
        # glob.glob handles absolute patterns; Path.glob does not.
        matches = sorted(Path(match) for match in glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No parquet files matched {pattern!r}")
        resolved.extend(matches)

    unique: list[str] = []
    seen: set[str] = set()
    for path in sorted(resolved, key=lambda p: p.name):
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(key)
    if not unique:
        raise FileNotFoundError(f"No parquet files matched {patterns!r}")
    return unique


def train_interleaved(
    config: TrainingConfig,
    parquet_paths: list[str],
    early_stopping: EarlyStoppingConfig | None = None,
    split_config: SplitConfig | None = None,
    epoch_shards: int = 1,
    stats_cache_path: str | None = None,
    shuffle_buffer_rows: int = DEFAULT_SHUFFLE_BUFFER_ROWS,
    val_max_rows: int = DEFAULT_VAL_MAX_ROWS,
    input_dropout: float = 0.0,
    hidden_dropout: float = 0.0,
    test_parquet_path: str | None = None,
    convergence_rows: int = 5000,
    warmup_steps: int = 0,
) -> Path:
    """Train on every file in ``parquet_paths``, proportionally interleaved."""
    early_stopping = early_stopping or EarlyStoppingConfig()
    split_config = split_config or SplitConfig(
        val_fraction=round(1.0 - config.train_fraction, 10), seed=config.seed
    )

    seed_everything(config.seed)
    run_dir = Path(config.output_dir) / datetime.now(UTC).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=False)

    feature_specs = load_feature_specs(config.feature_spec_path)

    # ---- per-file statistics (cached; one scan per file, ever) ----
    print(
        f"Collecting statistics over {len(parquet_paths)} parquet files ...",
        file=sys.stderr,
        flush=True,
    )
    stats = load_or_compute_stats(
        paths=list(parquet_paths),
        feature_specs=feature_specs,
        row_filter=config.row_filter,
        cache_path=stats_cache_path,
        threads=config.threads,
    )
    total_rows = stats.total_rows
    if total_rows == 0:
        raise RuntimeError(f"No rows matched the training filter: {config.row_filter}")

    sources = [
        SampleSource(sample_id=fs.sample_id, path=fs.path, rows=fs.rows)
        for fs in stats.per_file
    ]

    # ---- device, AMP, GPU ceiling ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if device.type == "cuda":
        torch.set_float32_matmul_precision("highest")
    budget_report = gpu_budget.apply()
    gpu_environment = gpu_budget.describe_environment()

    hidden_dims = [int(dim) for dim in config.hidden_dims]
    model = _build_model(
        stats.categorical_specs,
        stats.numeric_specs,
        hidden_dims,
        config.latent_dim,
        input_dropout=input_dropout,
        hidden_dropout=hidden_dropout,
    ).to(device)

    batch_size = config.batch_size
    batch_budget_report: dict | None = None
    if device.type == "cuda":
        per_row = gpu_budget.bytes_per_row(
            n_categorical=len(stats.categorical_specs),
            n_numeric=len(stats.numeric_specs),
            embedding_dims=model.config.embedding_dims,
            hidden_dims=hidden_dims,
            latent_dim=config.latent_dim,
            categorical_cardinalities=model.config.categorical_cardinalities,
            amp=use_amp,
        )
        batch_size, batch_budget_report = gpu_budget.resolve_batch_size(
            config.batch_size, per_row
        )
        if batch_budget_report["action"] != "kept":
            print(
                f"[gpu_budget] batch_size {config.batch_size:,} projects to "
                f"{batch_budget_report['projected_batch_gb']} GB; budget allows "
                f"{batch_budget_report['max_batch_rows']:,} rows "
                f"(policy={batch_budget_report['policy']}, "
                f"action={batch_budget_report['action']})",
                file=sys.stderr,
                flush=True,
            )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)
    warmup_scheduler = (
        torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps
        )
        if warmup_steps > 0
        else None
    )
    global_step = 0

    # ---- interleaved loaders ----
    encoder = RowEncoder(
        categorical_specs=stats.categorical_specs,
        numeric_specs=stats.numeric_specs,
        numeric_means=stats.numeric_means,
        numeric_stds=stats.numeric_stds,
    )
    common = dict(
        sources=sources,
        encoder=encoder,
        split_config=split_config,
        row_filter=config.row_filter,
        batch_size=batch_size,
        seed=config.seed,
        shuffle_buffer_rows=shuffle_buffer_rows,
    )
    train_source = InterleavedRowSource(
        split="train", epoch_shards=epoch_shards, epoch_varying=True, **common
    )
    val_source = InterleavedRowSource(
        split="val", epoch_shards=1, epoch_varying=False, **common
    )
    val_limits = val_source.cap_rows(val_max_rows, split_config.val_fraction)

    train_dataset = InterleavedDataset(train_source)
    val_dataset = InterleavedDataset(val_source)
    train_loader = DataLoader(train_dataset, batch_size=None)
    val_loader = DataLoader(val_dataset, batch_size=None)

    interleave = train_source.describe()

    # ---- convergence tracking ----
    convergence_tracker = None
    if test_parquet_path:
        from uv_vae.convergence import ConvergenceTracker

        print(
            f"Loading convergence test set ({convergence_rows:,} rows) ...",
            file=sys.stderr,
            flush=True,
        )
        convergence_tracker = ConvergenceTracker.from_parquet(
            parquet_path=test_parquet_path,
            categorical_specs=stats.categorical_specs,
            numeric_specs=stats.numeric_specs,
            numeric_means=stats.numeric_means,
            numeric_stds=stats.numeric_stds,
            device=device,
            row_filter=config.row_filter,
            max_rows=convergence_rows,
            seed=config.seed,
            threads=config.threads,
        )

    # ---- training ----
    monitor = EarlyStoppingMonitor(early_stopping)
    history: list[dict[str, float | int]] = []
    diagnostics_history: list[dict[str, object]] = []
    best_state: dict[str, torch.Tensor] | None = None

    train_rows_per_epoch = int(total_rows * config.train_fraction / max(1, epoch_shards))
    print(
        f"Interleaved training over {len(sources)} samples: {total_rows:,} filtered rows, "
        f"~{train_rows_per_epoch:,} train rows/epoch (epoch_shards={epoch_shards}), "
        f"~{train_rows_per_epoch // max(1, batch_size):,} batches/epoch, "
        f"val capped at {val_max_rows:,} rows, split={split_config.strategy}, "
        f"device={device}, amp={use_amp}",
        file=sys.stderr,
        flush=True,
    )

    progress = tqdm(range(1, config.epochs + 1), desc="epochs", leave=False)
    for epoch in progress:
        train_dataset.set_epoch(epoch)

        train_metrics, steps = _run_training_epoch(
            model=model,
            loader=train_loader,
            device=device,
            kl_weight=config.kl_weight,
            optimizer=optimizer,
            scaler=scaler,
            warmup_scheduler=warmup_scheduler,
            warmup_steps=warmup_steps,
            global_step=global_step,
        )
        global_step += steps

        val_metrics, diagnostics = run_val_epoch_with_diagnostics(
            model=model,
            loader=val_loader,
            device=device,
            kl_weight=config.kl_weight,
            active_unit_threshold=early_stopping.active_unit_threshold,
            kl_collapse_threshold=early_stopping.kl_collapse_threshold,
            use_amp=use_amp,
        )

        val_loss = val_metrics["total_loss"]
        active_units = int(diagnostics["active_units"])

        convergence_metrics = None
        if convergence_tracker is not None:
            convergence_metrics = convergence_tracker.evaluate_epoch(
                model, epoch, use_amp=use_amp
            )

        epoch_metrics: dict[str, float | int] = {
            "epoch": epoch,
            "train_total_loss": train_metrics["total_loss"],
            "train_numeric_loss": train_metrics["numeric_loss"],
            "train_categorical_loss": train_metrics["categorical_loss"],
            "train_kl_loss": train_metrics["kl_loss"],
            "val_total_loss": val_metrics["total_loss"],
            "val_numeric_loss": val_metrics["numeric_loss"],
            "val_categorical_loss": val_metrics["categorical_loss"],
            "val_kl_loss": val_metrics["kl_loss"],
            "active_units": active_units,
            "kl_total": float(diagnostics["kl_total"]),
            "mean_kl": float(diagnostics["mean_kl"]),
            "collapsed_dims": int(diagnostics["collapsed_dims"]),
            "collapsed_pct": float(diagnostics["collapsed_pct"]),
            "train_rows": int(train_source.last_report.rows_emitted),
            "val_rows": int(val_source.last_report.rows_emitted),
        }
        if convergence_metrics and "procrustes_distance" in convergence_metrics:
            epoch_metrics["procrustes_distance"] = convergence_metrics["procrustes_distance"]
            epoch_metrics["linear_cka"] = convergence_metrics["linear_cka"]
            epoch_metrics["trustworthiness"] = convergence_metrics["trustworthiness"]
        history.append(epoch_metrics)
        diagnostics_history.append({"epoch": epoch, **diagnostics})

        if monitor.is_new_best(val_loss):
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

        should_stop = monitor.update(
            epoch=epoch, val_loss=val_loss, active_units=active_units
        )

        postfix = dict(
            train=f"{train_metrics['total_loss']:.4f}",
            val=f"{val_loss:.4f}",
            au=active_units,
            stag=monitor.stagnant_epochs,
        )
        if convergence_metrics and "procrustes_distance" in convergence_metrics:
            postfix["proc"] = f"{convergence_metrics['procrustes_distance']:.4f}"
        progress.set_postfix(**postfix)

        if should_stop:
            progress.close()
            print(
                f"Early stopping at epoch {epoch}: {monitor.stop_reason}. "
                f"Restoring best weights from epoch {monitor.best_epoch} "
                f"(val_total_loss={monitor.best_val_loss:.6f}).",
                file=sys.stderr,
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_source.close()
    val_source.close()

    # ---- artifacts (identical contract to the other three trainers) ----
    epochs_run = len(history)
    early_stopping_report = {
        "enabled": early_stopping.enabled,
        "config": asdict(early_stopping),
        "stopped_early": monitor.stopped_early,
        "stop_reason": monitor.stop_reason,
        "epochs_requested": config.epochs,
        "epochs_run": epochs_run,
        "epochs_saved": config.epochs - epochs_run,
        "best_epoch": monitor.best_epoch,
        "best_val_total_loss": monitor.best_val_loss,
        "final_active_units": int(history[-1]["active_units"]) if history else 0,
    }

    sampling_report = {
        "mode": "interleaved_multi_parquet",
        "parquet_paths": list(parquet_paths),
        "sample_count": len(sources),
        "total_filtered_rows": total_rows,
        "interleave": interleave.as_dict(),
        "train_rows_served": dict(train_source.last_report.rows_served),
        "val_rows_served": dict(val_source.last_report.rows_served),
        "train_batches_last_epoch": train_source.last_report.batches,
        "train_ragged_batches_last_epoch": train_source.last_report.ragged_batches,
        "train_ragged_rows_last_epoch": train_source.last_report.ragged_rows,
        "train_ragged_row_fraction_last_epoch": (
            train_source.last_report.ragged_rows / train_source.last_report.rows_emitted
            if train_source.last_report.rows_emitted
            else 0.0
        ),
        "epoch_shards": epoch_shards,
        "shuffle_buffer_rows": shuffle_buffer_rows,
        "split": split_config.as_dict(),
        "val_max_rows": val_max_rows,
        "val_row_group_limits": val_limits,
        "row_group_shuffling": True,
        "stats_cache_path": stats_cache_path,
    }

    feature_report = {
        "row_filter": config.row_filter,
        "eligible_rows_in_parquet": total_rows,
        "sample_rows": total_rows,
        "dropped_all_null_features": stats.dropped_all_null_features,
        "dropped_sample_null_features": [],
        "active_categorical_features": [s.name for s in stats.categorical_specs],
        "active_numeric_features": [s.name for s in stats.numeric_specs],
        "non_null_counts": stats.non_null_counts,
        "streaming": True,
        "interleaved": True,
        "rows_by_sample": stats.rows_by_sample,
    }
    preprocess_report = {
        "numeric_means": stats.numeric_means,
        "numeric_stds": stats.numeric_stds,
        "categorical_cardinalities": {
            s.name: s.cardinality for s in stats.categorical_specs
        },
    }
    training_report = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "config": asdict(config),
        "parquet_paths": list(parquet_paths),
        "early_stopping": early_stopping_report,
        "history": history,
        "streaming": True,
        "interleaved": True,
        "sampling": sampling_report,
        "input_dropout": input_dropout,
        "hidden_dropout": hidden_dropout,
        "amp": use_amp,
        "effective_batch_size": batch_size,
        "gpu_budget": budget_report.as_dict(),
        "gpu_batch_budget": batch_budget_report,
        "gpu_environment": gpu_environment,
    }
    diagnostics_report = {
        "active_unit_threshold": early_stopping.active_unit_threshold,
        "kl_collapse_threshold": early_stopping.kl_collapse_threshold,
        "latent_dim": config.latent_dim,
        "early_stopping": early_stopping_report,
        "per_epoch": diagnostics_history,
    }

    write_json(run_dir / "feature_report.json", feature_report)
    write_json(run_dir / "preprocess_report.json", preprocess_report)
    write_json(run_dir / "training_report.json", training_report)
    write_json(run_dir / "diagnostics_report.json", diagnostics_report)

    if convergence_tracker is not None and convergence_tracker.history:
        write_json(
            run_dir / "convergence_report.json",
            {
                "test_parquet_path": test_parquet_path,
                "convergence_rows": convergence_rows,
                "per_epoch": convergence_tracker.history,
            },
        )

    checkpoint_payload = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model.config),
        "training_config": asdict(config),
        "feature_report": feature_report,
        "preprocess_report": preprocess_report,
        "history": history,
        "early_stopping": early_stopping_report,
        "diagnostics_history": diagnostics_history,
        "sampling": sampling_report,
    }
    if convergence_tracker is not None:
        checkpoint_payload["convergence_history"] = convergence_tracker.history
    torch.save(checkpoint_payload, run_dir / "model.pt")

    summary = {
        "run_dir": str(run_dir),
        "device": str(device),
        "row_filter": config.row_filter,
        "eligible_rows_in_parquet": total_rows,
        "sample_rows": total_rows,
        "all_null_features": stats.dropped_all_null_features,
        "sample_only_null_features": [],
        "early_stopping": early_stopping_report,
        "final_epoch": history[-1] if history else {},
        "streaming": True,
        "interleaved": True,
        "sample_count": len(sources),
        "sampling": sampling_report,
        "convergence_tracking": convergence_tracker is not None,
        "effective_batch_size": batch_size,
        "gpu_budget_gb": budget_report.budget_gb if budget_report.enabled else None,
    }
    write_json(run_dir / "summary.json", summary)
    return run_dir
