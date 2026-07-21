"""Early-stopping instrumentation for VAE training.

This module is **purely additive**: it does not modify `uv_vae.training`. It reuses
`seed_everything`, `build_model` and `run_epoch` from there so the optimisation math is
identical to the stock pipeline, and it writes the exact same run artifacts
(`model.pt`, `feature_report.json`, `preprocess_report.json`, `training_report.json`,
`summary.json`) so `LatentInference.from_checkpoint` and the clustering pipeline consume
the output unchanged. Reverting to the stock behaviour means calling
`uv_vae.training.train` instead of `train_with_early_stopping` — nothing else changes.

What it adds on top of the stock loop, per epoch:

* per-dimension KL divergence on the validation split
* per-dimension variance of the posterior mean, and the **active unit (AU)** count from
  Burda et al. 2016 ("Importance Weighted Autoencoders"), where a unit is active when
  Cov_x(E[z_d | x]) exceeds a threshold (~0.01)
* a stopping rule that fires only when the validation ELBO has stagnated **and** the
  active-unit count has stopped moving, sustained over a patience window

The AND is the point: validation loss alone can flatten while latent dimensions are
still collapsing, so stopping on loss alone can freeze a model mid-collapse.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from uv_vae.data import connect_duckdb, get_non_null_counts, get_row_count, sample_frame
from uv_vae.features import load_feature_specs
from uv_vae.model import TabularVAE
from uv_vae.preprocess import prepare_tensors
from uv_vae.training import (
    TrainingConfig,
    build_model,
    run_epoch,
    seed_everything,
    write_json,
)

DEFAULT_ACTIVE_UNIT_THRESHOLD = 0.01


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Knobs for the stopping rule.

    patience <= 0 disables early stopping entirely, in which case this trainer still
    records all the diagnostics but runs the full `epochs` budget.
    """

    patience: int = 0
    min_delta: float = 1e-3
    active_unit_threshold: float = DEFAULT_ACTIVE_UNIT_THRESHOLD

    @property
    def enabled(self) -> bool:
        return self.patience > 0


def compute_latent_diagnostics(
    model: TabularVAE,
    loader: DataLoader,
    device: torch.device,
    active_unit_threshold: float = DEFAULT_ACTIVE_UNIT_THRESHOLD,
) -> dict[str, object]:
    """Per-dimension latent statistics over a full pass of `loader`.

    Accumulates in float64 so the variance stays stable over millions of rows, and
    streams rather than materialising every `mu` so memory does not scale with the
    validation split size.
    """
    was_training = model.training
    model.eval()
    latent_dim = model.config.latent_dim

    count = 0
    mu_sum = torch.zeros(latent_dim, dtype=torch.float64, device=device)
    mu_square_sum = torch.zeros(latent_dim, dtype=torch.float64, device=device)
    kl_sum = torch.zeros(latent_dim, dtype=torch.float64, device=device)

    with torch.inference_mode():
        for categorical_inputs, numeric_inputs, _ in loader:
            categorical_inputs = categorical_inputs.to(device, non_blocking=True)
            numeric_inputs = numeric_inputs.to(device, non_blocking=True)
            mu, logvar = model.encode(categorical_inputs, numeric_inputs)

            mu64 = mu.double()
            logvar64 = logvar.double()
            mu_sum += mu64.sum(dim=0)
            mu_square_sum += (mu64 * mu64).sum(dim=0)
            kl_sum += (-0.5 * (1.0 + logvar64 - mu64.pow(2) - logvar64.exp())).sum(dim=0)
            count += int(mu.shape[0])

    model.train(mode=was_training)

    if count == 0:
        raise RuntimeError("No batches were available for latent diagnostics")

    mu_mean = mu_sum / count
    mu_variance = torch.clamp((mu_square_sum / count) - mu_mean.pow(2), min=0.0)
    kl_per_dim = kl_sum / count
    active_mask = mu_variance > active_unit_threshold

    return {
        "active_units": int(active_mask.sum().item()),
        "latent_dim": latent_dim,
        "active_unit_threshold": float(active_unit_threshold),
        "active_unit_mask": [bool(value) for value in active_mask.cpu().tolist()],
        "posterior_mean_variance": [float(value) for value in mu_variance.cpu().tolist()],
        "kl_per_dim": [float(value) for value in kl_per_dim.cpu().tolist()],
        "kl_total": float(kl_per_dim.sum().item()),
        "rows_evaluated": count,
    }


class EarlyStoppingMonitor:
    """Tracks ELBO stagnation and active-unit stability jointly.

    An epoch only counts against patience when the validation loss failed to improve by
    more than `min_delta` (relative) *and* the active-unit count is unchanged from the
    previous epoch. Any real loss improvement, or any movement in the active-unit count,
    resets the counter — the model is still doing something.
    """

    def __init__(self, config: EarlyStoppingConfig) -> None:
        self.config = config
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.previous_active_units: int | None = None
        self.stagnant_epochs = 0
        self.stopped_early = False
        self.stop_reason: str | None = None

    def update(self, epoch: int, val_loss: float, active_units: int) -> bool:
        """Record one epoch. Returns True when training should stop."""
        if self.best_val_loss == float("inf"):
            improved = True
        else:
            relative_gain = (self.best_val_loss - val_loss) / max(abs(self.best_val_loss), 1e-8)
            improved = relative_gain > self.config.min_delta

        active_units_stable = (
            self.previous_active_units is not None and active_units == self.previous_active_units
        )

        is_new_best = val_loss < self.best_val_loss
        if is_new_best:
            self.best_val_loss = val_loss
            self.best_epoch = epoch

        if self.config.enabled:
            if improved or not active_units_stable:
                self.stagnant_epochs = 0
            else:
                self.stagnant_epochs += 1

        self.previous_active_units = active_units

        if self.config.enabled and self.stagnant_epochs >= self.config.patience:
            self.stopped_early = True
            self.stop_reason = (
                f"validation loss improved by <{self.config.min_delta:g} (relative) and "
                f"active units held at {active_units} for {self.stagnant_epochs} consecutive epochs"
            )
            return True
        return False

    def is_new_best(self, val_loss: float) -> bool:
        """Whether `val_loss` would become the new best. Call before `update`.

        Uses strict `<` to match `update`, so the snapshotted weights always correspond to
        the epoch reported as `best_epoch` (ties keep the earlier epoch).
        """
        return val_loss < self.best_val_loss


def train_with_early_stopping(
    config: TrainingConfig,
    early_stopping: EarlyStoppingConfig | None = None,
) -> Path:
    """Mirror of `uv_vae.training.train` with per-epoch latent diagnostics + early stopping.

    Sampling, preprocessing, model construction and the per-epoch optimisation are all
    delegated to the existing pipeline functions, so a run here is directly comparable to
    a stock run with the same seed and config.
    """
    early_stopping = early_stopping or EarlyStoppingConfig()

    seed_everything(config.seed)
    output_root = Path(config.output_dir)
    run_dir = output_root / datetime.now(UTC).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=False)

    feature_specs = load_feature_specs(config.feature_spec_path)
    feature_names = [spec.name for spec in feature_specs]
    hidden_dims = [int(part) for part in config.hidden_dims]

    with connect_duckdb(config.threads) as count_conn:
        total_rows = get_row_count(count_conn, config.parquet_path, where=config.row_filter)
        if total_rows == 0:
            raise RuntimeError(f"No rows matched the training filter: {config.row_filter}")
        sample_rows = min(config.sample_rows, total_rows)
        non_null_counts = get_non_null_counts(
            count_conn,
            config.parquet_path,
            feature_names,
            where=config.row_filter,
        )

    # DuckDB's REPEATABLE sampling is only deterministic with a single thread.
    with connect_duckdb(threads=1) as sample_conn:
        sampled_frame = sample_frame(
            conn=sample_conn,
            parquet_path=config.parquet_path,
            feature_names=feature_names,
            sample_rows=sample_rows,
            seed=config.seed,
            where=config.row_filter,
        )

    prepared = prepare_tensors(
        frame=sampled_frame,
        specs=feature_specs,
        non_null_counts=non_null_counts,
        total_rows=total_rows,
        train_fraction=config.train_fraction,
        seed=config.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("highest")

    train_loader = DataLoader(
        TensorDataset(prepared.train_cat, prepared.train_num, prepared.train_mask),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        TensorDataset(prepared.val_cat, prepared.val_num, prepared.val_mask),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = build_model(prepared, hidden_dims=hidden_dims, latent_dim=config.latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    monitor = EarlyStoppingMonitor(early_stopping)
    history: list[dict[str, float | int]] = []
    diagnostics_history: list[dict[str, object]] = []
    best_state: dict[str, torch.Tensor] | None = None

    progress = tqdm(range(1, config.epochs + 1), desc="epochs", leave=False)
    for epoch in progress:
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            kl_weight=config.kl_weight,
            optimizer=optimizer,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            kl_weight=config.kl_weight,
        )
        diagnostics = compute_latent_diagnostics(
            model=model,
            loader=val_loader,
            device=device,
            active_unit_threshold=early_stopping.active_unit_threshold,
        )

        val_loss = val_metrics["total_loss"]
        active_units = int(diagnostics["active_units"])

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
        }
        history.append(epoch_metrics)
        diagnostics_history.append({"epoch": epoch, **diagnostics})

        # Snapshot the best weights before the monitor mutates its running best.
        if monitor.is_new_best(val_loss):
            best_state = {
                name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            }

        should_stop = monitor.update(epoch=epoch, val_loss=val_loss, active_units=active_units)

        progress.set_postfix(
            train=f"{train_metrics['total_loss']:.4f}",
            val=f"{val_loss:.4f}",
            au=active_units,
            stag=monitor.stagnant_epochs,
        )

        if should_stop:
            progress.close()
            # stderr on purpose: callers capture stdout as JSON.
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

    feature_report = {
        "row_filter": config.row_filter,
        "eligible_rows_in_parquet": prepared.total_rows,
        "sample_rows": prepared.sample_size,
        "dropped_all_null_features": prepared.dropped_all_null_features,
        "dropped_sample_null_features": prepared.dropped_sample_null_features,
        "active_categorical_features": [spec.name for spec in prepared.categorical_specs],
        "active_numeric_features": [spec.name for spec in prepared.numeric_specs],
        "non_null_counts": prepared.non_null_counts,
    }
    preprocess_report = {
        "numeric_means": prepared.numeric_means,
        "numeric_stds": prepared.numeric_stds,
        "categorical_cardinalities": {
            spec.name: spec.cardinality for spec in prepared.categorical_specs
        },
    }
    training_report = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "config": asdict(config),
        "early_stopping": early_stopping_report,
        "history": history,
    }
    diagnostics_report = {
        "active_unit_threshold": early_stopping.active_unit_threshold,
        "latent_dim": config.latent_dim,
        "early_stopping": early_stopping_report,
        "per_epoch": diagnostics_history,
    }

    write_json(run_dir / "feature_report.json", feature_report)
    write_json(run_dir / "preprocess_report.json", preprocess_report)
    write_json(run_dir / "training_report.json", training_report)
    write_json(run_dir / "diagnostics_report.json", diagnostics_report)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model.config),
            "training_config": asdict(config),
            "feature_report": feature_report,
            "preprocess_report": preprocess_report,
            "history": history,
            "early_stopping": early_stopping_report,
            "diagnostics_history": diagnostics_history,
        },
        run_dir / "model.pt",
    )

    summary = {
        "run_dir": str(run_dir),
        "device": str(device),
        "row_filter": config.row_filter,
        "eligible_rows_in_parquet": prepared.total_rows,
        "sample_rows": prepared.sample_size,
        "all_null_features": prepared.dropped_all_null_features,
        "sample_only_null_features": prepared.dropped_sample_null_features,
        "early_stopping": early_stopping_report,
        "final_epoch": history[-1] if history else {},
    }
    write_json(run_dir / "summary.json", summary)
    return run_dir
