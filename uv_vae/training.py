from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from uv_vae.data import connect_duckdb, get_non_null_counts, get_row_count, sample_frame
from uv_vae.features import DEFAULT_FEATURE_SPEC_PATH, load_feature_specs
from uv_vae.model import TabularVAE, VAEConfig
from uv_vae.preprocess import PreparedTensors, infer_embedding_dim, prepare_tensors

DEFAULT_TRAINING_ROW_FILTER = "st = 'MIXED' AND et = 'MIXED' AND FILT = 1"
DEFAULT_TRAINING_SAMPLE_ROWS = 1_000_000
DEFAULT_TRAINING_FEATURE_SPEC_PATH = DEFAULT_FEATURE_SPEC_PATH


@dataclass(frozen=True)
class TrainingConfig:
    parquet_path: str
    feature_spec_path: str
    output_dir: str
    row_filter: str
    sample_rows: int
    epochs: int
    batch_size: int
    latent_dim: int
    hidden_dims: list[int]
    learning_rate: float
    kl_weight: float
    train_fraction: float
    seed: int
    threads: int | None
    data_seed: int | None = None


def seed_everything(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def build_model(prepared: PreparedTensors, hidden_dims: list[int], latent_dim: int) -> TabularVAE:
    categorical_cardinalities = {
        spec.name: spec.cardinality for spec in prepared.categorical_specs
    }
    embedding_dims = {
        name: infer_embedding_dim(cardinality) for name, cardinality in categorical_cardinalities.items()
    }
    config = VAEConfig(
        categorical_cardinalities=categorical_cardinalities,
        embedding_dims=embedding_dims,
        numeric_dim=len(prepared.numeric_specs),
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
    )
    return TabularVAE(config)


def compute_loss(
    model: TabularVAE,
    categorical_inputs: torch.Tensor,
    numeric_inputs: torch.Tensor,
    numeric_mask: torch.Tensor,
    kl_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    numeric_output, categorical_output, mu, logvar = model(categorical_inputs, numeric_inputs)

    observed_entries = numeric_mask.sum().clamp_min(1.0)
    numeric_loss = (((numeric_output - numeric_inputs) ** 2) * numeric_mask).sum() / observed_entries

    if categorical_output:
        categorical_loss = torch.stack(
            [
                F.cross_entropy(categorical_output[name], categorical_inputs[:, index])
                for index, name in enumerate(model.categorical_names)
            ]
        ).mean()
    else:
        categorical_loss = torch.zeros((), device=numeric_inputs.device)

    kl_loss = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()
    total_loss = numeric_loss + categorical_loss + kl_weight * kl_loss

    return total_loss, {
        "numeric_loss": float(numeric_loss.detach().cpu()),
        "categorical_loss": float(categorical_loss.detach().cpu()),
        "kl_loss": float(kl_loss.detach().cpu()),
        "total_loss": float(total_loss.detach().cpu()),
    }


def run_epoch(
    model: TabularVAE,
    loader: DataLoader,
    device: torch.device,
    kl_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(mode=training)
    running = {"numeric_loss": 0.0, "categorical_loss": 0.0, "kl_loss": 0.0, "total_loss": 0.0}
    num_batches = 0

    for categorical_inputs, numeric_inputs, numeric_mask in loader:
        categorical_inputs = categorical_inputs.to(device, non_blocking=True)
        numeric_inputs = numeric_inputs.to(device, non_blocking=True)
        numeric_mask = numeric_mask.to(device, non_blocking=True)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            loss, metrics = compute_loss(
                model=model,
                categorical_inputs=categorical_inputs,
                numeric_inputs=numeric_inputs,
                numeric_mask=numeric_mask,
                kl_weight=kl_weight,
            )
            if optimizer is not None:
                loss.backward()
                optimizer.step()

        for key, value in metrics.items():
            running[key] += value
        num_batches += 1

    if num_batches == 0:
        raise RuntimeError("No batches were produced during training")
    return {key: value / num_batches for key, value in running.items()}


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


@dataclass
class RunTimer:
    """End-to-end wall clock for one training run.

    The existing decode timings measure only the loader, and the loader runs
    *concurrently* with the GPU, so they cannot be summed into a run duration and
    cannot say which side was the bottleneck. This records the actual clock.

    Compare ``epoch_seconds`` against the per-epoch decode delta: if they track each
    other the loader is the constraint (more ``decode_workers``, ``pre_buffer``);
    if ``epoch_seconds`` is much larger the GPU is, and the loader is already
    keeping up.

    Durations come from ``perf_counter`` (monotonic — a clock adjustment mid-run
    cannot produce a negative epoch); the ISO timestamps are wall clock and are
    there so a run can be lined up against cluster logs.
    """

    started_utc: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )
    _t0: float = field(default_factory=time.perf_counter, repr=False)
    _setup_seconds: float | None = field(default=None, repr=False)
    _loop_seconds: float | None = field(default=None, repr=False)
    _epoch_t0: float | None = field(default=None, repr=False)
    _epochs: list[float] = field(default_factory=list, repr=False)

    def mark_setup_done(self) -> None:
        """Call once, immediately before the epoch loop."""
        self._setup_seconds = time.perf_counter() - self._t0

    def mark_loop_done(self) -> None:
        """Call once, immediately after the epoch loop (including on early stop)."""
        setup = self._setup_seconds or 0.0
        self._loop_seconds = (time.perf_counter() - self._t0) - setup

    def epoch_start(self) -> None:
        self._epoch_t0 = time.perf_counter()

    def epoch_seconds(self) -> float:
        """Seconds since the matching ``epoch_start``; 0.0 if it was never called.

        Recorded for the ``epoch_*_seconds`` aggregates, so call it exactly once per
        epoch — the value goes in that epoch's history entry.
        """
        if self._epoch_t0 is None:
            return 0.0
        elapsed = round(time.perf_counter() - self._epoch_t0, 3)
        self._epochs.append(elapsed)
        return elapsed

    def as_dict(self) -> dict[str, object]:
        """The wall-clock block. Call at artifact-write time, not before."""
        total = time.perf_counter() - self._t0
        setup = self._setup_seconds
        loop = self._loop_seconds
        return {
            "started_utc": self.started_utc,
            "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "total_seconds": round(total, 3),
            "total_hours": round(total / 3600.0, 4),
            "setup_seconds": None if setup is None else round(setup, 3),
            "train_loop_seconds": None if loop is None else round(loop, 3),
            # Everything after the loop: best-weight restore, artifact writes,
            # torch.save of the checkpoint.
            "finalize_seconds": (
                None
                if (setup is None or loop is None)
                else round(total - setup - loop, 3)
            ),
            # Carried here rather than left to be re-derived from `history`, because
            # train_result.json (what the tmux runner prints) keeps only final_epoch.
            "epochs_timed": len(self._epochs),
            "epoch_mean_seconds": (
                round(sum(self._epochs) / len(self._epochs), 3) if self._epochs else None
            ),
            "epoch_min_seconds": min(self._epochs) if self._epochs else None,
            "epoch_max_seconds": max(self._epochs) if self._epochs else None,
        }


def train(config: TrainingConfig) -> Path:
    timer = RunTimer()
    seed_everything(config.seed)
    output_root = Path(config.output_dir)
    run_dir = output_root / datetime.now(UTC).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=False)

    feature_specs = load_feature_specs(config.feature_spec_path)
    feature_names = [spec.name for spec in feature_specs]
    hidden_dims = [int(part) for part in config.hidden_dims]

    with connect_duckdb(config.threads) as count_conn:
        total_rows = get_row_count(
            count_conn,
            config.parquet_path,
            where=config.row_filter,
        )
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
    sampling_seed = config.data_seed if config.data_seed is not None else config.seed
    with connect_duckdb(threads=1) as sample_conn:
        sampled_frame = sample_frame(
            conn=sample_conn,
            parquet_path=config.parquet_path,
            feature_names=feature_names,
            sample_rows=sample_rows,
            seed=sampling_seed,
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

    # Cap GPU memory before anything allocates. Tensors stay in host RAM here
    # (TensorDataset) and only a batch at a time crosses to the device, so this
    # path is light -- but the cap still stops a large --batch-size from expanding
    # into a card that other work is sharing.
    from uv_vae import gpu_budget  # local import: keeps torch-free imports of this module cheap

    budget_report = gpu_budget.apply()

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

    history: list[dict[str, float | int]] = []
    timer.mark_setup_done()
    progress = tqdm(range(1, config.epochs + 1), desc="epochs", leave=False)
    for epoch in progress:
        timer.epoch_start()
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
        epoch_metrics = {
            "epoch": epoch,
            "train_total_loss": train_metrics["total_loss"],
            "train_numeric_loss": train_metrics["numeric_loss"],
            "train_categorical_loss": train_metrics["categorical_loss"],
            "train_kl_loss": train_metrics["kl_loss"],
            "val_total_loss": val_metrics["total_loss"],
            "val_numeric_loss": val_metrics["numeric_loss"],
            "val_categorical_loss": val_metrics["categorical_loss"],
            "val_kl_loss": val_metrics["kl_loss"],
            "epoch_seconds": timer.epoch_seconds(),
        }
        history.append(epoch_metrics)
        progress.set_postfix(
            train=f"{train_metrics['total_loss']:.4f}",
            val=f"{val_metrics['total_loss']:.4f}",
        )

    timer.mark_loop_done()

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
        "history": history,
        "gpu_budget": budget_report.as_dict(),
        "gpu_environment": gpu_budget.describe_environment(),
        "wall_clock": timer.as_dict(),
    }

    write_json(run_dir / "feature_report.json", feature_report)
    write_json(run_dir / "preprocess_report.json", preprocess_report)
    write_json(run_dir / "training_report.json", training_report)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model.config),
            "training_config": asdict(config),
            "feature_report": feature_report,
            "preprocess_report": preprocess_report,
            "history": history,
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
        "final_epoch": history[-1] if history else {},
        "wall_clock": timer.as_dict(),
    }
    write_json(run_dir / "summary.json", summary)
    return run_dir
