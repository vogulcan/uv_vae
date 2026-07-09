from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
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


def train(config: TrainingConfig) -> Path:
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

    history: list[dict[str, float | int]] = []
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
        }
        history.append(epoch_metrics)
        progress.set_postfix(
            train=f"{train_metrics['total_loss']:.4f}",
            val=f"{val_metrics['total_loss']:.4f}",
        )

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
    }
    write_json(run_dir / "summary.json", summary)
    return run_dir
