"""Streaming parquet DataLoader for full-dataset VAE training.

Reads data from parquet via DuckDB in a streaming fashion so memory stays flat
regardless of dataset size.  Preprocessing (categorical encoding, numeric
normalisation) is applied per-chunk as data flows through.

Train/val split is deterministic: row *i* in the stream goes to validation when
``i % val_denominator == seed % val_denominator``, otherwise to training.  Parquet
row order is stable, so this is reproducible across runs with the same seed.

Shuffling is done within fixed-size windows (default 500 000 rows) — large enough
to break any correlation from the parquet's physical layout (e.g. samples
concatenated in order) without needing to hold the full dataset in memory.
"""

from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from uv_vae.data import (
    connect_duckdb,
    get_non_null_counts,
    get_row_count,
    quote_ident,
    split_specs,
    stream_parquet_batches,
)
from uv_vae.early_stopping import (
    EarlyStoppingConfig,
    EarlyStoppingMonitor,
)
from uv_vae.features import FeatureSpec, load_feature_specs
from uv_vae.model import TabularVAE, VAEConfig
from uv_vae.preprocess import encode_categorical_column, encode_numeric_column, infer_embedding_dim
from uv_vae.training import (
    TrainingConfig,
    seed_everything,
    write_json,
)

_CUDF_AVAILABLE = False
try:
    import cudf as _cudf
    import cupy as _cupy
    _CUDF_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Normalisation stats via SQL
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StreamingStats:
    """Normalisation statistics computed via DuckDB SQL (no materialisation)."""

    numeric_means: dict[str, float]
    numeric_stds: dict[str, float]
    categorical_specs: list[FeatureSpec]
    numeric_specs: list[FeatureSpec]
    dropped_all_null_features: list[str]
    non_null_counts: dict[str, int]
    total_rows: int


def compute_streaming_stats(
    parquet_path: str,
    feature_specs: list[FeatureSpec],
    non_null_counts: dict[str, int],
    total_rows: int,
    row_filter: str | None = None,
    threads: int | None = None,
) -> StreamingStats:
    """Compute normalisation statistics for numeric features via a single DuckDB query."""
    categorical_specs, numeric_specs, dropped_all_null = split_specs(
        feature_specs, non_null_counts
    )

    if not numeric_specs:
        return StreamingStats(
            numeric_means={},
            numeric_stds={},
            categorical_specs=categorical_specs,
            numeric_specs=numeric_specs,
            dropped_all_null_features=sorted(dropped_all_null),
            non_null_counts=non_null_counts,
            total_rows=total_rows,
        )

    agg_parts: list[str] = []
    for spec in numeric_specs:
        col = quote_ident(spec.name)
        agg_parts.append(f"AVG(CAST({col} AS DOUBLE))")
        agg_parts.append(f"STDDEV_SAMP(CAST({col} AS DOUBLE))")

    sql = f"SELECT {', '.join(agg_parts)} FROM read_parquet(?)"
    if row_filter:
        sql += f" WHERE {row_filter}"

    with connect_duckdb(threads=threads) as conn:
        row = conn.execute(sql, [str(parquet_path)]).fetchone()

    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for i, spec in enumerate(numeric_specs):
        raw_mean = row[i * 2]
        raw_std = row[i * 2 + 1]
        mean = float(raw_mean) if raw_mean is not None else 0.0
        std = float(raw_std) if raw_std is not None else 1.0
        if not math.isfinite(mean):
            mean = 0.0
        if not math.isfinite(std) or std < 1e-6:
            std = 1.0
        means[spec.name] = mean
        stds[spec.name] = std

    return StreamingStats(
        numeric_means=means,
        numeric_stds=stds,
        categorical_specs=categorical_specs,
        numeric_specs=numeric_specs,
        dropped_all_null_features=sorted(dropped_all_null),
        non_null_counts=non_null_counts,
        total_rows=total_rows,
    )


# ---------------------------------------------------------------------------
# Streaming dataset
# ---------------------------------------------------------------------------

class StreamingParquetDataset(IterableDataset):
    """Streams rows from parquet via DuckDB, encoding and normalising on the fly.

    Each call to ``__iter__`` opens a fresh DuckDB connection and streams through
    the entire parquet file, yielding pre-batched ``(cat, num, mask)`` tuples
    ready for ``run_epoch``.

    Use ``DataLoader(dataset, batch_size=None)`` so the loader passes through
    the pre-batched tuples without re-batching.
    """

    def __init__(
        self,
        parquet_path: str,
        categorical_specs: list[FeatureSpec],
        numeric_specs: list[FeatureSpec],
        numeric_means: dict[str, float],
        numeric_stds: dict[str, float],
        row_filter: str | None,
        batch_size: int,
        split: str,
        train_fraction: float = 0.9,
        seed: int = 42,
        threads: int | None = None,
        shuffle: bool = True,
        shuffle_buffer_rows: int = 500_000,
        read_chunk_rows: int = 100_000,
    ) -> None:
        self.parquet_path = parquet_path
        self.categorical_specs = categorical_specs
        self.numeric_specs = numeric_specs
        self.row_filter = row_filter
        self.batch_size = batch_size
        self.split = split
        self.seed = seed
        self.threads = threads
        self.shuffle = shuffle and split == "train"
        self.shuffle_buffer_rows = shuffle_buffer_rows
        self.read_chunk_rows = read_chunk_rows
        self._epoch = 0

        self.val_denominator = max(2, round(1.0 / max(1e-9, 1.0 - train_fraction)))
        self.val_remainder = seed % self.val_denominator

        self.means_arr = np.array(
            [numeric_means[s.name] for s in numeric_specs], dtype=np.float32
        )
        self.stds_arr = np.array(
            [numeric_stds[s.name] for s in numeric_specs], dtype=np.float32
        )

        self.feature_names = [s.name for s in categorical_specs] + [
            s.name for s in numeric_specs
        ]
        self._use_cudf = _CUDF_AVAILABLE and torch.cuda.is_available()

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def _encode_chunk(
        self, df: pl.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = df.height

        if self.categorical_specs:
            cat_matrix = np.stack(
                [encode_categorical_column(df, s) for s in self.categorical_specs],
                axis=1,
            )
        else:
            cat_matrix = np.zeros((n, 0), dtype=np.int64)

        if self.numeric_specs:
            num_arrays: list[np.ndarray] = []
            mask_arrays: list[np.ndarray] = []
            for spec in self.numeric_specs:
                values, mask = encode_numeric_column(df, spec)
                num_arrays.append(values)
                mask_arrays.append(mask)
            num_matrix = np.stack(num_arrays, axis=1)
            mask_matrix = np.stack(mask_arrays, axis=1)
            num_matrix = np.where(mask_matrix > 0, num_matrix, self.means_arr)
            num_matrix = (num_matrix - self.means_arr) / self.stds_arr
        else:
            num_matrix = np.zeros((n, 0), dtype=np.float32)
            mask_matrix = np.zeros((n, 0), dtype=np.float32)

        return cat_matrix, num_matrix, mask_matrix

    def _encode_chunk_gpu(
        self, df: "_cudf.DataFrame"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode a cuDF DataFrame using GPU-accelerated vectorised ops."""
        n = len(df)

        if self.categorical_specs:
            cat_arrays: list[np.ndarray] = []
            for spec in self.categorical_specs:
                null_index = spec.values.get(spec.null_token, 0)
                col = df[spec.name].fillna(spec.null_token).astype(str)
                codes = col.map(spec.values).fillna(null_index).astype("int64")
                cat_arrays.append(_cupy.asnumpy(codes.values))
            cat_matrix = np.stack(cat_arrays, axis=1)
        else:
            cat_matrix = np.zeros((n, 0), dtype=np.int64)

        if self.numeric_specs:
            num_arrays: list[np.ndarray] = []
            mask_arrays: list[np.ndarray] = []
            for i, spec in enumerate(self.numeric_specs):
                col = df[spec.name].astype("float32")
                mask_col = (~col.isna()).astype("float32")
                filled = col.fillna(float(self.means_arr[i]))
                num_arrays.append(_cupy.asnumpy(filled.values))
                mask_arrays.append(_cupy.asnumpy(mask_col.values))
            num_matrix = np.stack(num_arrays, axis=1).astype(np.float32)
            mask_matrix = np.stack(mask_arrays, axis=1).astype(np.float32)
            num_matrix = (num_matrix - self.means_arr) / self.stds_arr
        else:
            num_matrix = np.zeros((n, 0), dtype=np.float32)
            mask_matrix = np.zeros((n, 0), dtype=np.float32)

        return cat_matrix, num_matrix, mask_matrix

    def _split_mask(self, n_rows: int, global_offset: int) -> np.ndarray:
        indices = np.arange(global_offset, global_offset + n_rows)
        is_val = (indices % self.val_denominator) == self.val_remainder
        return ~is_val if self.split == "train" else is_val

    def _yield_batches(
        self, cat: np.ndarray, num: np.ndarray, mask: np.ndarray
    ):
        n = cat.shape[0]
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            yield (
                torch.from_numpy(cat[start:end].copy()).long(),
                torch.from_numpy(num[start:end].copy()).float(),
                torch.from_numpy(mask[start:end].copy()).float(),
            )

    def _flush_buffer(
        self,
        buf_cat: list[np.ndarray],
        buf_num: list[np.ndarray],
        buf_mask: list[np.ndarray],
        rng: np.random.Generator,
    ):
        cat = np.concatenate(buf_cat)
        num = np.concatenate(buf_num)
        mask = np.concatenate(buf_mask)
        perm = rng.permutation(cat.shape[0])
        cat = cat[perm]
        num = num[perm]
        mask = mask[perm]
        yield from self._yield_batches(cat, num, mask)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        conn = connect_duckdb(threads=self.threads)
        try:
            reader = stream_parquet_batches(
                conn=conn,
                parquet_path=self.parquet_path,
                select_columns=self.feature_names,
                rows_per_batch=self.read_chunk_rows,
                where=self.row_filter,
            )

            buf_cat: list[np.ndarray] = []
            buf_num: list[np.ndarray] = []
            buf_mask: list[np.ndarray] = []
            buf_rows = 0
            global_offset = 0

            for record_batch in reader:
                n_rows = record_batch.num_rows
                keep = self._split_mask(n_rows, global_offset)
                global_offset += n_rows

                if not keep.any():
                    continue

                if self._use_cudf:
                    df = _cudf.DataFrame.from_arrow(record_batch)
                    split_df = df[_cudf.Series(keep)]
                    cat, num, mask = self._encode_chunk_gpu(split_df)
                else:
                    df = pl.from_arrow(record_batch)
                    split_df = df.filter(pl.Series(keep))
                    cat, num, mask = self._encode_chunk(split_df)

                if self.shuffle:
                    buf_cat.append(cat)
                    buf_num.append(num)
                    buf_mask.append(mask)
                    buf_rows += cat.shape[0]
                    if buf_rows >= self.shuffle_buffer_rows:
                        yield from self._flush_buffer(buf_cat, buf_num, buf_mask, rng)
                        buf_cat, buf_num, buf_mask = [], [], []
                        buf_rows = 0
                else:
                    yield from self._yield_batches(cat, num, mask)

            if self.shuffle and buf_rows > 0:
                yield from self._flush_buffer(buf_cat, buf_num, buf_mask, rng)
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Model with dropout
# ---------------------------------------------------------------------------

class TabularVAEWithDropout(TabularVAE):
    """TabularVAE with input dropout and hidden-layer dropout.

    Overrides ``encode`` and ``decode`` to inject dropout without changing the
    underlying ``nn.Sequential`` layout, so the ``state_dict`` keys are identical
    to ``TabularVAE`` and checkpoints are interchangeable.
    """

    def __init__(
        self,
        config: VAEConfig,
        input_dropout: float = 0.1,
        hidden_dropout: float = 0.4,
    ) -> None:
        super().__init__(config)
        self.input_drop = nn.Dropout(input_dropout)
        self.hidden_drop = nn.Dropout(hidden_dropout)

    def encode(
        self, categorical_inputs: torch.Tensor, numeric_inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pieces = []
        for index, name in enumerate(self.categorical_names):
            pieces.append(self.embeddings[name](categorical_inputs[:, index]))
        pieces.append(numeric_inputs)
        x = self.input_drop(torch.cat(pieces, dim=1))
        for module in self.encoder:
            x = module(x)
            if isinstance(module, nn.ReLU):
                x = self.hidden_drop(x)
        return self.mu(x), self.logvar(x)

    def decode(
        self, latent: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = latent
        for module in self.decoder:
            x = module(x)
            if isinstance(module, nn.ReLU):
                x = self.hidden_drop(x)
        numeric_output = self.numeric_head(x)
        categorical_output = {
            name: head(x) for name, head in self.categorical_heads.items()
        }
        return numeric_output, categorical_output


# ---------------------------------------------------------------------------
# AMP-aware training epoch
# ---------------------------------------------------------------------------

def _run_training_epoch(
    model: TabularVAE,
    loader: DataLoader,
    device: torch.device,
    kl_weight: float,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> dict[str, float]:
    """One training epoch with optional mixed-precision (AMP).

    When ``scaler`` is disabled (CPU) the call path is equivalent to
    ``training.run_epoch`` with an optimizer.
    """
    model.train()
    use_amp = scaler.is_enabled()
    running = {
        "numeric_loss": 0.0,
        "categorical_loss": 0.0,
        "kl_loss": 0.0,
        "total_loss": 0.0,
    }
    num_batches = 0

    for cat_inputs, num_inputs, num_mask in loader:
        cat_inputs = cat_inputs.to(device, non_blocking=True)
        num_inputs = num_inputs.to(device, non_blocking=True)
        num_mask = num_mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            numeric_out, categorical_out, mu, logvar = model(
                cat_inputs, num_inputs
            )
            observed = num_mask.sum().clamp_min(1.0)
            numeric_loss = (
                ((numeric_out - num_inputs) ** 2) * num_mask
            ).sum() / observed
            if categorical_out:
                categorical_loss = torch.stack(
                    [
                        F.cross_entropy(categorical_out[name], cat_inputs[:, i])
                        for i, name in enumerate(model.categorical_names)
                    ]
                ).mean()
            else:
                categorical_loss = torch.zeros((), device=device)
            kl_loss = (
                -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
            ).mean()
            total_loss = numeric_loss + categorical_loss + kl_weight * kl_loss

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running["numeric_loss"] += float(numeric_loss.detach())
        running["categorical_loss"] += float(categorical_loss.detach())
        running["kl_loss"] += float(kl_loss.detach())
        running["total_loss"] += float(total_loss.detach())
        num_batches += 1

    if num_batches == 0:
        raise RuntimeError("No batches were produced during training")
    return {k: v / num_batches for k, v in running.items()}


# ---------------------------------------------------------------------------
# Combined validation + diagnostics (with optional AMP)
# ---------------------------------------------------------------------------

def run_val_epoch_with_diagnostics(
    model: TabularVAE,
    loader: DataLoader,
    device: torch.device,
    kl_weight: float,
    active_unit_threshold: float,
    use_amp: bool = False,
) -> tuple[dict[str, float], dict[str, object]]:
    """Validation loss AND latent diagnostics in a single pass through ``loader``."""
    model.eval()
    latent_dim = model.config.latent_dim

    running = {
        "numeric_loss": 0.0,
        "categorical_loss": 0.0,
        "kl_loss": 0.0,
        "total_loss": 0.0,
    }
    num_batches = 0

    diag_count = 0
    mu_sum = torch.zeros(latent_dim, dtype=torch.float64, device=device)
    mu_sq_sum = torch.zeros(latent_dim, dtype=torch.float64, device=device)
    kl_dim_sum = torch.zeros(latent_dim, dtype=torch.float64, device=device)

    with torch.inference_mode():
        for cat_inputs, num_inputs, num_mask in loader:
            cat_inputs = cat_inputs.to(device, non_blocking=True)
            num_inputs = num_inputs.to(device, non_blocking=True)
            num_mask = num_mask.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                numeric_out, categorical_out, mu, logvar = model(
                    cat_inputs, num_inputs
                )
                observed = num_mask.sum().clamp_min(1.0)
                numeric_loss = (
                    ((numeric_out - num_inputs) ** 2) * num_mask
                ).sum() / observed
                if categorical_out:
                    categorical_loss = torch.stack(
                        [
                            F.cross_entropy(
                                categorical_out[name], cat_inputs[:, i]
                            )
                            for i, name in enumerate(model.categorical_names)
                        ]
                    ).mean()
                else:
                    categorical_loss = torch.zeros((), device=num_inputs.device)
                kl_loss = (
                    -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
                ).mean()
                total_loss = numeric_loss + categorical_loss + kl_weight * kl_loss

            running["numeric_loss"] += float(numeric_loss)
            running["categorical_loss"] += float(categorical_loss)
            running["kl_loss"] += float(kl_loss)
            running["total_loss"] += float(total_loss)
            num_batches += 1

            mu64 = mu.to(dtype=torch.float64)
            logvar64 = logvar.to(dtype=torch.float64)
            mu_sum += mu64.sum(dim=0)
            mu_sq_sum += (mu64 * mu64).sum(dim=0)
            kl_dim_sum += (
                -0.5 * (1.0 + logvar64 - mu64.pow(2) - logvar64.exp())
            ).sum(dim=0)
            diag_count += int(mu.shape[0])

    if num_batches == 0:
        raise RuntimeError("No batches were produced during validation")

    val_metrics = {k: v / num_batches for k, v in running.items()}

    mu_mean = mu_sum / diag_count
    mu_var = torch.clamp((mu_sq_sum / diag_count) - mu_mean.pow(2), min=0.0)
    kl_per_dim = kl_dim_sum / diag_count
    active_mask = mu_var > active_unit_threshold

    diagnostics: dict[str, object] = {
        "active_units": int(active_mask.sum().item()),
        "latent_dim": latent_dim,
        "active_unit_threshold": float(active_unit_threshold),
        "active_unit_mask": [bool(v) for v in active_mask.cpu().tolist()],
        "posterior_mean_variance": [float(v) for v in mu_var.cpu().tolist()],
        "kl_per_dim": [float(v) for v in kl_per_dim.cpu().tolist()],
        "kl_total": float(kl_per_dim.sum().item()),
        "rows_evaluated": diag_count,
    }

    return val_metrics, diagnostics


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def _build_model(
    categorical_specs: list[FeatureSpec],
    numeric_specs: list[FeatureSpec],
    hidden_dims: list[int],
    latent_dim: int,
    input_dropout: float = 0.0,
    hidden_dropout: float = 0.0,
) -> TabularVAE:
    cardinalities = {s.name: s.cardinality for s in categorical_specs}
    embedding_dims = {
        name: infer_embedding_dim(card) for name, card in cardinalities.items()
    }
    vae_config = VAEConfig(
        categorical_cardinalities=cardinalities,
        embedding_dims=embedding_dims,
        numeric_dim=len(numeric_specs),
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
    )
    if input_dropout > 0 or hidden_dropout > 0:
        return TabularVAEWithDropout(vae_config, input_dropout, hidden_dropout)
    return TabularVAE(vae_config)


# ---------------------------------------------------------------------------
# Streaming training loop
# ---------------------------------------------------------------------------

def train_with_early_stopping_streaming(
    config: TrainingConfig,
    early_stopping: EarlyStoppingConfig | None = None,
    input_dropout: float = 0.0,
    hidden_dropout: float = 0.0,
    test_parquet_path: str | None = None,
    convergence_rows: int = 5000,
) -> Path:
    """Full-dataset VAE training with streaming I/O and early stopping.

    Writes the same run artifacts and checkpoint format so that
    ``LatentInference.from_checkpoint`` and the clustering pipeline consume the
    output unchanged.
    """
    early_stopping = early_stopping or EarlyStoppingConfig()

    seed_everything(config.seed)
    output_root = Path(config.output_dir)
    run_dir = output_root / datetime.now(UTC).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=False)

    feature_specs = load_feature_specs(config.feature_spec_path)
    feature_names = [spec.name for spec in feature_specs]
    hidden_dims = [int(d) for d in config.hidden_dims]

    # ---- lightweight SQL: row count + non-null counts ----
    with connect_duckdb(config.threads) as conn:
        total_rows = get_row_count(conn, config.parquet_path, where=config.row_filter)
        if total_rows == 0:
            raise RuntimeError(
                f"No rows matched the training filter: {config.row_filter}"
            )
        non_null_counts = get_non_null_counts(
            conn, config.parquet_path, feature_names, where=config.row_filter
        )

    # ---- normalisation stats via SQL (no materialisation) ----
    print(
        f"Computing normalisation stats over {total_rows:,} filtered rows ...",
        file=sys.stderr,
        flush=True,
    )
    stats = compute_streaming_stats(
        parquet_path=config.parquet_path,
        feature_specs=feature_specs,
        non_null_counts=non_null_counts,
        total_rows=total_rows,
        row_filter=config.row_filter,
        threads=config.threads,
    )

    # ---- device + AMP ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if device.type == "cuda":
        torch.set_float32_matmul_precision("highest")

    # ---- model (with dropout when requested) ----
    model = _build_model(
        stats.categorical_specs,
        stats.numeric_specs,
        hidden_dims,
        config.latent_dim,
        input_dropout=input_dropout,
        hidden_dropout=hidden_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    # ---- streaming data loaders ----
    common_ds_kwargs = dict(
        parquet_path=config.parquet_path,
        categorical_specs=stats.categorical_specs,
        numeric_specs=stats.numeric_specs,
        numeric_means=stats.numeric_means,
        numeric_stds=stats.numeric_stds,
        row_filter=config.row_filter,
        batch_size=config.batch_size,
        train_fraction=config.train_fraction,
        seed=config.seed,
        threads=config.threads,
    )
    train_dataset = StreamingParquetDataset(split="train", shuffle=True, **common_ds_kwargs)
    val_dataset = StreamingParquetDataset(split="val", shuffle=False, **common_ds_kwargs)
    train_loader = DataLoader(train_dataset, batch_size=None)
    val_loader = DataLoader(val_dataset, batch_size=None)

    # ---- convergence tracking (optional, requires test parquet) ----
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

    # ---- training loop ----
    monitor = EarlyStoppingMonitor(early_stopping)
    history: list[dict[str, float | int]] = []
    diagnostics_history: list[dict[str, object]] = []
    best_state: dict[str, torch.Tensor] | None = None

    est_train_rows = int(total_rows * config.train_fraction)
    est_batches = est_train_rows // config.batch_size
    dropout_desc = ""
    if input_dropout > 0 or hidden_dropout > 0:
        dropout_desc = f", dropout=({input_dropout}/{hidden_dropout})"
    print(
        f"Streaming training: ~{est_train_rows:,} train rows, "
        f"~{total_rows - est_train_rows:,} val rows, "
        f"~{est_batches:,} batches/epoch, device={device}, "
        f"amp={use_amp}, cudf={train_dataset._use_cudf}{dropout_desc}",
        file=sys.stderr,
        flush=True,
    )

    progress = tqdm(range(1, config.epochs + 1), desc="epochs", leave=False)
    for epoch in progress:
        train_dataset.set_epoch(epoch)

        train_metrics = _run_training_epoch(
            model=model,
            loader=train_loader,
            device=device,
            kl_weight=config.kl_weight,
            optimizer=optimizer,
            scaler=scaler,
        )
        val_metrics, diagnostics = run_val_epoch_with_diagnostics(
            model=model,
            loader=val_loader,
            device=device,
            kl_weight=config.kl_weight,
            active_unit_threshold=early_stopping.active_unit_threshold,
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

    # ---- artifacts (same format as early_stopping.train_with_early_stopping) ----
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
        "eligible_rows_in_parquet": total_rows,
        "sample_rows": total_rows,
        "dropped_all_null_features": stats.dropped_all_null_features,
        "dropped_sample_null_features": [],
        "active_categorical_features": [s.name for s in stats.categorical_specs],
        "active_numeric_features": [s.name for s in stats.numeric_specs],
        "non_null_counts": stats.non_null_counts,
        "streaming": True,
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
        "early_stopping": early_stopping_report,
        "history": history,
        "streaming": True,
        "input_dropout": input_dropout,
        "hidden_dropout": hidden_dropout,
        "amp": use_amp,
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

    if convergence_tracker is not None and convergence_tracker.history:
        convergence_report = {
            "test_parquet_path": test_parquet_path,
            "convergence_rows": convergence_rows,
            "per_epoch": convergence_tracker.history,
        }
        write_json(run_dir / "convergence_report.json", convergence_report)

    checkpoint_payload = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model.config),
        "training_config": asdict(config),
        "feature_report": feature_report,
        "preprocess_report": preprocess_report,
        "history": history,
        "early_stopping": early_stopping_report,
        "diagnostics_history": diagnostics_history,
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
        "convergence_tracking": convergence_tracker is not None,
    }
    write_json(run_dir / "summary.json", summary)
    return run_dir
