from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl
import torch

from uv_vae.data import split_specs
from uv_vae.features import FeatureSpec


@dataclass(frozen=True)
class PreparedTensors:
    train_cat: torch.Tensor
    train_num: torch.Tensor
    train_mask: torch.Tensor
    val_cat: torch.Tensor
    val_num: torch.Tensor
    val_mask: torch.Tensor
    categorical_specs: list[FeatureSpec]
    numeric_specs: list[FeatureSpec]
    dropped_all_null_features: list[str]
    dropped_sample_null_features: list[str]
    non_null_counts: dict[str, int]
    numeric_means: dict[str, float]
    numeric_stds: dict[str, float]
    sample_size: int
    total_rows: int


def infer_embedding_dim(cardinality: int) -> int:
    return max(2, min(16, math.ceil(cardinality**0.5) * 2))


def encode_categorical_column(frame: pl.DataFrame, spec: FeatureSpec) -> np.ndarray:
    if spec.values is None:
        raise ValueError(f"Categorical feature {spec.name} is missing a category map")
    null_index = spec.values.get(spec.null_token, 0)
    values = frame.get_column(spec.name).cast(pl.String).fill_null(spec.null_token).to_list()
    return np.fromiter(
        (spec.values.get(value, null_index) for value in values),
        dtype=np.int64,
        count=frame.height,
    )


def encode_numeric_column(frame: pl.DataFrame, spec: FeatureSpec) -> tuple[np.ndarray, np.ndarray]:
    values = (
        frame.get_column(spec.name)
        .cast(pl.Float32)
        .fill_null(float("nan"))
        .to_numpy()
        .astype(np.float32, copy=False)
    )
    mask = (~np.isnan(values)).astype(np.float32, copy=False)
    return values, mask


def prepare_tensors(
    frame: pl.DataFrame,
    specs: list[FeatureSpec],
    non_null_counts: dict[str, int],
    total_rows: int,
    train_fraction: float,
    seed: int,
) -> PreparedTensors:
    categorical_specs, numeric_specs, dropped_all_null_features = split_specs(specs, non_null_counts)

    sample_null_numeric: list[str] = []
    numeric_values: list[np.ndarray] = []
    numeric_masks: list[np.ndarray] = []
    retained_numeric_specs: list[FeatureSpec] = []
    numeric_means: dict[str, float] = {}
    numeric_stds: dict[str, float] = {}

    for spec in numeric_specs:
        values, mask = encode_numeric_column(frame, spec)
        if float(mask.sum()) == 0.0:
            sample_null_numeric.append(spec.name)
            continue
        retained_numeric_specs.append(spec)
        numeric_values.append(values)
        numeric_masks.append(mask)

    retained_categorical_specs: list[FeatureSpec] = []
    sample_null_categorical: list[str] = []
    categorical_values: list[np.ndarray] = []
    for spec in categorical_specs:
        encoded = encode_categorical_column(frame, spec)
        if encoded.size == 0 or np.unique(encoded).size == 1:
            only_value = encoded[0] if encoded.size else None
            if only_value == spec.values.get(spec.null_token, -1):
                sample_null_categorical.append(spec.name)
                continue
        retained_categorical_specs.append(spec)
        categorical_values.append(encoded)

    dropped_sample_null_features = sorted(sample_null_numeric + sample_null_categorical)

    if not retained_numeric_specs and not retained_categorical_specs:
        raise RuntimeError("No usable features remain after dropping all-null columns")

    row_count = frame.height
    categorical_matrix = (
        np.stack(categorical_values, axis=1) if categorical_values else np.zeros((row_count, 0), dtype=np.int64)
    )
    numeric_matrix = (
        np.stack(numeric_values, axis=1) if numeric_values else np.zeros((row_count, 0), dtype=np.float32)
    )
    numeric_mask_matrix = (
        np.stack(numeric_masks, axis=1) if numeric_masks else np.zeros((row_count, 0), dtype=np.float32)
    )

    rng = np.random.default_rng(seed)
    indices = rng.permutation(row_count)
    train_size = max(1, min(row_count - 1, int(row_count * train_fraction)))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    if val_indices.size == 0:
        val_indices = train_indices[:1]

    train_numeric = numeric_matrix[train_indices]
    train_numeric_mask = numeric_mask_matrix[train_indices]
    val_numeric = numeric_matrix[val_indices]
    val_numeric_mask = numeric_mask_matrix[val_indices]

    if retained_numeric_specs:
        for column_index, spec in enumerate(retained_numeric_specs):
            observed = train_numeric[:, column_index][train_numeric_mask[:, column_index] > 0]
            mean = float(observed.mean()) if observed.size else 0.0
            std = float(observed.std()) if observed.size else 1.0
            if not math.isfinite(std) or std < 1e-6:
                std = 1.0
            numeric_means[spec.name] = mean
            numeric_stds[spec.name] = std

        means = np.array([numeric_means[spec.name] for spec in retained_numeric_specs], dtype=np.float32)
        stds = np.array([numeric_stds[spec.name] for spec in retained_numeric_specs], dtype=np.float32)
        numeric_matrix = np.where(numeric_mask_matrix > 0, numeric_matrix, means)
        numeric_matrix = (numeric_matrix - means) / stds
        train_numeric = numeric_matrix[train_indices]
        val_numeric = numeric_matrix[val_indices]

    train_cat = torch.from_numpy(categorical_matrix[train_indices]).long()
    val_cat = torch.from_numpy(categorical_matrix[val_indices]).long()
    train_num = torch.from_numpy(train_numeric).float()
    val_num = torch.from_numpy(val_numeric).float()
    train_mask = torch.from_numpy(train_numeric_mask).float()
    val_mask = torch.from_numpy(val_numeric_mask).float()

    return PreparedTensors(
        train_cat=train_cat,
        train_num=train_num,
        train_mask=train_mask,
        val_cat=val_cat,
        val_num=val_num,
        val_mask=val_mask,
        categorical_specs=retained_categorical_specs,
        numeric_specs=retained_numeric_specs,
        dropped_all_null_features=sorted(dropped_all_null_features),
        dropped_sample_null_features=dropped_sample_null_features,
        non_null_counts=non_null_counts,
        numeric_means=numeric_means,
        numeric_stds=numeric_stds,
        sample_size=row_count,
        total_rows=total_rows,
    )


def transform_frame(
    frame: pl.DataFrame,
    categorical_specs: list[FeatureSpec],
    numeric_specs: list[FeatureSpec],
    numeric_means: dict[str, float],
    numeric_stds: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    row_count = frame.height
    categorical_values = [encode_categorical_column(frame, spec) for spec in categorical_specs]
    categorical_matrix = (
        np.stack(categorical_values, axis=1) if categorical_values else np.zeros((row_count, 0), dtype=np.int64)
    )

    numeric_values: list[np.ndarray] = []
    for spec in numeric_specs:
        values, mask = encode_numeric_column(frame, spec)
        mean = np.float32(numeric_means.get(spec.name, 0.0))
        std = np.float32(numeric_stds.get(spec.name, 1.0))
        if not math.isfinite(float(std)) or float(std) < 1e-6:
            std = np.float32(1.0)
        normalized = np.where(mask > 0, values, mean)
        normalized = ((normalized - mean) / std).astype(np.float32, copy=False)
        numeric_values.append(normalized)

    numeric_matrix = (
        np.stack(numeric_values, axis=1) if numeric_values else np.zeros((row_count, 0), dtype=np.float32)
    )

    return (
        torch.from_numpy(categorical_matrix).long(),
        torch.from_numpy(numeric_matrix).float(),
    )
