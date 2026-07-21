"""Per-epoch latent-space convergence tracking.

Encodes a fixed test sample through the VAE after each training epoch and
computes Procrustes distance, Linear CKA, and trustworthiness between
consecutive epochs.  The convergence trajectory shows whether the latent
representation is stabilising as training progresses.

This module is additive — it does not modify any original uv_vae files.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.spatial import procrustes
from sklearn.manifold import trustworthiness

from uv_vae.data import connect_duckdb, get_row_count, sample_frame
from uv_vae.features import FeatureSpec
from uv_vae.preprocess import encode_categorical_column, encode_numeric_column


def _linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA computed in feature space — O(n*d^2) instead of O(n^2*d).

    Much more efficient when n >> d (many rows, few latent dims).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)

    xy = x.T @ y   # d_x × d_y
    xx = x.T @ x   # d_x × d_x
    yy = y.T @ y   # d_y × d_y

    numerator = np.sum(xy * xy)                              # ||X^T Y||_F^2
    denominator = np.sqrt(np.sum(xx * xx) * np.sum(yy * yy))  # ||X^T X||_F * ||Y^T Y||_F

    return float(numerator / max(denominator, 1e-12))


def compare_epoch_latents(
    prev: np.ndarray,
    curr: np.ndarray,
    n_neighbors: int = 10,
) -> dict[str, float]:
    """Compare latent embeddings between two consecutive epochs.

    Returns Procrustes distance, Linear CKA, and trustworthiness — three
    complementary views of whether the latent space has changed.
    """
    prev = np.asarray(prev, dtype=np.float64)
    curr = np.asarray(curr, dtype=np.float64)

    n = min(prev.shape[0], curr.shape[0])
    prev = prev[:n]
    curr = curr[:n]

    _, _, disparity = procrustes(prev, curr)

    cka = _linear_cka(prev, curr)

    k = min(n_neighbors, max(1, n // 2 - 1))
    if k >= 1 and n > k:
        trust = float(trustworthiness(prev, curr, n_neighbors=k))
    else:
        trust = 1.0

    return {
        "procrustes_distance": float(disparity),
        "linear_cka": float(cka),
        "trustworthiness": float(trust),
    }


def load_test_sample(
    parquet_path: str,
    categorical_specs: list[FeatureSpec],
    numeric_specs: list[FeatureSpec],
    numeric_means: dict[str, float],
    numeric_stds: dict[str, float],
    row_filter: str | None = None,
    max_rows: int = 5000,
    seed: int = 42,
    threads: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a fixed sample from a test parquet and encode to tensors.

    Uses threads=1 for deterministic reservoir sampling.
    """
    feature_names = [s.name for s in categorical_specs] + [s.name for s in numeric_specs]

    with connect_duckdb(threads=1) as conn:
        total = get_row_count(conn, parquet_path, where=row_filter)
        sample_n = min(max_rows, total)
        df = sample_frame(
            conn=conn,
            parquet_path=parquet_path,
            feature_names=feature_names,
            sample_rows=sample_n,
            seed=seed,
            where=row_filter,
        )

    n = df.height
    print(
        f"  convergence test set: {n:,} rows from {parquet_path}",
        file=sys.stderr,
        flush=True,
    )

    if categorical_specs:
        cat_matrix = np.stack(
            [encode_categorical_column(df, s) for s in categorical_specs],
            axis=1,
        )
    else:
        cat_matrix = np.zeros((n, 0), dtype=np.int64)

    means_arr = np.array([numeric_means[s.name] for s in numeric_specs], dtype=np.float32)
    stds_arr = np.array([numeric_stds[s.name] for s in numeric_specs], dtype=np.float32)

    if numeric_specs:
        num_arrays: list[np.ndarray] = []
        mask_arrays: list[np.ndarray] = []
        for spec in numeric_specs:
            values, mask = encode_numeric_column(df, spec)
            num_arrays.append(values)
            mask_arrays.append(mask)
        num_matrix = np.stack(num_arrays, axis=1)
        mask_matrix = np.stack(mask_arrays, axis=1)
        num_matrix = np.where(mask_matrix > 0, num_matrix, means_arr)
        num_matrix = (num_matrix - means_arr) / stds_arr
    else:
        num_matrix = np.zeros((n, 0), dtype=np.float32)

    return (
        torch.from_numpy(cat_matrix).long(),
        torch.from_numpy(num_matrix).float(),
    )


@dataclass
class ConvergenceTracker:
    """Tracks epoch-to-epoch latent-space convergence on a fixed test set.

    After each epoch, encodes the test data through the current model and
    compares the latent embeddings to the previous epoch's output using
    Procrustes distance, Linear CKA, and trustworthiness.  Convergence is
    indicated by Procrustes → 0, CKA → 1, trustworthiness → 1.
    """

    test_cat: torch.Tensor
    test_num: torch.Tensor
    device: torch.device
    prev_latents: np.ndarray | None = field(default=None, init=False, repr=False)
    history: list[dict] = field(default_factory=list, init=False, repr=False)

    @classmethod
    def from_parquet(
        cls,
        parquet_path: str,
        categorical_specs: list[FeatureSpec],
        numeric_specs: list[FeatureSpec],
        numeric_means: dict[str, float],
        numeric_stds: dict[str, float],
        device: torch.device,
        row_filter: str | None = None,
        max_rows: int = 5000,
        seed: int = 42,
        threads: int | None = None,
    ) -> ConvergenceTracker:
        cat, num = load_test_sample(
            parquet_path=parquet_path,
            categorical_specs=categorical_specs,
            numeric_specs=numeric_specs,
            numeric_means=numeric_means,
            numeric_stds=numeric_stds,
            row_filter=row_filter,
            max_rows=max_rows,
            seed=seed,
            threads=threads,
        )
        return cls(
            test_cat=cat.to(device),
            test_num=num.to(device),
            device=device,
        )

    def evaluate_epoch(
        self,
        model: torch.nn.Module,
        epoch: int,
        use_amp: bool = False,
    ) -> dict[str, float]:
        """Encode test data and compare to the previous epoch's latents."""
        model.eval()
        with torch.inference_mode():
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                mu, _ = model.encode(self.test_cat, self.test_num)
        latents = mu.detach().cpu().float().numpy()

        metrics: dict[str, float] = {"epoch": float(epoch)}
        if self.prev_latents is not None:
            comparison = compare_epoch_latents(self.prev_latents, latents)
            metrics.update(comparison)

        self.prev_latents = latents.copy()
        self.history.append(metrics)
        return metrics
