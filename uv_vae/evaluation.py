from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.manifold import trustworthiness
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import procrustes

from uv_vae.data import connect_duckdb, get_non_null_counts, get_row_count, sample_frame
from uv_vae.features import load_feature_specs
from uv_vae.model import TabularVAE, VAEConfig
from uv_vae.preprocess import PreparedTensors, transform_frame
from uv_vae.training import build_model, compute_loss, run_epoch, seed_everything


MAX_SUBSAMPLE_ROWS = 10_000_000


def _coerce_frame(data: Any) -> Any:
    if data is None:
        raise TypeError("data must be a dataframe-like object")
    if hasattr(data, "to_dict") and hasattr(data, "columns"):
        return data
    if hasattr(data, "__dataframe__"):
        return data
    if isinstance(data, np.ndarray):
        raise TypeError("numpy arrays are not supported by the built-in uv_vae training adapter; provide a dataframe-like object with column names")
    raise TypeError("data must be dataframe-like and expose column names")


def _coerce_array(values: Sequence[float] | np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return array.reshape(1)
    return array


def _sample_rows(data: Any, n_rows: int, random_seed: int) -> Any:
    if hasattr(data, "__getitem__") and not isinstance(data, (str, bytes)):
        size = len(data)
    else:
        raise TypeError("data must support len() or be a sequence-like object")

    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    if n_rows > size:
        n_rows = size

    if hasattr(data, "sample") and hasattr(data, "columns"):
        try:
            return data.sample(n=n_rows, random_state=random_seed)
        except TypeError:
            return data.sample(n=n_rows, with_replacement=False, seed=random_seed)

    if hasattr(data, "sample") and hasattr(data, "height"):
        try:
            return data.sample(n=n_rows, with_replacement=False, seed=random_seed)
        except TypeError:
            return data.sample(n=n_rows, random_state=random_seed)

    rng = random.Random(random_seed)
    indices = list(range(size))
    rng.shuffle(indices)
    sampled_indices = indices[:n_rows]

    if isinstance(data, np.ndarray):
        return data[sampled_indices]
    return np.asarray(data[sampled_indices])


def _estimate_peak_memory_mb() -> float:
    try:
        import tracemalloc

        tracemalloc.start()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak / (1024 * 1024)
    except Exception:
        return float("nan")


def _train_with_uv_vae_pipeline(
    data: Any,
    *,
    feature_specs: Sequence[Any] | None = None,
    hidden_dims: Sequence[int] | None = None,
    latent_dim: int = 16,
    epochs: int = 10,
    batch_size: int = 4096,
    learning_rate: float = 1e-3,
    kl_weight: float = 0.05,
    train_fraction: float = 0.9,
    seed: int = 67,
    parquet_path: str | Path | None = None,
    row_filter: str | None = None,
    feature_spec_path: str | Path | None = None,
    threads: int | None = None,
    random_seed: int | None = None,
    fraction: float | None = None,
) -> tuple[Any, Callable[[Any], np.ndarray], dict[str, np.ndarray]]:
    """Train a TabularVAE using the existing uv_vae preprocessing and training utilities."""
    if feature_specs is None:
        if feature_spec_path is None:
            raise ValueError("feature_specs or feature_spec_path must be provided when using the built-in uv_vae training adapter")
        feature_specs = load_feature_specs(feature_spec_path)

    frame = _coerce_frame(data)
    feature_specs = list(feature_specs)
    feature_names = [spec.name for spec in feature_specs]
    if hasattr(frame, "columns"):
        missing_columns = [name for name in feature_names if name not in frame.columns]
        if missing_columns:
            raise ValueError(f"Frame is missing required feature columns: {missing_columns}")

    non_null_counts = {}
    if hasattr(frame, "get_column"):
        for name in feature_names:
            column = frame.get_column(name)
            if hasattr(column, "null_count") and hasattr(column, "len"):
                non_null_counts[name] = int(column.len() - column.null_count())
            else:
                non_null_counts[name] = 0
    else:
        for name in feature_names:
            non_null_counts[name] = 0

    effective_seed = int(random_seed if random_seed is not None else seed)

    prepared = None
    try:
        from uv_vae.preprocess import prepare_tensors
    except Exception as exc:  # pragma: no cover - defensive branch
        raise RuntimeError("uv_vae preprocessing utilities could not be imported") from exc

    prepared = prepare_tensors(
        frame=frame,
        specs=feature_specs,
        non_null_counts=non_null_counts,
        total_rows=frame.height if hasattr(frame, "height") else len(frame),
        train_fraction=train_fraction,
        seed=effective_seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(effective_seed)
    model = build_model(prepared, hidden_dims=[int(dim) for dim in (hidden_dims or [256, 128])], latent_dim=latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(prepared.train_cat, prepared.train_num, prepared.train_mask),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(effective_seed),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(prepared.val_cat, prepared.val_num, prepared.val_mask),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    for _ in range(epochs):
        run_epoch(model=model, loader=train_loader, device=device, kl_weight=kl_weight, optimizer=optimizer)
        run_epoch(model=model, loader=val_loader, device=device, kl_weight=kl_weight)

    model.eval()

    def encode_fn(batch: Any) -> np.ndarray:
        batch_frame = _coerce_frame(batch)
        categorical_inputs, numeric_inputs = transform_frame(
            frame=batch_frame,
            categorical_specs=prepared.categorical_specs,
            numeric_specs=prepared.numeric_specs,
            numeric_means=prepared.numeric_means,
            numeric_stds=prepared.numeric_stds,
        )
        with torch.inference_mode():
            mu, _ = model.encode(categorical_inputs.to(device), numeric_inputs.to(device))
            return mu.detach().cpu().numpy().astype(np.float32, copy=False)

    with torch.inference_mode():
        categorical_inputs, numeric_inputs = transform_frame(
            frame=frame,
            categorical_specs=prepared.categorical_specs,
            numeric_specs=prepared.numeric_specs,
            numeric_means=prepared.numeric_means,
            numeric_stds=prepared.numeric_stds,
        )
        mu, logvar = model.encode(categorical_inputs.to(device), numeric_inputs.to(device))
        kl_per_dim = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=0)).cpu().numpy()
        variance_per_dim = mu.var(dim=0).cpu().numpy()

    diagnostics = {
        "latent_dim_kl": np.asarray(kl_per_dim, dtype=float),
        "latent_dim_variance": np.asarray(variance_per_dim, dtype=float),
    }
    return model, encode_fn, diagnostics


def run_parquet_subsample_experiment(
    parquet_path: str | Path,
    feature_spec_path: str | Path,
    n_fractions: Sequence[float],
    row_filter: str | None = None,
    random_seed: int = 67,
    reference_rows: int | None = None,
    hidden_dims: Sequence[int] | None = None,
    latent_dim: int = 16,
    epochs: int = 10,
    batch_size: int = 4096,
    learning_rate: float = 1e-3,
    kl_weight: float = 0.05,
    train_fraction: float = 0.9,
    threads: int | None = None,
    output_dir: str | Path | None = None,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Run the subsampling experiment directly against a parquet file using the project pipeline."""
    feature_specs = load_feature_specs(feature_spec_path)
    feature_names = [spec.name for spec in feature_specs]

    with connect_duckdb(threads=threads) as count_conn:
        eligible_rows = get_row_count(count_conn, parquet_path, where=row_filter)
        non_null_counts = get_non_null_counts(count_conn, parquet_path, feature_names, where=row_filter)

    max_rows = min(eligible_rows, MAX_SUBSAMPLE_ROWS)
    if reference_rows is None:
        reference_rows = min(10_000, max_rows)

    results: list[dict[str, Any]] = []
    for fraction in n_fractions:
        sample_rows = max(1, int(math.ceil(max_rows * fraction)))
        with connect_duckdb(threads=1) as sample_conn:
            sampled_frame = sample_frame(
                conn=sample_conn,
                parquet_path=parquet_path,
                feature_names=feature_names,
                sample_rows=min(sample_rows, eligible_rows),
                seed=random_seed + int(round(fraction * 1_000_000)),
                where=row_filter,
            )

        training_kwargs = {
            "feature_specs": feature_specs,
            "feature_spec_path": feature_spec_path,
            "hidden_dims": hidden_dims,
            "latent_dim": latent_dim,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "kl_weight": kl_weight,
            "train_fraction": train_fraction,
            "seed": random_seed + int(round(fraction * 1_000_000)),
            "parquet_path": parquet_path,
            "row_filter": row_filter,
            "feature_spec_path": feature_spec_path,
            "threads": threads,
        }
        start_time = time.perf_counter()
        model, encode_fn, diagnostics = _train_with_uv_vae_pipeline(sampled_frame, **training_kwargs)
        elapsed = time.perf_counter() - start_time
        peak_memory_mb = _estimate_peak_memory_mb()
        embeddings = encode_fn(sampled_frame)
        results.append(
            {
                "fraction": float(fraction),
                "n_rows": int(sample_rows),
                "train_time_seconds": float(elapsed),
                "peak_memory_mb": float(peak_memory_mb),
                "model": model,
                "latent_embeddings": embeddings,
                "diagnostics": diagnostics,
                "reference_data": sampled_frame,
            }
        )

    summary_df = aggregate_results(results)
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_path / "subsample_evaluation_summary.csv", index=False)
    return results, summary_df


def run_subsample_experiment(
    data: Any,
    n_fractions: Sequence[float],
    vae_train_fn: Callable[..., tuple[Any, Callable[[Any], np.ndarray], dict[str, np.ndarray]]] | None = None,
    random_seed: int = 67,
    reference_rows: int | None = None,
    reference_data: Any | None = None,
    cluster_labels_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    feature_specs: Sequence[Any] | None = None,
    feature_spec_path: str | Path | None = None,
    hidden_dims: Sequence[int] | None = None,
    latent_dim: int = 16,
    epochs: int = 10,
    batch_size: int = 4096,
    learning_rate: float = 1e-3,
    kl_weight: float = 0.05,
    train_fraction: float = 0.9,
) -> list[dict[str, Any]]:
    """Run a single-draw VAE subsampling experiment over a range of fractions.

    Parameters
    ----------
    data:
        Sequence-like array or table-like object with a len() method. For array-like data,
        a row subsample is drawn directly.
    n_fractions:
        Fractions of the capped dataset size to use for training. Values should be in $(0, 1]$.
    vae_train_fn:
        Callable invoked as ``vae_train_fn(subsample, random_seed=random_seed, **kwargs)`` and
        expected to return ``(model, encode_fn, diagnostics)``.
    random_seed:
        Seed used for deterministic row sampling.
    reference_rows:
        Optional number of rows to reserve for the held-out reference set. Defaults to the
        smaller of 10,000 and the full dataset size.
    reference_data:
        Optional precomputed reference array. If omitted, a random subset is drawn from ``data``.
    cluster_labels_fn:
        Optional callable used to produce clustering labels from latent embeddings for ARI/NMI.

    Returns
    -------
    list[dict[str, Any]]
        One result per fraction with encoded embeddings, diagnostics, and metadata.
    """
    if not n_fractions:
        return []

    if hasattr(data, "__len__"):
        total_rows = len(data)
    else:
        raise TypeError("data must implement len()")

    cap_rows = min(total_rows, MAX_SUBSAMPLE_ROWS)
    if reference_rows is None:
        reference_rows = min(10_000, cap_rows)

    if reference_data is None:
        reference_data = _sample_rows(data, reference_rows, random_seed + 1234)

    results: list[dict[str, Any]] = []
    for fraction in n_fractions:
        if not 0 < fraction <= 1.0:
            raise ValueError("Each fraction must be in the interval (0, 1]")

        train_rows = max(1, int(math.ceil(cap_rows * fraction)))
        train_rows = min(train_rows, cap_rows)
        sample = _sample_rows(data, train_rows, random_seed + int(round(fraction * 1_000_000)))

        training_kwargs = {
            "random_seed": random_seed,
            "fraction": fraction,
            "feature_specs": feature_specs,
            "feature_spec_path": feature_spec_path,
            "hidden_dims": hidden_dims,
            "latent_dim": latent_dim,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "kl_weight": kl_weight,
            "train_fraction": train_fraction,
        }
        if vae_train_fn is None:
            start_time = time.perf_counter()
            model, encode_fn, diagnostics = _train_with_uv_vae_pipeline(sample, **training_kwargs)
            elapsed = time.perf_counter() - start_time
            peak_memory_mb = _estimate_peak_memory_mb()
        else:
            start_time = time.perf_counter()
            model, encode_fn, diagnostics = vae_train_fn(sample, **training_kwargs)
            elapsed = time.perf_counter() - start_time
            peak_memory_mb = _estimate_peak_memory_mb()

        reference_embeddings = encode_fn(reference_data)
        if reference_embeddings.ndim == 1:
            reference_embeddings = reference_embeddings.reshape(1, -1)

        result = {
            "fraction": float(fraction),
            "n_rows": int(train_rows),
            "train_time_seconds": float(elapsed),
            "peak_memory_mb": float(peak_memory_mb),
            "model": model,
            "latent_embeddings": reference_embeddings,
            "diagnostics": diagnostics,
            "reference_data": reference_data,
        }
        if cluster_labels_fn is not None:
            result["cluster_labels"] = cluster_labels_fn(reference_embeddings)
        results.append(result)

    return results


def diagnose_latent_collapse(
    vae_model: Any | None,
    latent_dim_kl: Sequence[float] | np.ndarray | None,
    latent_dim_variance: Sequence[float] | np.ndarray | None,
    threshold: float = 0.01,
    reconstruction_errors: Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Identify collapsed or low-variance latent dimensions.

    A dimension is marked collapsed when either its KL term or variance falls below the
    supplied threshold. A simple diagnostic summary is returned along with the worst
    reconstructed input columns when reconstruction errors are supplied.
    """
    kl = _coerce_array(latent_dim_kl)
    variance = _coerce_array(latent_dim_variance)

    if kl is None or variance is None:
        raise ValueError("Both latent_dim_kl and latent_dim_variance must be provided")

    if kl.shape != variance.shape:
        raise ValueError("latent_dim_kl and latent_dim_variance must have the same shape")

    collapsed = (kl < threshold) | (variance < threshold)
    active_mask = ~collapsed
    active_latent_dims_pct = float(active_mask.mean()) if active_mask.size else 0.0

    diagnostics: dict[str, Any] = {
        "collapsed_dimensions": int(collapsed.sum()),
        "active_latent_dims_pct": active_latent_dims_pct,
        "active_dimensions": np.flatnonzero(active_mask).tolist(),
        "collapsed_dimensions_indices": np.flatnonzero(collapsed).tolist(),
    }

    if reconstruction_errors is not None:
        errors = np.asarray(reconstruction_errors, dtype=float)
        diagnostics["worst_reconstruction_columns"] = np.argsort(errors)[::-1].tolist()
        diagnostics["reconstruction_error_by_column"] = errors.tolist()

    return diagnostics


def _linear_cka_similarity(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of rows")

    gram_x = x @ x.T
    gram_y = y @ y.T
    numerator = np.linalg.norm(gram_x - gram_y, ord="fro")
    denominator = max(np.linalg.norm(gram_x, ord="fro") * np.linalg.norm(gram_y, ord="fro"), 1e-12)
    return float(1.0 - (numerator / denominator))


def compare_latent_spaces(reference_latent: np.ndarray, comparison_latent: np.ndarray) -> dict[str, float]:
    """Compare two latent spaces with several alignment-free similarity metrics."""
    reference_latent = np.asarray(reference_latent, dtype=float)
    comparison_latent = np.asarray(comparison_latent, dtype=float)
    if reference_latent.shape[0] != comparison_latent.shape[0]:
        common_rows = min(reference_latent.shape[0], comparison_latent.shape[0])
        reference_latent = reference_latent[:common_rows]
        comparison_latent = comparison_latent[:common_rows]

    _, _, disparity = procrustes(reference_latent, comparison_latent)
    cka_similarity = _linear_cka_similarity(reference_latent, comparison_latent)

    n_neighbors = min(30, max(1, reference_latent.shape[0] - 1))
    neigh = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    neigh.fit(reference_latent)
    _, reference_nn = neigh.kneighbors(reference_latent, n_neighbors=n_neighbors)
    reference_nn = reference_nn[:, 1:]

    neigh_comp = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    neigh_comp.fit(comparison_latent)
    _, comparison_nn = neigh_comp.kneighbors(comparison_latent, n_neighbors=n_neighbors)
    comparison_nn = comparison_nn[:, 1:]

    def _jaccard_overlap(k: int) -> float:
        if k <= 0 or reference_latent.shape[0] <= 1:
            return 1.0
        k = min(k, n_neighbors)
        knn_ref = reference_nn[:, :k]
        knn_comp = comparison_nn[:, :k]
        overlaps = []
        for row_ref, row_comp in zip(knn_ref, knn_comp, strict=False):
            union = np.union1d(row_ref, row_comp)
            if union.size == 0:
                overlaps.append(1.0)
            else:
                overlaps.append(np.intersect1d(row_ref, row_comp).size / union.size)
        return float(np.mean(overlaps))

    jaccard_knn10 = _jaccard_overlap(10)
    jaccard_knn30 = _jaccard_overlap(30)

    max_neighbors = max(1, min(10, reference_latent.shape[0] // 2 - 1))
    trust = trustworthiness(reference_latent, comparison_latent, n_neighbors=max_neighbors)
    continuity = trustworthiness(comparison_latent, reference_latent, n_neighbors=max_neighbors)

    return {
        "procrustes_distance": float(disparity),
        "cka_similarity": float(cka_similarity),
        "jaccard_knn10": float(jaccard_knn10),
        "jaccard_knn30": float(jaccard_knn30),
        "trustworthiness": float(trust),
        "continuity": float(continuity),
    }


def compare_clusterings(reference_labels: np.ndarray, comparison_labels: np.ndarray) -> dict[str, float]:
    """Compare two sets of cluster labels with ARI and NMI."""
    return {
        "ari": float(adjusted_rand_score(reference_labels, comparison_labels)),
        "nmi": float(normalized_mutual_info_score(reference_labels, comparison_labels)),
    }


def aggregate_results(results: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate per-fraction evaluation results into a single DataFrame."""
    if not results:
        return pd.DataFrame(columns=[
            "fraction",
            "n_rows",
            "train_time_seconds",
            "peak_memory_mb",
            "active_latent_dims_pct",
            "procrustes_distance",
            "cka_similarity",
            "jaccard_knn10",
            "jaccard_knn30",
            "trustworthiness",
            "continuity",
            "ari",
            "nmi",
        ])

    reference_result = max(results, key=lambda item: item["n_rows"])
    rows: list[dict[str, Any]] = []
    for result in results:
        diagnostics = diagnose_latent_collapse(
            None,
            result.get("diagnostics", {}).get("latent_dim_kl"),
            result.get("diagnostics", {}).get("latent_dim_variance"),
        )
        comparison = compare_latent_spaces(reference_result["latent_embeddings"], result["latent_embeddings"])
        if "cluster_labels" in result and "cluster_labels" in reference_result:
            clustering = compare_clusterings(reference_result["cluster_labels"], result["cluster_labels"])
        else:
            clustering = {"ari": float("nan"), "nmi": float("nan")}

        rows.append(
            {
                "fraction": result["fraction"],
                "n_rows": result["n_rows"],
                "train_time_seconds": result["train_time_seconds"],
                "peak_memory_mb": result["peak_memory_mb"],
                "active_latent_dims_pct": diagnostics["active_latent_dims_pct"],
                "procrustes_distance": comparison["procrustes_distance"],
                "cka_similarity": comparison["cka_similarity"],
                "jaccard_knn10": comparison["jaccard_knn10"],
                "jaccard_knn30": comparison["jaccard_knn30"],
                "trustworthiness": comparison["trustworthiness"],
                "continuity": comparison["continuity"],
                "ari": clustering["ari"],
                "nmi": clustering["nmi"],
            }
        )

    return pd.DataFrame(rows)


def plot_performance_vs_n(results_df: pd.DataFrame) -> None:
    """Plot evaluation metrics versus training size with a simple elbow annotation."""
    import matplotlib.pyplot as plt

    if results_df.empty:
        raise ValueError("results_df must not be empty")

    metrics = [
        "procrustes_distance",
        "cka_similarity",
        "ari",
        "nmi",
        "trustworthiness",
        "active_latent_dims_pct",
    ]
    if any(metric not in results_df.columns for metric in metrics):
        raise ValueError("results_df is missing expected columns")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.flatten()

    full_reference = results_df.iloc[-1]
    for ax, metric in zip(axes, metrics, strict=False):
        values = results_df[metric].to_numpy(dtype=float)
        ax.plot(results_df["n_rows"], values, marker="o")
        ax.set_xlabel("N rows")
        ax.set_ylabel(metric)
        ax.set_xscale("log")

        if metric == "active_latent_dims_pct":
            if np.isfinite(values).all():
                ax.set_ylim(0.0, 1.0)
        if metric in {"procrustes_distance", "cka_similarity", "trustworthiness"}:
            if metric == "cka_similarity":
                ax.set_ylim(0.0, 1.0)
            elif metric == "trustworthiness":
                ax.set_ylim(0.0, 1.0)

        if metric in {"procrustes_distance", "cka_similarity", "ari", "nmi", "trustworthiness"}:
            reference_value = full_reference[metric]
            tolerance = 0.05
            candidates = np.where(np.abs(values - reference_value) <= tolerance)[0]
            if candidates.size:
                elbow_n = int(results_df.iloc[candidates[0]]["n_rows"])
                ax.axvline(elbow_n, color="red", linestyle="--", linewidth=1)
                ax.annotate(
                    f"elbow ~{elbow_n}",
                    xy=(elbow_n, values[candidates[0]]),
                    xytext=(8, 8),
                    textcoords="offset points",
                    color="red",
                )

    ax_time = fig.add_subplot(2, 2, 4)
    ax_time.plot(results_df["n_rows"], results_df["train_time_seconds"], marker="o", color="tab:blue")
    ax_time.set_xlabel("N rows")
    ax_time.set_ylabel("Training time (s)")
    ax_time.set_xscale("log")
    ax_time_twin = ax_time.twinx()
    ax_time_twin.plot(results_df["n_rows"], results_df["peak_memory_mb"], marker="s", color="tab:orange")
    ax_time_twin.set_ylabel("Peak memory (MB)")
    plt.show()


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    data = rng.normal(size=(400, 8))
    data[:, 0] += 2.0
    data[:, 1] -= 1.0

    def dummy_train_fn(subsample: np.ndarray, *, random_seed: int, **_: object):
        model = object()

        def encode_fn(batch: np.ndarray) -> np.ndarray:
            return batch[:, :3].astype(float)

        diagnostics = {
            "latent_dim_kl": np.array([0.12, 0.04, 0.21], dtype=float),
            "latent_dim_variance": np.array([0.5, 0.01, 0.9], dtype=float),
        }
        return model, encode_fn, diagnostics

    results = run_subsample_experiment(
        data=data,
        n_fractions=[0.25, 0.5, 1.0],
        vae_train_fn=dummy_train_fn,
        random_seed=11,
    )
    df = aggregate_results(results)
    print(df)
    plot_performance_vs_n(df)
