from __future__ import annotations

import numpy as np

from uv_vae.evaluation import aggregate_results, diagnose_latent_collapse, run_subsample_experiment


class DummyVAE:
    def __init__(self, latent_dim: int) -> None:
        self.latent_dim = latent_dim

    def encode(self, data: np.ndarray) -> np.ndarray:
        return data[:, : self.latent_dim].astype(float)


def dummy_vae_train_fn(data: np.ndarray, *, random_seed: int, **_: object):
    model = DummyVAE(latent_dim=3)

    def encode_fn(batch: np.ndarray) -> np.ndarray:
        return model.encode(batch)

    diagnostics = {
        "latent_dim_kl": np.array([0.1, 0.05, 0.2], dtype=float),
        "latent_dim_variance": np.array([0.4, 0.01, 0.7], dtype=float),
    }
    return model, encode_fn, diagnostics


def test_run_subsample_experiment_and_aggregate_results() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(120, 8))

    results = run_subsample_experiment(
        data=data,
        n_fractions=[0.25, 0.5, 1.0],
        vae_train_fn=dummy_vae_train_fn,
        random_seed=7,
    )

    assert len(results) == 3
    assert all("latent_embeddings" in result for result in results)

    df = aggregate_results(results)
    assert list(df.columns) == [
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
    ]

    collapse = diagnose_latent_collapse(None, np.array([0.1, 0.0, 0.2]), np.array([0.4, 0.01, 0.6]))
    assert collapse["active_latent_dims_pct"] == 2 / 3
