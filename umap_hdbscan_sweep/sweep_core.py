"""UMAP + HDBSCAN primitives that can FIT on the dedup set and INFER on everything else.

This differs from ``clust_regime_sweep/clustering_core.py`` in two ways that matter:

1. **GPU first.** cuML's UMAP/HDBSCAN are used when CUDA is available, falling back to
   umap-learn/hdbscan on CPU. At the 95-file cohort scale the CPU path is a formality --
   umap-learn with ``random_state`` set is single-threaded and will not finish.
2. **Models are kept, not discarded.** ``clustering_core`` fits and throws the model away
   because it only ever clusters one fixed matrix. Here the fitted UMAP and HDBSCAN are
   returned so ``transform`` / ``approximate_predict`` can label rows the fit never saw --
   that is the only way all 5.08B reads reach SigProfiler.

HDBSCAN is fit with ``prediction_data=True`` so ``approximate_predict`` works. On CPU that
costs extra memory (the fit keeps its kd-tree); on GPU it is required by cuML.

Note on the MST cache: ``clustering_core.hdbscan_grid`` shares one joblib ``memory`` cache
across a ``min_cluster_size`` grid at fixed ``min_samples``, so the expensive
mutual-reachability/MST step runs once. cuML has no equivalent, so on GPU every
(min_cluster_size, min_samples) pair is a fresh fit -- acceptable because the GPU fit is
orders of magnitude faster than the CPU one it replaces.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np


# ── Backend selection ──────────────────────────────────────────────────────────

def gpu_available() -> bool:
    """True when cuML is importable and a CUDA device is visible.

    ``UV_VAE_FORCE_CPU=1`` overrides, which is how the CPU parity path is exercised.
    """
    if os.environ.get("UV_VAE_FORCE_CPU", "") == "1":
        return False
    if os.environ.get("CUDA_VISIBLE_DEVICES", "unset") == "":
        return False
    try:
        import cuml  # noqa: F401
    except Exception:
        return False
    return True


# ── GPU budget (co-tenancy with a running trainer) ─────────────────────────────

# How much of this process's budget RMM (cuDF/cuML) may take, per stage. The two
# allocators SHARE one budget -- gpu_budget.apply caps RMM first and gives torch the
# remainder -- so these are the split, not two separate ceilings.
#
#   embed  torch encodes, nothing touches cuML            -> torch gets everything
#   sweep  cuML fits UMAP/HDBSCAN, torch is never loaded  -> RMM gets everything it can
#          (gpu_budget.resolve_rmm_share clamps at 0.9, deliberately: it refuses to
#           starve torch entirely, and the leftover is harmless headroom here)
#   apply  torch encodes AND cuML transforms/predicts     -> a genuine split
STAGE_RMM_SHARE = {"embed": 0.0, "sweep": 0.9, "apply": 0.5}


def device_memory_gb() -> tuple[float, float] | None:
    """(free, total) GB on device 0, or None without CUDA.

    ``mem_get_info`` reports what the *driver* sees, so free memory already accounts
    for every other process on the card -- which is the entire point when a trainer
    is running alongside this one.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    except Exception:
        return None
    return free_bytes / (1024 ** 3), total_bytes / (1024 ** 3)


def apply_gpu_budget(
    stage: str,
    budget_gb: float | None = None,
    rmm_share: float | None = None,
    require_free: bool = True,
    verbose: bool = True,
) -> dict:
    """Cap this process at ``budget_gb``, refusing to start if that is not actually free.

    Two distinct things are happening:

    * **The cap.** ``gpu_budget.apply`` bounds torch and the RMM pool against one shared
      budget. Without this call the shell's ``UV_VAE_RMM_SHARE`` is inert -- nothing reads
      it -- and cuML's pool grows toward all free memory.
    * **The co-tenancy check.** The cap is per process, so two capped processes sum. When
      a trainer already holds part of the card, asking for a slice larger than what is
      free does not fail at ``apply`` time (torch's cap is a ceiling, not a reservation) --
      it fails much later, mid-fit, as a CUDA OOM. Checking ``mem_get_info`` up front turns
      that into an immediate, readable error.

    Note that RMM's ``initial_pool_size`` really is allocated up front (a quarter of the
    pool), so the sweep's footprint appears on the card as soon as this returns.
    """
    import sys

    from uv_vae import gpu_budget

    share = STAGE_RMM_SHARE.get(stage, gpu_budget.DEFAULT_RMM_SHARE) if rmm_share is None else rmm_share
    budget = gpu_budget.resolve_budget_gb(budget_gb)
    if os.environ.get("UV_VAE_SKIP_FREE_CHECK", "") == "1":
        require_free = False

    memory = device_memory_gb()
    if memory is not None:
        free_gb, total_gb = memory
        in_use_gb = total_gb - free_gb
        if verbose:
            print(
                f"[gpu] device {total_gb:.1f} GB total, {free_gb:.1f} GB free "
                f"({in_use_gb:.1f} GB held by other processes); this stage wants {budget:.1f} GB",
                file=sys.stderr, flush=True,
            )
        # 1 GB covers this process's own CUDA context plus fragmentation.
        if require_free and budget > free_gb - 1.0:
            raise SystemExit(
                f"GPU budget {budget:.1f} GB does not fit: only {free_gb:.1f} GB is free "
                f"({in_use_gb:.1f} GB is already held, likely the trainer).\n"
                f"  Lower it:            SWEEP_GPU_GB={max(1.0, free_gb - 2.0):.0f}\n"
                f"  Or wait for the trainer to finish, or run this stage with FORCE_CPU=1.\n"
                f"  To proceed anyway (risks a mid-fit OOM): UV_VAE_SKIP_FREE_CHECK=1"
            )

    report = gpu_budget.apply(budget_gb=budget, rmm_share=share, verbose=verbose)
    payload = report.as_dict() if hasattr(report, "as_dict") else {"budget_gb": budget}
    payload["stage"] = stage
    payload["rmm_share"] = share
    if memory is not None:
        payload["device_free_gb_before"] = round(memory[0], 2)
        payload["device_total_gb"] = round(memory[1], 2)
    return payload


def _as_float32(X: Any) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(X, dtype=np.float32))


def _to_host(array: Any) -> np.ndarray:
    """Bring a cuDF/cuPy/torch array back to a numpy array on the host."""
    for attr in ("to_numpy", "get"):
        if hasattr(array, attr):
            return np.asarray(getattr(array, attr)())
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    return np.asarray(array)


# ── UMAP ───────────────────────────────────────────────────────────────────────

@dataclass
class UmapConfig:
    n_neighbors: int = 30
    min_dist: float = 0.05
    n_components: int = 2
    metric: str = "euclidean"
    seed: int = 42
    # MUST stay pinned rather than None. umap-learn derives transform()'s epoch count from
    # its input size when n_epochs is None (100 rows<=10k, else 30), which makes transform
    # results depend on the batch size they were computed in -- so a 5.08B-row pass would
    # give different coordinates for different --transform-batch-size values. With n_epochs
    # set, transform always runs n_epochs//3 and batching becomes invariant. 200 is also
    # umap-learn's own fit default above 10k rows, so pinning it does not change the fit.
    n_epochs: int = 200
    # Layout parameters, all at umap-learn's own defaults so an existing grid is unchanged.
    # spread pairs with min_dist -- together they fit the (a, b) of the output kernel, which
    # is why neither is interpretable alone. repulsion_strength weights negative samples
    # against positive ones; set_op_mix_ratio blends union (1.0) and intersection (0.0) of
    # the fuzzy simplicial sets.
    spread: float = 1.0
    repulsion_strength: float = 1.0
    set_op_mix_ratio: float = 1.0

    def label(self) -> str:
        # Off-default parameters append, exactly as HdbscanConfig.label does, so a config
        # that does not touch them keeps the historical "nn30_md0.05_nc2" directory name
        # and stage 2's resume-by-directory-match still finds completed cells.
        # min_dist keeps plain str() formatting, NOT :g -- ":g" would render 0.0 as "0",
        # renaming every existing min_dist=0.0 cell directory and forcing a full re-run.
        parts = [f"nn{self.n_neighbors}", f"md{self.min_dist}", f"nc{self.n_components}"]
        if self.spread != 1.0:
            parts.append(f"sp{self.spread:g}")
        if self.repulsion_strength != 1.0:
            parts.append(f"rs{self.repulsion_strength:g}")
        if self.set_op_mix_ratio != 1.0:
            parts.append(f"som{self.set_op_mix_ratio:g}")
        return "_".join(parts)

    def as_dict(self) -> dict:
        return {
            "n_neighbors": self.n_neighbors,
            "min_dist": self.min_dist,
            "n_components": self.n_components,
            "metric": self.metric,
            "seed": self.seed,
            "n_epochs": self.n_epochs,
            "spread": self.spread,
            "repulsion_strength": self.repulsion_strength,
            "set_op_mix_ratio": self.set_op_mix_ratio,
        }


@dataclass
class FittedUmap:
    model: Any
    embedding: np.ndarray
    config: UmapConfig
    backend: str

    def transform(self, X: np.ndarray, batch_size: int = 5_000_000) -> np.ndarray:
        """Project unseen rows into the fitted embedding, in batches.

        **Batching is not coordinate-exact, and cannot be made so.** umap-learn prunes each
        transform's graph at ``graph.data.max() / n_epochs`` -- a threshold computed from
        the batch -- and its numba negative-sampling loop consumes a shared ``rng_state``
        in an order that depends on how many points are in flight. Pinning
        ``UmapConfig.n_epochs`` removes one source of drift but not these two.

        What was measured (3 separated 16-D blobs, 1500 held-out rows, batch 400 vs
        single-shot): max coordinate drift 2.73 units, yet **identical HDBSCAN labels for
        every row, ARI = 1.0000**. The drift is small next to inter-cluster distance, so it
        moves points within their cluster rather than across a boundary.

        Do not assume that carries to the real latent space, where clusters are far less
        separated -- boundary reads can flip. Treat ``transform_batch_size`` as a recorded
        parameter of a run (stage 2 and 3 both persist it) and hold it fixed when comparing
        results, rather than as a free performance knob.
        """
        X = _as_float32(X)
        if X.shape[0] <= batch_size:
            return _to_host(self.model.transform(X)).astype(np.float32, copy=False)
        chunks = [
            _to_host(self.model.transform(X[start:start + batch_size])).astype(np.float32, copy=False)
            for start in range(0, X.shape[0], batch_size)
        ]
        return np.concatenate(chunks, axis=0)


def fit_umap(X: np.ndarray, config: UmapConfig, use_gpu: bool | None = None) -> FittedUmap:
    """Fit UMAP on ``X`` and return the model alongside its embedding."""
    use_gpu = gpu_available() if use_gpu is None else use_gpu
    X = _as_float32(X)

    if use_gpu:
        from cuml.manifold import UMAP as CuUMAP

        model = CuUMAP(
            n_neighbors=int(config.n_neighbors),
            min_dist=float(config.min_dist),
            n_components=int(config.n_components),
            metric=config.metric,
            random_state=int(config.seed),
            n_epochs=int(config.n_epochs),
            spread=float(config.spread),
            repulsion_strength=float(config.repulsion_strength),
            set_op_mix_ratio=float(config.set_op_mix_ratio),
        )
        embedding = _to_host(model.fit_transform(X)).astype(np.float32, copy=False)
        return FittedUmap(model=model, embedding=embedding, config=config, backend="cuml")

    from umap import UMAP

    # low_memory switches pynndescent to the memory-efficient kNN build; without it the
    # C extension segfaults on OOM above ~1M rows.
    model = UMAP(
        n_neighbors=int(config.n_neighbors),
        min_dist=float(config.min_dist),
        n_components=int(config.n_components),
        metric=config.metric,
        random_state=int(config.seed),
        n_epochs=int(config.n_epochs),
        spread=float(config.spread),
        repulsion_strength=float(config.repulsion_strength),
        set_op_mix_ratio=float(config.set_op_mix_ratio),
        low_memory=True,
    )
    embedding = np.asarray(model.fit_transform(X), dtype=np.float32)
    return FittedUmap(model=model, embedding=embedding, config=config, backend="umap-learn")


# ── HDBSCAN ────────────────────────────────────────────────────────────────────

@dataclass
class HdbscanConfig:
    min_cluster_size: int = 500
    min_samples: int = 25
    metric: str = "euclidean"
    # Both default to hdbscan's own defaults, so an existing grid behaves exactly as it did
    # before these fields existed -- including ``label()``, which still returns
    # "mcs500_ms25" unless one of them is moved off its default. That is load-bearing:
    # stage 2 resumes by matching those directory names, so a changed label would silently
    # re-run every completed cell into a new directory.
    cluster_selection_method: str = "eom"
    cluster_selection_epsilon: float = 0.0

    def label(self) -> str:
        parts = [f"mcs{self.min_cluster_size}", f"ms{self.min_samples}"]
        if self.cluster_selection_method != "eom":
            parts.append(self.cluster_selection_method)
        if self.cluster_selection_epsilon:
            parts.append(f"eps{self.cluster_selection_epsilon:g}")
        return "_".join(parts)

    def as_dict(self) -> dict:
        return {
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
            "metric": self.metric,
            "cluster_selection_method": self.cluster_selection_method,
            "cluster_selection_epsilon": self.cluster_selection_epsilon,
        }


@dataclass
class FittedHdbscan:
    clusterer: Any
    labels: np.ndarray
    probabilities: np.ndarray
    config: HdbscanConfig
    backend: str
    dbcv: float | None = None

    def predict(self, X: np.ndarray, batch_size: int = 5_000_000) -> tuple[np.ndarray, np.ndarray]:
        """Assign unseen rows to the fitted clusters via ``approximate_predict``.

        Points too far from any cluster come back as noise (-1), which is the intended
        behaviour: a read whose latent position is unlike anything in the fit set should
        not be forced into a cluster.
        """
        X = _as_float32(X)
        approximate_predict = _approximate_predict_fn(self.backend)
        if X.shape[0] <= batch_size:
            labels, probabilities = approximate_predict(self.clusterer, X)
            return (
                _to_host(labels).astype(np.int32, copy=False),
                _to_host(probabilities).astype(np.float32, copy=False),
            )
        label_chunks, probability_chunks = [], []
        for start in range(0, X.shape[0], batch_size):
            labels, probabilities = approximate_predict(self.clusterer, X[start:start + batch_size])
            label_chunks.append(_to_host(labels).astype(np.int32, copy=False))
            probability_chunks.append(_to_host(probabilities).astype(np.float32, copy=False))
        return np.concatenate(label_chunks), np.concatenate(probability_chunks)


def _approximate_predict_fn(backend: str):
    if backend == "cuml":
        from cuml.cluster.hdbscan import approximate_predict

        return approximate_predict
    import hdbscan as hdbscan_cpu

    return hdbscan_cpu.approximate_predict


def fit_hdbscan(
    X: np.ndarray,
    config: HdbscanConfig,
    use_gpu: bool | None = None,
    core_dist_n_jobs: int = 1,
    cache_dir: str | None = None,
) -> FittedHdbscan:
    """Fit HDBSCAN with prediction data enabled so unseen rows can be labelled later."""
    use_gpu = gpu_available() if use_gpu is None else use_gpu
    X = _as_float32(X)

    # Forwarded only when it is off its default. 0.0 IS the default in both backends, so
    # omitting it is behaviourally identical -- and it means a cuML build that predates the
    # parameter still runs every existing grid rather than dying on an unexpected kwarg.
    epsilon = (
        {"cluster_selection_epsilon": float(config.cluster_selection_epsilon)}
        if config.cluster_selection_epsilon else {}
    )

    if use_gpu:
        from cuml.cluster.hdbscan import HDBSCAN as CuHDBSCAN

        clusterer = CuHDBSCAN(
            min_cluster_size=int(config.min_cluster_size),
            min_samples=int(config.min_samples),
            metric=config.metric,
            cluster_selection_method=config.cluster_selection_method,
            prediction_data=True,
            gen_min_span_tree=True,
            **epsilon,
        )
        clusterer.fit(X)
        backend = "cuml"
    else:
        import hdbscan as hdbscan_cpu
        from joblib import Memory

        # hdbscan calls memory.cache(...) unconditionally, so memory=None raises
        # AttributeError -- its own default is Memory(None), the no-op cache, not None.
        memory = Memory(cache_dir, verbose=0) if cache_dir else Memory(None, verbose=0)
        clusterer = hdbscan_cpu.HDBSCAN(
            min_cluster_size=int(config.min_cluster_size),
            min_samples=int(config.min_samples),
            metric=config.metric,
            cluster_selection_method=config.cluster_selection_method,
            core_dist_n_jobs=int(core_dist_n_jobs),
            gen_min_span_tree=True,
            memory=memory,
            prediction_data=True,
            **epsilon,
        )
        clusterer.fit(np.asarray(X, dtype=np.float64))
        backend = "hdbscan"

    # relative_validity_ (DBCV) needs the MST; cuML has not always exposed it.
    try:
        dbcv = float(clusterer.relative_validity_)
    except Exception:
        dbcv = None

    return FittedHdbscan(
        clusterer=clusterer,
        labels=_to_host(clusterer.labels_).astype(np.int32, copy=False),
        probabilities=_to_host(clusterer.probabilities_).astype(np.float32, copy=False),
        config=config,
        backend=backend,
        dbcv=dbcv,
    )


# ── Grid construction ──────────────────────────────────────────────────────────

@dataclass
class SweepGrid:
    """The cartesian product, ordered so the expensive axis (UMAP) varies slowest."""

    umap_configs: list[UmapConfig] = field(default_factory=list)
    hdbscan_configs: list[HdbscanConfig] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.umap_configs) * len(self.hdbscan_configs)

    def iter_cells(self) -> Iterator[tuple[UmapConfig, HdbscanConfig]]:
        for umap_config in self.umap_configs:
            for hdbscan_config in self.hdbscan_configs:
                yield umap_config, hdbscan_config


def build_grid(
    n_neighbors: list[int],
    min_dist: list[float],
    n_components: list[int],
    min_cluster_size: list[int],
    min_samples: list[int],
    metric: str = "euclidean",
    seed: int = 42,
    n_epochs: int = 200,
) -> SweepGrid:
    umap_configs = [
        UmapConfig(n_neighbors=nn, min_dist=md, n_components=nc, metric=metric, seed=seed,
                   n_epochs=n_epochs)
        for nn in n_neighbors
        for md in min_dist
        for nc in n_components
    ]
    hdbscan_configs = [
        HdbscanConfig(min_cluster_size=mcs, min_samples=ms, metric=metric)
        # min_samples varies slowest within HDBSCAN so the CPU joblib MST cache (shared
        # across min_cluster_size at fixed min_samples) stays warm for a full inner run.
        for ms in min_samples
        for mcs in min_cluster_size
    ]
    return SweepGrid(umap_configs=umap_configs, hdbscan_configs=hdbscan_configs)
