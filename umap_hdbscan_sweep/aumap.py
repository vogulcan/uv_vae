"""Approximate UMAP (aUMAP) -- replace transform() with a kNN lookup in latent space.

Phase A measured the cost split of the UMAP stage: fitting 5 M rows takes 119 s, while
``transform()`` of the remaining 152.5 M takes ~3291 s. 98% of the stage is spent placing
rows the fit never saw, and ``transform()`` is doing real optimisation work to do it --
umap-learn/cuML run ``n_epochs // 3`` SGD steps per batch to settle each new point against
the fitted graph.

aUMAP (Wassenaar et al., arXiv:2404.04001) skips that entirely. A new point's 2-D position
is the inverse-distance-weighted mean of the 2-D positions of its k nearest neighbours in
the ORIGINAL space -- here the 16-D VAE latent. No optimisation, no graph, no epochs: one
kNN query plus a gather.

    z_new = sum_i w_i * z_ref(i)      w_i = (1 / (d_i + eps)) / sum_j (1 / (d_j + eps))

The reference set is the UMAP fit set itself, which already has exact embeddings from the
fit. The paper reports mean error of 0.083--0.256 embedding standard deviations against
true ``transform()`` -- close, but "close" is dataset-dependent, which is why this module
ships with a validation entry point rather than being wired straight into the sweep.

**This is an approximation and it is not free of assumptions.** It reproduces
``transform()`` well only where the fit set is dense enough that a new point's neighbours
bracket it. A point in a sparse region gets pulled toward whichever island happens to be
nearest, where true ``transform()`` would have let repulsion push it into the gap. Phase A
found no out-of-sample repulsion bias here (shift ratio 0.994), which is mildly reassuring
for that failure mode, but the validation below is what actually decides it.

Run the validation before trusting it::

    python umap_hdbscan_sweep/aumap.py \\
        --embed-dir <stage1-output> \\
        --output-dir umap_hdbscan_sweep/aumap_validation
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

# Walk up to the directory that holds uv_vae/ rather than counting levels: these scripts
# get reorganised into subfolders (umap/, hdbscan/, tests/), and a hard-coded parents[N]
# silently resolves to the wrong root the moment one moves, failing on `import uv_vae`.
REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "uv_vae").is_dir()),
                 Path(__file__).resolve().parents[1])
for candidate in (REPO_ROOT / "uv_vae", REPO_ROOT, Path(__file__).resolve().parent):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np

import sweep_core as core

# Full-cohort row count from stage 0 minus the 5 M fit set, used to extrapolate a probe
# timing to what the real run would cost. Only affects reporting.
FULL_COHORT_HELD_ROWS = 152_501_580


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# ── the estimator ──────────────────────────────────────────────────────────────

class AUmap:
    """kNN-interpolation stand-in for ``FittedUmap.transform``.

    Built from an already-fitted UMAP plus the latent rows that fit saw. Holds a kNN
    index over those latent rows and their known 2-D coordinates; ``transform`` queries
    the index and blends.

    ``n_interp_neighbors`` is this method's only real hyperparameter and is *independent*
    of UMAP's own ``n_neighbors`` -- it controls how many reference points are blended,
    not the manifold estimate. Larger values smooth more (safer in sparse regions, blurs
    genuine boundaries); smaller values track ``transform()`` more sharply but can snap a
    point onto a single neighbour.
    """

    def __init__(
        self,
        reference_latent: np.ndarray,
        reference_embedding: np.ndarray,
        n_interp_neighbors: int = 15,
        use_gpu: bool | None = None,
        eps: float = 1e-8,
    ) -> None:
        if len(reference_latent) != len(reference_embedding):
            raise ValueError(
                f"reference_latent has {len(reference_latent):,} rows but "
                f"reference_embedding has {len(reference_embedding):,}"
            )
        self.use_gpu = core.gpu_available() if use_gpu is None else use_gpu
        self.n_interp_neighbors = int(n_interp_neighbors)
        self.eps = float(eps)
        self.reference_embedding = np.ascontiguousarray(
            np.asarray(reference_embedding, dtype=np.float32)
        )

        reference_latent = np.ascontiguousarray(np.asarray(reference_latent, dtype=np.float32))
        started = perf_counter()
        if self.use_gpu:
            from cuml.neighbors import NearestNeighbors as CuNN

            # Brute force is exact and, at 16 dimensions, GPU-bound rather than
            # memory-bound -- cuML tiles the distance matrix so the reference set size
            # does not have to fit as an N_query x N_ref block.
            self._index = CuNN(n_neighbors=self.n_interp_neighbors, algorithm="brute")
            self.backend = "cuml"
        else:
            from sklearn.neighbors import NearestNeighbors

            self._index = NearestNeighbors(n_neighbors=self.n_interp_neighbors, algorithm="auto")
            self.backend = "sklearn"
        self._index.fit(reference_latent)
        self.index_build_seconds = perf_counter() - started

    @classmethod
    def from_fitted_umap(
        cls,
        fitted: core.FittedUmap,
        reference_latent: np.ndarray,
        n_interp_neighbors: int = 15,
        **kwargs,
    ) -> "AUmap":
        """Build from a ``sweep_core.fit_umap`` result and the latent rows it was fit on."""
        return cls(
            reference_latent=reference_latent,
            reference_embedding=fitted.embedding,
            n_interp_neighbors=n_interp_neighbors,
            **kwargs,
        )

    def _blend(self, distances, indices):
        """Inverse-distance-weighted mean of the neighbours' embeddings.

        Kept backend-agnostic: the arithmetic is identical under numpy and cupy, so the
        GPU path never round-trips to the host mid-batch.
        """
        xp = _array_module(distances)
        reference = self.reference_embedding
        if xp.__name__ == "cupy":
            reference = self._reference_embedding_device(xp)

        weights = 1.0 / (distances + self.eps)
        weights = weights / weights.sum(axis=1, keepdims=True)
        neighbour_coords = reference[indices]            # (B, k, n_components)
        return (neighbour_coords * weights[:, :, None]).sum(axis=1)

    def _reference_embedding_device(self, xp):
        cached = getattr(self, "_reference_embedding_gpu", None)
        if cached is None:
            cached = xp.asarray(self.reference_embedding)
            self._reference_embedding_gpu = cached
        return cached

    def transform(self, X: np.ndarray, batch_size: int = 2_000_000) -> np.ndarray:
        """Project unseen rows by blending their latent-space neighbours' coordinates.

        Unlike ``FittedUmap.transform``, this **is** batch-invariant: every row's result
        depends only on its own neighbours, so batching changes nothing but memory use.
        That removes the coordinate-drift caveat documented on ``FittedUmap.transform``.
        """
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        chunks = []
        for start in range(0, X.shape[0], batch_size):
            distances, indices = self._index.kneighbors(X[start:start + batch_size])
            chunks.append(core._to_host(self._blend(distances, indices)).astype(np.float32, copy=False))
        return np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]

    def transform_multi_k(
        self, X: np.ndarray, k_values: list[int], batch_size: int = 2_000_000
    ) -> dict[int, np.ndarray]:
        """One kNN query, several blend widths -- for choosing ``n_interp_neighbors``.

        The search is the expensive half, so sweeping k by re-querying wastes most of the
        work. Query once at ``max(k_values)``; the neighbour lists come back sorted by
        distance, so a prefix of width k is exactly the k-NN result.
        """
        if max(k_values) > self.n_interp_neighbors:
            raise ValueError(
                f"index was built for {self.n_interp_neighbors} neighbours, "
                f"cannot slice {max(k_values)}"
            )
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        collected: dict[int, list] = {k: [] for k in k_values}
        for start in range(0, X.shape[0], batch_size):
            distances, indices = self._index.kneighbors(X[start:start + batch_size])
            for k in k_values:
                blended = self._blend(distances[:, :k], indices[:, :k])
                collected[k].append(core._to_host(blended).astype(np.float32, copy=False))
        return {k: np.concatenate(parts, axis=0) for k, parts in collected.items()}


def _array_module(array):
    """cupy for device arrays, numpy otherwise -- without importing cupy needlessly."""
    module_name = type(array).__module__.split(".")[0]
    if module_name == "cupy":
        import cupy

        return cupy
    return np


# ── validation ─────────────────────────────────────────────────────────────────

def knn_overlap(a: np.ndarray, b: np.ndarray, k: int = 15) -> float:
    """Mean fraction of shared k-nearest neighbours between two embeddings of the same rows.

    Invariant to rotation, reflection, translation and scale, which matters because two
    UMAP-space coordinate sets are only ever comparable up to those. A raw coordinate
    error can look large while the neighbourhood structure -- the thing clustering
    consumes -- is identical.
    """
    from sklearn.neighbors import NearestNeighbors

    def neighbours(space: np.ndarray) -> np.ndarray:
        index = NearestNeighbors(n_neighbors=k + 1).fit(space)
        # column 0 is the point itself
        return index.kneighbors(space, return_distance=False)[:, 1:]

    neighbours_a, neighbours_b = neighbours(a), neighbours(b)
    shared = [
        len(np.intersect1d(row_a, row_b, assume_unique=True))
        for row_a, row_b in zip(neighbours_a, neighbours_b)
    ]
    return float(np.mean(shared) / k)


def compare_embeddings(true_coords: np.ndarray, approx_coords: np.ndarray, k: int = 15) -> dict:
    """Score an approximate embedding against the true ``transform()`` of the same rows."""
    errors = np.linalg.norm(true_coords - approx_coords, axis=1)
    # The paper's unit: error relative to the spread of the embedding itself, so the
    # number means the same thing across configs with different spatial extents.
    embedding_std = float(np.std(true_coords))
    return {
        "mean_error_raw": round(float(np.mean(errors)), 5),
        "median_error_raw": round(float(np.median(errors)), 5),
        "p95_error_raw": round(float(np.percentile(errors, 95)), 5),
        "embedding_std": round(embedding_std, 5),
        "mean_error_in_std": round(float(np.mean(errors)) / embedding_std, 5),
        "median_error_in_std": round(float(np.median(errors)) / embedding_std, 5),
        "p95_error_in_std": round(float(np.percentile(errors, 95)) / embedding_std, 5),
        "knn_overlap": round(knn_overlap(true_coords, approx_coords, k=k), 5),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate aUMAP against true UMAP transform")
    parser.add_argument("--embed-dir", required=True, help="stage 1 output containing latent.npy")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fit-rows", type=int, default=5_000_000)
    parser.add_argument("--probe-rows", type=int, default=200_000,
                        help="held-out rows scored both ways")
    parser.add_argument("--interp-k", default="5,15,30",
                        help="comma-separated n_interp_neighbors values to compare")
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-budget-gb", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    started = perf_counter()
    args = parse_args()
    k_values = [int(part) for part in args.interp_k.split(",") if part.strip()]

    embed_dir = Path(args.embed_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    latent_path = embed_dir / "latent.npy"
    if not latent_path.exists():
        raise SystemExit(f"latent.npy not found in {embed_dir}")

    if core.gpu_available() and args.gpu_budget_gb is not None:
        core.apply_gpu_budget("sweep", budget_gb=args.gpu_budget_gb)

    latent = np.load(latent_path, mmap_mode="r")
    total_rows, latent_dim = latent.shape
    log(f"{latent_path.name}: {total_rows:,} rows x {latent_dim} dims")

    # Same draw as stage2_sweep.fit_indices, so the reference set is the one the real
    # sweep would have.
    rng = np.random.default_rng(args.seed)
    fit_idx = np.sort(rng.choice(total_rows, size=args.fit_rows, replace=False))
    outside = np.setdiff1d(np.arange(total_rows), fit_idx, assume_unique=True)
    probe_idx = np.sort(
        np.random.default_rng(args.seed + 1).choice(
            outside, size=min(args.probe_rows, len(outside)), replace=False
        )
    )
    log(f"fit set {len(fit_idx):,} rows, probe set {len(probe_idx):,} held-out rows")

    config = core.UmapConfig(
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        n_components=2,
        seed=args.seed,
        n_epochs=200,
    )
    log(f"UMAP config: {config.as_dict()}")

    fit_latent = latent[fit_idx]
    log(f"fitting UMAP on {len(fit_idx):,} rows ...")
    t0 = perf_counter()
    fitted = core.fit_umap(fit_latent, config)
    fit_seconds = perf_counter() - t0
    log(f"  fit in {fit_seconds:.1f}s (backend={fitted.backend})")

    probe_latent = np.ascontiguousarray(latent[probe_idx])

    # Ground truth: what the pipeline does today.
    log(f"true transform() of {len(probe_idx):,} probe rows ...")
    t0 = perf_counter()
    true_coords = fitted.transform(probe_latent)
    true_transform_seconds = perf_counter() - t0
    log(f"  transform in {true_transform_seconds:.1f}s "
        f"(-> {true_transform_seconds * FULL_COHORT_HELD_ROWS / len(probe_idx) / 60:.0f} min "
        f"for the full cohort)")

    log(f"building aUMAP index over {len(fit_idx):,} reference rows ...")
    aumap = AUmap.from_fitted_umap(fitted, fit_latent, n_interp_neighbors=max(k_values))
    log(f"  index built in {aumap.index_build_seconds:.1f}s (backend={aumap.backend})")

    log(f"aUMAP transform of {len(probe_idx):,} probe rows, k in {k_values} ...")
    t0 = perf_counter()
    approx_by_k = aumap.transform_multi_k(probe_latent, k_values, batch_size=args.batch_size)
    aumap_seconds = perf_counter() - t0
    log(f"  aUMAP transform in {aumap_seconds:.1f}s for all {len(k_values)} k values")

    # One query served every k, so per-k cost is the shared search plus a cheap blend.
    per_k_seconds = aumap_seconds / len(k_values)
    extrapolated = per_k_seconds * FULL_COHORT_HELD_ROWS / len(probe_idx)
    log(f"  -> extrapolated full cohort ({FULL_COHORT_HELD_ROWS:,} rows): "
        f"{extrapolated:.0f}s ({extrapolated / 60:.1f} min)")

    quality = {}
    for k in k_values:
        quality[k] = compare_embeddings(true_coords, approx_by_k[k], k=15)
        q = quality[k]
        log(f"  k={k:>3}: error {q['mean_error_in_std']:.3f} std (median "
            f"{q['median_error_in_std']:.3f}, p95 {q['p95_error_in_std']:.3f}), "
            f"kNN-overlap {q['knn_overlap']:.3f}")

    best_k = max(k_values, key=lambda k: quality[k]["knn_overlap"])
    speedup = (true_transform_seconds / per_k_seconds) if per_k_seconds > 0 else float("nan")

    results = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_rows": int(total_rows),
        "latent_dim": int(latent_dim),
        "umap_config": config.as_dict(),
        "fit_rows": int(len(fit_idx)),
        "probe_rows": int(len(probe_idx)),
        "backend": fitted.backend,
        "timing": {
            "umap_fit_seconds": round(fit_seconds, 2),
            "true_transform_probe_seconds": round(true_transform_seconds, 2),
            "true_transform_full_cohort_minutes": round(
                true_transform_seconds * FULL_COHORT_HELD_ROWS / len(probe_idx) / 60, 1
            ),
            "aumap_index_build_seconds": round(aumap.index_build_seconds, 2),
            "aumap_probe_seconds_per_k": round(per_k_seconds, 2),
            "aumap_full_cohort_seconds": round(extrapolated, 1),
            "aumap_full_cohort_minutes": round(extrapolated / 60, 2),
            "speedup_vs_true_transform": round(speedup, 1),
        },
        "quality_by_interp_k": {str(k): quality[k] for k in k_values},
        "best_interp_k_by_knn_overlap": int(best_k),
        "paper_reference_mean_error_in_std": [0.083, 0.256],
        "total_seconds": round(perf_counter() - started, 1),
    }
    out_path = output_dir / "aumap_validation.json"
    out_path.write_text(json.dumps(results, indent=2))
    log(f"wrote {out_path}")

    best = quality[best_k]
    print("\n-- aUMAP validation ------------------------------------")
    print(f"probe rows: {len(probe_idx):,} held out of a {len(fit_idx):,}-row fit")
    print()
    print(f"true transform : {true_transform_seconds:6.1f}s probe  -> "
          f"{results['timing']['true_transform_full_cohort_minutes']:.0f} min full cohort")
    print(f"aUMAP          : {per_k_seconds:6.1f}s probe  -> "
          f"{extrapolated / 60:.1f} min full cohort   ({speedup:.0f}x faster)")
    print(f"index build    : {aumap.index_build_seconds:6.1f}s (one-off, per UMAP fit)")
    print()
    print(f"quality at best k = {best_k}:")
    print(f"  mean error   {best['mean_error_in_std']:.3f} embedding std "
          f"(paper reports 0.083-0.256)")
    print(f"  p95 error    {best['p95_error_in_std']:.3f} embedding std")
    print(f"  kNN-overlap  {best['knn_overlap']:.3f}  "
          f"(1.0 = identical neighbourhood structure)")
    print("--------------------------------------------------------")
    print("\nkNN-overlap is the number that matters: clustering consumes neighbourhood")
    print("structure, not absolute coordinates. Judge acceptability against the Gate 1")
    print("determinism floor -- if two identical UMAP fits disagree by more than aUMAP")
    print("disagrees with transform(), the approximation is inside the noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
