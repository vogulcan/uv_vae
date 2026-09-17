"""The 14 metrics, split by what they actually measure and what they cost to compute.

Two families, because they answer different questions and only one of them needs
cluster labels:

**Embedding quality** -- is the 2-D map a faithful picture of the 16-D latent space?
Computed from the probe's own coordinates, so every fit size is scored on its own
merits without reference to any other run.

    trustworthiness   points that look close in 2-D but are not close in 16-D
    continuity        points close in 16-D that the map pushed apart
    RNX AUC           the two above generalised over every neighbourhood size at once
    spearman          rank correlation of pairwise distance, 16-D vs 2-D
    pearson           linear correlation of the same
    procrustes        geometric disagreement between two runs' coordinates

**Clustering** -- do two runs put the same reads together, and are the groups good?

    ARI / NMI / AMI / Jaccard    agreement between two runs' labels
    silhouette / Davies-Bouldin / Calinski-Harabasz / DBCV    quality within one run

Cost is the reason this module exists rather than a handful of sklearn calls.

``sklearn.manifold.trustworthiness`` materialises the full N x N distance matrix and
argsorts it. At the 250k rows we want to score that is 500 GB, so it is not an option.
Everything local here is instead computed from *truncated neighbour lists*: k-NN out to
``k_max``, from which the co-ranking matrix restricted to the top-left ``k_max`` block
follows directly. That block is all trustworthiness, continuity and the log-weighted RNX
AUC ever read -- the AUC weights K by 1/K, so the far tail it discards is the part that
contributes least. What the truncation does cost is the exact rank of a neighbour that
falls outside the window: those are clamped to ``k_max`` and therefore *under*-penalised,
so both scores are optimistic by a bounded amount. ``clamped_fraction`` is reported next
to every one of them; keep it small (< 0.05) and the bias is negligible.

The co-ranking machinery follows Lee et al. 2015, "Multi-scale similarities in stochastic
neighbour embedding", Neurocomputing 169:246-261 -- the RNX AUC source already cited in
README.md. Trustworthiness/continuity follow Venna & Kaski 2001 in the normalisation
sklearn also uses, so numbers here are comparable to sklearn's on inputs small enough to
run both (``tests`` checks exactly that).

Nothing in this module imports torch or cuML at module scope; the GPU is used when
present and every path falls back to CPU, matching the rest of the sweep.
"""
from __future__ import annotations

from typing import Any

import numpy as np

import sweep_core as core

# Which tier each metric belongs to, so callers and the JSON agree on one vocabulary.
# "full"  -- linear in rows, runs on the whole probe
# "knn"   -- needs neighbour lists, runs on the mid subsample
# "pair"  -- needs pairwise distances, runs on the small subsample
METRIC_TIERS = {
    # embedding quality
    "trustworthiness": "knn",
    "continuity": "knn",
    "rnx_auc": "knn",
    "spearman_distance": "pair",
    "pearson_distance": "pair",
    "procrustes_disparity": "full",
    # clustering
    "adjusted_rand": "full",
    "normalized_mutual_info": "full",
    "adjusted_mutual_info": "full",
    "jaccard": "full",
    "davies_bouldin": "full",
    "calinski_harabasz": "full",
    "silhouette": "pair",
    "dbcv": "clusterer",
}

EMBEDDING_METRICS = (
    "trustworthiness", "continuity", "rnx_auc",
    "spearman_distance", "pearson_distance", "procrustes_disparity",
)
CLUSTERING_METRICS = (
    "adjusted_rand", "normalized_mutual_info", "adjusted_mutual_info", "jaccard",
    "silhouette", "davies_bouldin", "calinski_harabasz", "dbcv",
)

# Direction each metric moves when the embedding/clustering gets better. Used by the
# convergence verdict, which otherwise cannot tell "settled high" from "settled low".
HIGHER_IS_BETTER = {
    "trustworthiness": True, "continuity": True, "rnx_auc": True,
    "spearman_distance": True, "pearson_distance": True,
    "procrustes_disparity": False,
    "adjusted_rand": True, "normalized_mutual_info": True,
    "adjusted_mutual_info": True, "jaccard": True,
    "silhouette": True, "davies_bouldin": False, "calinski_harabasz": True,
    "dbcv": True,
}


# ── neighbour lists ────────────────────────────────────────────────────────────

# Neighbour-list work is chunked to a fixed number of *elements*, not rows, because peak
# memory is rows x k and k varies by more than an order of magnitude across callers. A
# fixed row count that is comfortable at k=400 allocates 12x more at k=5000 and OOMs.
# These budgets reproduce the previous fixed row defaults at k≈400, so nothing about the
# earlier runs changes: only the loop trip count moves, never the result.
KNN_QUERY_ELEMENTS = 50_000_000
RANK_CHUNK_ELEMENTS = 20_000_000


def knn_indices(
    X: np.ndarray,
    k: int,
    use_gpu: bool | None = None,
    query_chunk: int | None = None,
) -> np.ndarray:
    """Indices of the ``k`` nearest neighbours of every row, self excluded.

    Brute force on purpose. At 2 and 16 dimensions an exact GPU brute-force scan beats a
    tree, and more importantly an *approximate* index would put its own recall error
    inside every metric downstream -- the thing being measured here is a few percent, and
    NN-Descent's recall miss is the same order.

    ``query_chunk`` defaults to whatever keeps one batch of results near
    ``KNN_QUERY_ELEMENTS``; pass an explicit value to override.
    """
    use_gpu = core.gpu_available() if use_gpu is None else use_gpu
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    n = X.shape[0]
    if k >= n:
        raise ValueError(f"k={k} needs at least {k + 1} rows, got {n}")
    if query_chunk is None:
        query_chunk = max(1, KNN_QUERY_ELEMENTS // (k + 1))

    if use_gpu:
        from cuml.neighbors import NearestNeighbors as CuNN

        index = CuNN(n_neighbors=k + 1, algorithm="brute")
    else:
        from sklearn.neighbors import NearestNeighbors

        index = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    index.fit(X)

    chunks = []
    for start in range(0, n, query_chunk):
        stop = min(start + query_chunk, n)
        found = index.kneighbors(X[start:stop], return_distance=False)
        found = np.asarray(core._to_host(found), dtype=np.int64)
        chunks.append(_drop_self(found, np.arange(start, stop), k))
    return np.concatenate(chunks, axis=0).astype(np.int32, copy=False)


def _drop_self(indices: np.ndarray, row_ids: np.ndarray, k: int) -> np.ndarray:
    """Remove each row's own index, then keep the nearest ``k`` that remain.

    The self-match is normally column 0, but exact duplicate rows -- which this latent
    space has plenty of, the cohort is 157 M reads over far fewer distinct positions --
    tie at distance 0 and can be returned in any order. Masking by identity rather than
    slicing off column 0 is what makes the result correct in the presence of ties.
    """
    is_self = indices == row_ids[:, None]
    # stable argsort on the boolean keeps the distance ordering inside each group
    order = np.argsort(is_self, axis=1, kind="stable")
    return np.take_along_axis(indices, order, axis=1)[:, :k]


def neighbour_ranks(
    nn_query: np.ndarray,
    nn_reference: np.ndarray,
    n_points: int,
    chunk_rows: int | None = None,
) -> np.ndarray:
    """Position of each ``nn_query`` entry inside the same row of ``nn_reference``.

    Returns ``k_reference`` for entries that do not appear in the reference list at all --
    the clamp the module docstring warns about. Returned ranks are 0-based.

    Implemented as one batched ``searchsorted``. numpy has no per-row searchsorted, so
    each row is shifted into its own disjoint band of a global key space
    (``row * n_points + index``); because the rows are laid out in order, the concatenated
    per-row-sorted keys are globally sorted and a single searchsorted resolves every row
    at once. That is O(nk log k) instead of the O(n * n_points) an explicit lookup table
    of positions would cost.

    Each chunk holds several int64 arrays of ``chunk_rows x k_reference``, so the chunk is
    sized in elements rather than rows -- see ``RANK_CHUNK_ELEMENTS``.
    """
    n, k_reference = nn_reference.shape
    if nn_query.shape[0] != n:
        raise ValueError(f"row mismatch: query {nn_query.shape[0]}, reference {n}")
    if chunk_rows is None:
        chunk_rows = max(1, RANK_CHUNK_ELEMENTS // max(k_reference, 1))
    out = np.empty(nn_query.shape, dtype=np.int32)

    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)
        reference = nn_reference[start:stop].astype(np.int64)
        query = nn_query[start:stop].astype(np.int64)
        rows = reference.shape[0]

        order = np.argsort(reference, axis=1, kind="stable")
        sorted_reference = np.take_along_axis(reference, order, axis=1)
        band = (np.arange(rows, dtype=np.int64) * n_points)[:, None]

        keys = (band + sorted_reference).ravel()
        probes = (band + query).ravel()
        located = np.searchsorted(keys, probes)
        located = np.minimum(located, keys.size - 1)
        hit = keys[located] == probes
        out[start:stop] = np.where(hit, order.ravel()[located], k_reference).reshape(rows, -1)
    return out


# ── embedding quality: local (neighbour lists) ─────────────────────────────────

def _rank_penalty_score(ranks: np.ndarray, k: int, n: int) -> float:
    """The shared Venna-Kaski sum behind both trustworthiness and continuity.

    ``ranks`` are the 0-based positions, in the *other* space, of this space's k nearest
    neighbours. A neighbour that is also within the other space's top k contributes
    nothing; one at rank r > k contributes r - k.
    """
    inside = ranks[:, :k]
    penalty = np.maximum(inside + 1 - k, 0).astype(np.float64).sum()
    normaliser = 2.0 / (n * k * (2.0 * n - 3.0 * k - 1.0))
    return float(1.0 - normaliser * penalty)


def trustworthiness(rank_high_of_low: np.ndarray, k: int, n: int) -> float:
    """Penalises points the map *invented* as neighbours -- close in 2-D, far in 16-D."""
    return _rank_penalty_score(rank_high_of_low, k, n)


def continuity(rank_low_of_high: np.ndarray, k: int, n: int) -> float:
    """Penalises real 16-D neighbours the map tore apart. The complement of the above."""
    return _rank_penalty_score(rank_low_of_high, k, n)


def qnx_curve(rank_high_of_low: np.ndarray) -> np.ndarray:
    """``Q_NX(K)`` for every ``K`` up to the neighbour-list width.

    ``Q_NX(K)`` is the mean fraction of each point's K nearest neighbours that are shared
    between the two spaces, i.e. the top-left K x K block of the co-ranking matrix. The
    whole curve is read off one small 2-D histogram of (rank in 16-D, position in 2-D)
    followed by a 2-D cumulative sum -- the histogram is only (k_max+1) x k_max, which is
    why truncating the neighbour lists collapses this from an N x N object to a trivial one.
    """
    n, k = rank_high_of_low.shape
    positions = np.broadcast_to(np.arange(k, dtype=np.int64), (n, k))
    # rank lives in [0, k] because k marks "outside the window"; position lives in [0, k)
    flat = rank_high_of_low.astype(np.int64).ravel() * k + positions.ravel()
    histogram = np.bincount(flat, minlength=(k + 1) * k).reshape(k + 1, k)
    cumulative = histogram.cumsum(axis=0).cumsum(axis=1)
    counts = np.array([cumulative[K - 1, K - 1] for K in range(1, k + 1)], dtype=np.float64)
    return counts / (np.arange(1, k + 1, dtype=np.float64) * n)


def rnx_auc(qnx: np.ndarray, n: int) -> float:
    """Log-weighted area under the ``R_NX(K)`` curve (Lee et al. 2015).

    ``R_NX`` rescales ``Q_NX`` against what a random embedding would score, so 0 means
    "no better than chance" and 1 means "perfect" at every scale. The 1/K weighting is the
    paper's: it makes the number scale-independent, and it is also what lets the truncated
    curve stand in for the full one, since the discarded large-K tail carries the least
    weight.
    """
    K = np.arange(1, len(qnx) + 1, dtype=np.float64)
    usable = K < (n - 1)
    K, qnx = K[usable], qnx[usable]
    rnx = ((n - 1.0) * qnx - K) / (n - 1.0 - K)
    weights = 1.0 / K
    return float((rnx * weights).sum() / weights.sum())


def local_quality(
    high: np.ndarray,
    low: np.ndarray,
    k: int = 15,
    k_max: int = 100,
    use_gpu: bool | None = None,
) -> dict:
    """Trustworthiness, continuity and RNX AUC for one 16-D / 2-D pair of the same rows."""
    n = high.shape[0]
    k_max = int(min(k_max, n - 1))
    if k >= k_max:
        raise ValueError(f"k={k} must be below k_max={k_max}")

    nn_high = knn_indices(high, k_max, use_gpu=use_gpu)
    nn_low = knn_indices(low, k_max, use_gpu=use_gpu)
    rank_high_of_low = neighbour_ranks(nn_low, nn_high, n_points=n)
    rank_low_of_high = neighbour_ranks(nn_high, nn_low, n_points=n)

    # How often the truncation had to guess. Both scores are optimistic by an amount that
    # grows with this, so it is reported rather than hidden.
    clamped = float(
        ((rank_high_of_low[:, :k] >= k_max).mean() + (rank_low_of_high[:, :k] >= k_max).mean()) / 2
    )
    curve = qnx_curve(rank_high_of_low)
    return {
        "trustworthiness": round(trustworthiness(rank_high_of_low, k, n), 5),
        "continuity": round(continuity(rank_low_of_high, k, n), 5),
        "rnx_auc": round(rnx_auc(curve, n), 5),
        "qnx_at_k": round(float(curve[k - 1]), 5),
        "clamped_fraction": round(clamped, 5),
        "k": k,
        "k_max": k_max,
        "rows": n,
    }


# ── embedding quality: global (pairwise distances) ─────────────────────────────

def distance_correlations(
    high: np.ndarray,
    low: np.ndarray,
    pairs: int = 20_000_000,
    seed: int = 42,
    chunk: int = 2_000_000,
) -> dict:
    """Spearman and Pearson correlation between 16-D and 2-D pairwise distances.

    Sampled pairs, not all of them. At the small subsample every pair would be ~2e8
    distances held twice in float64 plus an argsort for the rank correlation; a uniform
    sample of 2e7 pairs estimates a correlation to well under 0.001 and costs a tenth of
    that. The sample is drawn from a fixed seed so every fit size is scored on the *same*
    pairs and the comparison across sizes is exact rather than statistical.
    """
    n = high.shape[0]
    rng = np.random.default_rng(seed)
    high = np.asarray(high, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)

    high_distances, low_distances = [], []
    remaining = int(pairs)
    while remaining > 0:
        take = min(chunk, remaining)
        left = rng.integers(0, n, size=take)
        right = rng.integers(0, n, size=take)
        keep = left != right
        left, right = left[keep], right[keep]
        high_distances.append(np.linalg.norm(high[left] - high[right], axis=1))
        low_distances.append(np.linalg.norm(low[left] - low[right], axis=1))
        remaining -= take

    high_distances = np.concatenate(high_distances)
    low_distances = np.concatenate(low_distances)

    from scipy.stats import pearsonr, spearmanr

    spearman = float(spearmanr(high_distances, low_distances).statistic)
    pearson = float(pearsonr(high_distances, low_distances).statistic)
    return {
        "spearman_distance": round(spearman, 5),
        "pearson_distance": round(pearson, 5),
        "pairs_sampled": int(high_distances.size),
        "rows": n,
    }


def procrustes_disparity(source: np.ndarray, target: np.ndarray) -> float:
    """Residual after the best rotation / reflection / scale / translation of one onto the other.

    UMAP's objective is invariant to all four, so two runs of it are only ever comparable
    once that freedom is removed. 0 means the two coordinate sets are the same shape;
    1 means the alignment explained nothing. This is scipy's normalisation (both inputs
    scaled to unit Frobenius norm) computed directly, to avoid scipy's copies at 1 M rows.

    **Read this one with care on UMAP output.** Procrustes can only remove a *global*
    similarity transform, but what differs between two UMAP fits is mostly where each
    island got placed relative to the others -- and that placement is arbitrary, decided by
    the random initialisation, not by the data. A measured example from the real-backend
    run: two fits whose clusterings agreed at ARI 0.9989 scored procrustes disparity 0.93,
    which reads as total disagreement. It is a genuine signal about global geometry and a
    misleading one about whether the clustering moved. When the two disagree, believe ARI.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source = source - source.mean(axis=0)
    target = target - target.mean(axis=0)
    source_norm = np.linalg.norm(source)
    target_norm = np.linalg.norm(target)
    if source_norm == 0 or target_norm == 0:
        return float("nan")
    source /= source_norm
    target /= target_norm
    _, singular_values, _ = np.linalg.svd(source.T @ target)
    trace = singular_values.sum()
    # ||A||^2 + ||B||^2 - 2 tr = 2 - 2 tr, both being unit norm
    return float(max(0.0, 2.0 - 2.0 * trace))


# ── clustering: agreement between two labellings ───────────────────────────────

def _contingency(a: np.ndarray, b: np.ndarray):
    from scipy.sparse import coo_matrix

    a_codes, a_index = np.unique(a, return_inverse=True)
    b_codes, b_index = np.unique(b, return_inverse=True)
    matrix = coo_matrix(
        (np.ones(a_index.size, dtype=np.int64), (a_index, b_index)),
        shape=(a_codes.size, b_codes.size),
    ).tocsr()
    return matrix


def pair_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Pair-counting Jaccard: of all pairs grouped together by *either* run, what share by both.

    Unlike ARI this is not chance-corrected, so it is the more pessimistic of the two and
    a useful cross-check: a high ARI with a low Jaccard means the agreement is mostly the
    two runs agreeing on what to keep *apart*.
    """
    matrix = _contingency(a, b)
    counts = matrix.data.astype(np.float64)
    row_sums = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float64)
    column_sums = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64)

    def pairs(values):
        return float((values * (values - 1.0) / 2.0).sum())

    both = pairs(counts)
    in_a = pairs(row_sums)
    in_b = pairs(column_sums)
    union = in_a + in_b - both
    return float(both / union) if union > 0 else float("nan")


def label_agreement(a: np.ndarray, b: np.ndarray) -> dict:
    """ARI, NMI, AMI and Jaccard between two labellings of the same rows.

    HDBSCAN noise (-1) is carried through as an ordinary label rather than dropped. That is
    the conservative reading: two runs that disagree about *which* reads are unclusterable
    have genuinely disagreed, and dropping noise would let a run that calls 90% of the
    cohort noise score perfectly on the remainder.
    """
    from sklearn.metrics import (
        adjusted_mutual_info_score,
        adjusted_rand_score,
        normalized_mutual_info_score,
    )

    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"label shape mismatch: {a.shape} vs {b.shape}")
    return {
        "adjusted_rand": round(float(adjusted_rand_score(a, b)), 5),
        "normalized_mutual_info": round(float(normalized_mutual_info_score(a, b)), 5),
        "adjusted_mutual_info": round(float(adjusted_mutual_info_score(a, b)), 5),
        "jaccard": round(pair_jaccard(a, b), 5),
    }


# ── clustering: quality of one labelling ───────────────────────────────────────

def cluster_quality(
    coordinates: np.ndarray,
    labels: np.ndarray,
    silhouette_coordinates: np.ndarray | None = None,
    silhouette_labels: np.ndarray | None = None,
) -> dict:
    """Davies-Bouldin and Calinski-Harabasz on everything, silhouette on the small subsample.

    Silhouette is the odd one out: it is O(n^2) in distance computations, while the other
    two only need cluster centroids and so are linear. Passing the small subsample for it
    is the whole reason those arguments exist.

    All three are undefined for a single cluster; they come back as NaN rather than raising,
    because a fit size collapsing to one cluster is a *result*, not an error.
    """
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    coordinates = np.asarray(coordinates, dtype=np.float32)
    labels = np.asarray(labels)
    result: dict[str, Any] = {
        "n_clusters": int(len(set(labels.tolist()) - {-1})),
        "noise_fraction": round(float(np.mean(labels == -1)), 5),
    }
    for name, function in (
        ("davies_bouldin", davies_bouldin_score),
        ("calinski_harabasz", calinski_harabasz_score),
    ):
        try:
            result[name] = round(float(function(coordinates, labels)), 5)
        except Exception:
            result[name] = None

    if silhouette_coordinates is None:
        silhouette_coordinates, silhouette_labels = coordinates, labels
    try:
        result["silhouette"] = round(
            float(silhouette_score(np.asarray(silhouette_coordinates, dtype=np.float32),
                                   np.asarray(silhouette_labels))), 5
        )
        result["silhouette_rows"] = int(len(silhouette_labels))
    except Exception:
        result["silhouette"] = None
        result["silhouette_rows"] = int(len(silhouette_labels)) if silhouette_labels is not None else 0
    return result


# ── convergence verdict ────────────────────────────────────────────────────────

def converged_at(sizes: list[int], values: list, tolerance: float) -> int | None:
    """Smallest size whose value, and every larger size's, sits within ``tolerance`` of the last.

    "And every larger size's" is the part that matters. A metric can cross the reference
    on its way somewhere else; requiring the whole tail to stay inside the band is what
    separates convergence from a coincidence.
    """
    usable = [(size, value) for size, value in zip(sizes, values)
              if value is not None and np.isfinite(value)]
    if len(usable) < 2:
        return None
    reference = usable[-1][1]
    for position, (size, _) in enumerate(usable):
        if all(abs(value - reference) <= tolerance for _, value in usable[position:]):
            return int(size)
    return None
