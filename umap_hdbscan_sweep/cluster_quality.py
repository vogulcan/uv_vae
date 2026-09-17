#!/usr/bin/env python
"""Internal validity metrics for a density-based clustering: DBCV, probabilities,
persistence, connectivity.

These measure the geometry of the partition in the space it was fitted in. **None of them
knows anything about mutational biology**, which is both why they are cheap and why they must
never be read alone: the 2026-08-04 UMAP cells separate cleanly by every density criterion
here and are still 87 % single-trinucleotide-context, i.e. they score well precisely because
UMAP manufactured the separation the metric then rewards. Read these beside
``spectrum_metrics.top_channel_share``, never instead of it.

Nothing here is ranked or combined. The module returns numbers.

What is and is not computed
---------------------------
* **DBCV** -- delegated to ``rescore_dbcv.score_cell`` (already in this repo), which scores a
  subsample. Sampling is **stratified** by default: the same total budget is spent equally
  across clusters rather than in proportion to their size, so small clusters are no longer
  dropped for arriving under the minimum and there is no cluster-count ceiling. Because the
  sampled cluster sizes are then deliberately not the true ones, the aggregate is rebuilt
  from the per-cluster scores weighted by TRUE sizes -- see ``dbcv``. One limit this module
  does not hide: HDBSCAN's own ``relative_validity_`` is a cheaper MST approximation and
  *not* true DBCV.
* **Probabilities** -- mean and median are reported but the threshold fractions are the
  honest summary. ``probabilities_`` is lambda_p / lambda_max *within each cluster*, so it is
  renormalised per cluster: a partition of many small tight clusters posts a high mean without
  being better resolved.
* **Persistence** -- ``cluster_persistence_``, HDBSCAN's own stability measure. Partly
  circular under ``cluster_selection_method='eom'`` because that is the quantity EOM
  maximises; genuinely informative under ``leaf``.
* **Connectivity** -- fraction of each point's k nearest neighbours carrying its label.
  **Monotone in cluster count**: merge everything into one cluster and it is 1.0 by
  construction. It is reported next to ``n_clusters`` for that reason and is meaningless
  without it.
* **CDbw** -- deliberately absent. No maintained, validated Python implementation exists, it
  needs multiple representatives per cluster (O(n^2)-ish), and it would land as a third
  subsample-based density score next to DBCV without adding an independent question.
* **Dunn / generalised Dunn** -- deliberately absent, see the note below.

References
----------
* HDBSCAN cluster stability, and the excess-of-mass selection EOM maximises:
  Campello, Moulavi & Sander (2013), "Density-Based Clustering Based on Hierarchical Density
  Estimates", PAKDD, LNCS 7819, 160-172. Extended: Campello, Moulavi, Zimek & Sander (2015),
  ACM TKDD 10(1):5.
* ``cluster_persistence_`` / ``relative_validity_``: McInnes, Healy & Astels (2017),
  "hdbscan: Hierarchical density based clustering", JOSS 2(11):205. The library documents
  ``relative_validity_`` as "a fast approximation of the DBCV score" computed from the
  mutual-reachability MST, and warns it "might not be an objective measure of the goodness of
  clustering. It may only be used to compare results across different choices of
  hyper-parameters".
* DBCV: Moulavi, Jaskowiak, Campello, Zimek & Sander (2014), "Density-Based Clustering
  Validation", SDM, 839-847, doi:10.1137/1.9781611973440.96.
* Connectivity: Handl & Knowles (2005), "Computational cluster validation in post-genomic
  data analysis", Bioinformatics 21(15):3201-3212.
* CDbw: Halkidi & Vazirgiannis (2008), Pattern Recognition Letters 29(6):773-786.

On Dunn
-------
There is no canonical method called the "robust Dunn index". What exists is the family of
**generalised Dunn indices** in Bezdek & Pal (1998), "Some new indexes of cluster validity",
IEEE Trans. SMC-B 28(3):301-315, which identifies two deficiencies making the original Dunn
index (Dunn 1974, J. Cybernetics 4(1):95-104) "overly sensitive to noisy clusters" and
proposes generalisations "not as brittle to outliers".

Those generalisations are not adopted here, for a reason stated in that same paper: its
finding that minimum interset distance is the least reliable basis for a validity index holds
"when the clusters are expected to form volumetric clouds". HDBSCAN clusters are arbitrarily
shaped by construction, which is precisely the case DBCV was introduced to handle -- Moulavi
et al. open by noting that indices designed for globular clusters "may fail" on density-based
ones. A Dunn variant would also compress 1,000+ clusters into one scalar driven by the single
closest pair out of ~660,000, which cannot say WHICH clusters are poor.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Connectivity and DBCV are O(n log n) and O(n^2)-ish respectively; both are computed on a
# subsample. Fixed seeds keep the sample identical across cells so scores are comparable.
DEFAULT_CONNECTIVITY_ROWS = 200_000
DEFAULT_DBCV_ROWS = 25_000
# Points drawn from EVERY cluster. Fixing this rather than the total is what makes DBCV
# comparable between models: sampled DBCV's bias depends on per-cluster resolution, so a
# fixed total budget scores a 176-cluster cell and a 1,330-cluster cell at different
# resolutions and their scores cannot be read against each other.
#
# 400 rather than a smaller number because resolution, not total budget, is what decides
# whether DBCV can see contamination at all. Measured on 30 clusters where the true scores
# were clean +0.861 / contaminated -0.033 (a gap of 0.894), scoring at N points per cluster
# recovered:  N=50 -> 19% of the gap,  N=100 -> 44%,  N=200 -> 67%,  N=400 -> 97%.
# At N=50 a thoroughly contaminated clustering scores +0.72 and reads as excellent. Sampling
# thins the contamination away faster than it thins the clusters.
DEFAULT_DBCV_PER_CLUSTER = 400
PROBABILITY_THRESHOLDS = (0.5, 0.8)
# score_cell drops any cluster with fewer sampled points than this, so the stratified
# allocation must guarantee at least this many per cluster or it defeats its own purpose.
MIN_PER_CLUSTER = 4
# k-DBCV's own default. It refuses to score when the largest cluster's n_i x n_i distance
# matrix would exceed this, and signals that refusal with a -1 -- which is also a legal DBCV
# value, so the wrapper predicts the same quantity rather than trusting the return.
KDBCV_MEMORY_CUTOFF_GB = 25.0


def probability_stats(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    """Membership strength over the ASSIGNED points only.

    Noise carries probability 0 by definition, so including it would make this a restatement
    of the noise fraction with extra steps.
    """
    assigned = labels >= 0
    if not assigned.any():
        return {"prob_n_assigned": 0}

    values = np.asarray(probabilities, dtype=np.float64)[assigned]
    stats = {
        "prob_n_assigned": int(assigned.sum()),
        "prob_mean": float(values.mean()),
        "prob_median": float(np.median(values)),
        "prob_p10": float(np.percentile(values, 10)),
        "prob_p90": float(np.percentile(values, 90)),
    }
    for threshold in PROBABILITY_THRESHOLDS:
        key = str(threshold).replace(".", "")
        stats[f"prob_frac_above_{key}"] = float((values > threshold).mean())
    return stats


def persistence_stats(clusterer) -> dict:
    """HDBSCAN's per-cluster stability, if the backend exposes it.

    Stability is the excess of mass of a cluster in the condensed hierarchy -- the integral
    of its membership over the density scales it survives (Campello, Moulavi & Sander 2013).
    ``cluster_persistence_`` normalises it to 0-1: 1.0 is "persists over all distance
    scales", 0.0 is "perfectly ephemeral".

    EOM selection chooses the set of clusters MAXIMISING total stability, so under
    ``cluster_selection_method='eom'`` the sum is the objective the fit already optimised and
    is not independent evidence about the fit. Under ``leaf`` it is.
    """
    values = getattr(clusterer, "cluster_persistence_", None)
    if values is None:
        return {}
    values = np.asarray(_to_host(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}
    return {
        "persistence_mean": float(values.mean()),
        "persistence_median": float(np.median(values)),
        "persistence_min": float(values.min()),
        "persistence_max": float(values.max()),
        # Total stability mass. EOM maximises the sum, so under eom this is close to the
        # objective the fit already optimised and is not independent evidence.
        "persistence_sum": float(values.sum()),
    }


def relative_validity(clusterer) -> dict:
    """HDBSCAN's own ``relative_validity_``: a DBCV approximation over ALL fit points.

    Needs ``gen_min_span_tree=True`` on the fit, which retains the mutual-reachability MST
    the clustering already built -- so it costs almost nothing and, unlike the sampled DBCV
    above, involves no subsampling at all.

    **It is a cross-check, not a replacement.** Measured against full DBCV over 30 clusterings
    spanning separated blobs, heavy overlap, rings/moons and noise-contaminated data:
    Spearman +0.81, but 83 of 435 pairwise rankings discordant, and the disagreement is
    structured rather than random -- it penalises noisy clusterings hard (full +0.72 vs rel
    +0.35 at 10% noise) and over-rates arbitrarily-shaped ones (full +0.88 vs rel +0.94 on
    rings). Since noise fraction varies across this grid, ranking cells by it would import
    that bias directly. Record it, compare it to ``dbcv``, and treat a large gap as a signal
    about noise handling rather than as a score.

    The library's own caveat says the same thing: it "may only be used to compare results
    across different choices of hyper-parameters" (McInnes, Healy & Astels 2017).
    """
    value = getattr(clusterer, "relative_validity_", None)
    if value is None:
        return {}
    try:
        value = float(_to_host(value))
    except (TypeError, ValueError):
        return {}
    return {"relative_validity": value} if np.isfinite(value) else {}


def connectivity(space: np.ndarray, labels: np.ndarray, n_neighbours: int = 10,
                 max_rows: int = DEFAULT_CONNECTIVITY_ROWS, seed: int = 0) -> dict:
    """Fraction of each point's k nearest neighbours that share its label.

    Neighbours are searched in the FULL space but scored only for sampled query points, so
    the answer is about the real neighbourhood structure rather than the subsample's.

    Rises monotonically as clusters merge -- a single cluster scores 1.0 -- so it says
    'the labels are locally coherent', not 'the clustering is good'. Always read with
    n_clusters.
    """
    from sklearn.neighbors import NearestNeighbors

    assigned = np.nonzero(labels >= 0)[0]
    if assigned.size < n_neighbours + 1:
        return {"connectivity_n": 0}

    reference = np.asarray(space[assigned], dtype=np.float32)
    reference_labels = np.asarray(labels)[assigned]

    rng = np.random.default_rng(seed)
    if assigned.size > max_rows:
        query_positions = np.sort(rng.choice(assigned.size, size=max_rows, replace=False))
    else:
        query_positions = np.arange(assigned.size)

    index = NearestNeighbors(n_neighbors=n_neighbours + 1, algorithm="auto").fit(reference)
    _, neighbours = index.kneighbors(reference[query_positions])
    # Column 0 is the query point itself; scoring it would add a constant 1/(k+1).
    neighbour_labels = reference_labels[neighbours[:, 1:]]
    agreement = (neighbour_labels == reference_labels[query_positions][:, None]).mean(axis=1)

    return {
        "connectivity_n": int(query_positions.size),
        "connectivity_k": int(n_neighbours),
        "connectivity_mean": float(agreement.mean()),
        "connectivity_median": float(np.median(agreement)),
        # Points whose entire neighbourhood agrees -- the interior of a cluster.
        "connectivity_frac_pure": float((agreement == 1.0).mean()),
        # Points that agree with fewer than half; a boundary or a mis-assignment.
        "connectivity_frac_below_half": float((agreement < 0.5).mean()),
    }


def _allocate(sizes: np.ndarray, budget: int) -> np.ndarray:
    """Split ``budget`` points across clusters as evenly as their sizes allow.

    Water-filling: everyone gets an equal quota; clusters smaller than the quota contribute
    all their points and their surplus is redistributed over the rest. The result spends the
    whole budget whenever the clusters can absorb it, which matters because runtime is set by
    the TOTAL sampled points, not by how they are spread -- an allocation that under-spends
    is paying the same order of cost for a worse estimate.
    """
    quota = np.zeros(sizes.size, dtype=np.int64)
    remaining = np.arange(sizes.size)
    left = int(budget)
    while remaining.size and left > 0:
        per = left // remaining.size
        if per == 0:                       # fewer points left than clusters
            quota[remaining[:left]] += 1
            break
        small = sizes[remaining] <= per
        if small.any():
            idx = remaining[small]
            quota[idx] = sizes[idx]
            left -= int(sizes[idx].sum())
            remaining = remaining[~small]
        else:
            quota[remaining] = per
            break
    return quota


def stratified_sample(labels: np.ndarray, budget: int | None = None,
                      per_cluster: int | None = None, seed: int = 0) -> np.ndarray:
    """Row positions covering EVERY cluster. Two modes, and the choice matters.

    ``per_cluster=N`` -- take N points from every cluster (capped by its size). **This is the
    mode to use when comparing DBCV across models.** A fixed *total* budget spread over k
    clusters gives each cluster ``budget/k`` points, so a 176-cluster cell is scored at 142
    points per cluster and a 1,330-cluster cell at 18. Sampled DBCV is biased by sample
    resolution, so those two numbers carry different biases and are not comparable. Fixing
    points-per-cluster fixes the resolution, which is what makes the scores comparable.

    ``budget=N`` -- spend a fixed total, water-filled equally across clusters (a cluster
    smaller than its quota returns the surplus to the rest). Bounded cost, but the resolution
    -- and therefore the bias -- moves with the cluster count.

    Uniform sampling, which neither mode does, allocates in proportion to cluster size, so at
    high cluster counts small clusters draw too few points to score at all.
    """
    if (budget is None) == (per_cluster is None):
        raise ValueError("pass exactly one of budget= or per_cluster=")

    assigned = np.nonzero(labels >= 0)[0]
    if assigned.size == 0:
        return np.empty(0, dtype=np.int64)

    _, inverse = np.unique(labels[assigned], return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    grouped, sizes = assigned[order], np.bincount(inverse)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])

    quota = (np.minimum(sizes, int(per_cluster)) if per_cluster is not None
             else _allocate(sizes, budget))
    rng = np.random.default_rng(seed)
    picked = []
    for i, (start, size, take) in enumerate(zip(starts, sizes, quota)):
        if take <= 0:
            continue
        block = grouped[start:start + size]
        picked.append(block if take >= size
                      else rng.choice(block, size=int(take), replace=False))
    if not picked:
        return np.empty(0, dtype=np.int64)
    return np.sort(np.concatenate(picked))


def dbcv(space: np.ndarray, labels: np.ndarray, max_rows: int = DEFAULT_DBCV_ROWS,
         max_clusters: int | None = None, seed: int = 0, stratified: bool = True,
         min_per_cluster: int = MIN_PER_CLUSTER,
         per_cluster_rows: int | None = DEFAULT_DBCV_PER_CLUSTER,
         backend: str = "auto") -> dict:
    """Density-based validity over a subsample, via rescore_dbcv.score_cell.

    Delegated rather than reimplemented so this and the standalone rescore path cannot drift.

    Two things make this scoreable at any cluster count, where the uniform version refused
    above ~500:

    **Stratified sampling.** Points are allocated equally across clusters instead of in
    proportion to their size, so no cluster arrives under ``MIN_SAMPLED_POINTS_PER_CLUSTER``
    just for being small.

    ``per_cluster_rows`` (the default) takes a FIXED number of points from every cluster,
    which is what makes scores comparable between models -- see ``stratified_sample``. Set it
    to None to fall back to a fixed total ``max_rows`` budget, which bounds cost but lets the
    per-cluster resolution, and therefore the bias, drift with the cluster count.

    Cost is dominated by the O(k^2) loop over cluster PAIRS, not by the point count: measured
    at a fixed 6,000 points, 50 -> 400 clusters costs 0.39s -> 11.02s, while at a fixed 30
    clusters, 1,500 -> 12,000 points costs 0.26s -> 2.68s. So scoring more points per cluster
    is comparatively cheap and raising ``per_cluster_rows`` buys accuracy at a discount.

    ``backend="auto"`` prefers k-DBCV (KD-tree intercluster distances, up to 42x faster at
    k=400 and agreeing to within 0.001) and falls back to the hdbscan implementation when it
    is not installed. ``"hdbscan"`` forces the original.

    **Size reweighting.** DBCV's aggregate is sum_i (|C_i| / n) * V_i, weighted by cluster
    size -- and validity_index computes those weights from the SAMPLE. Under stratification
    the sampled sizes are deliberately not the true ones, so its aggregate would weight a
    400k-point cluster like a 4k-point one. Taking the per-cluster scores and reweighting by
    the true sizes restores the quantity uniform sampling was estimating. Both are reported:
    ``dbcv`` is the size-weighted (comparable) figure, ``dbcv_sample_weighted`` is what
    validity_index returned.

    ``max_clusters=None`` means no ceiling. Pass an int to keep the old refusal.
    """
    import rescore_dbcv

    n_clusters = int(labels.max()) + 1 if (labels >= 0).any() else 0
    if max_clusters is not None and n_clusters > max_clusters:
        return {"dbcv": None,
                "dbcv_note": f"{n_clusters} clusters exceeds max_clusters={max_clusters}"}

    total = labels.shape[0]
    if stratified and per_cluster_rows:
        sample = stratified_sample(labels, per_cluster=max(int(per_cluster_rows),
                                                           int(min_per_cluster)), seed=seed)
    elif stratified:
        # Never sample fewer than the floor score_cell drops clusters below, or stratifying
        # would guarantee every cluster is dropped -- the failure it exists to prevent.
        budget = max(int(max_rows), n_clusters * int(min_per_cluster))
        sample = stratified_sample(labels, budget=budget, seed=seed)
    else:
        rng = np.random.default_rng(seed)
        sample = (np.sort(rng.choice(total, size=max_rows, replace=False))
                  if total > max_rows else np.arange(total))

    used = backend
    try:
        if backend in ("auto", "kdbcv"):
            try:
                result, used = score_kdbcv(space, labels, sample), "kdbcv"
            except ImportError:
                if backend == "kdbcv":
                    raise
                result, used = rescore_dbcv.score_cell(
                    np.asarray(space, dtype=np.float64), np.asarray(labels), sample,
                    per_cluster=stratified), "hdbscan"
        else:
            result, used = rescore_dbcv.score_cell(
                np.asarray(space, dtype=np.float64), np.asarray(labels), sample,
                per_cluster=stratified), "hdbscan"
    except Exception as exc:  # noqa: BLE001 - a DBCV failure must not cost the cell
        return {"dbcv": None, "dbcv_note": f"{type(exc).__name__}: {exc}"}
    result["dbcv_backend"] = used
    # score_cell explains a null score under "note"; this wrapper uses "dbcv_note" for the
    # cases it declines itself. Two keys for one concept means a caller checking the wrong
    # one reports "no reason given" for a refusal that did give a reason -- so normalise.
    if "note" in result:
        result["dbcv_note"] = result.pop("note")
    result["dbcv_sampling"] = (
        f"stratified_{per_cluster_rows}_per_cluster" if stratified and per_cluster_rows
        else "stratified_fixed_budget" if stratified else "uniform")
    result["dbcv_sampled_rows"] = int(sample.size)
    # Comparability is only claimable when the per-cluster resolution matched. Record it so
    # two cells' scores can be checked against each other rather than assumed comparable.
    if sample.size:
        drawn = np.bincount(labels[sample][labels[sample] >= 0])
        drawn = drawn[drawn > 0]
        result["dbcv_points_per_cluster_median"] = int(np.median(drawn))
        result["dbcv_points_per_cluster_min"] = int(drawn.min())
    if stratified and result.get("per_cluster_dbcv"):
        result.update(_reweight(result, labels))
    return result


def _reweight(result: dict, labels: np.ndarray) -> dict:
    """Recover the size-weighted aggregate from per-cluster scores and TRUE cluster sizes."""
    scores = np.asarray(result["per_cluster_dbcv"], dtype=np.float64)
    ids = np.asarray(result["scored_cluster_ids"])
    true_sizes = np.bincount(labels[labels >= 0], minlength=int(labels.max()) + 1)[ids]

    finite = np.isfinite(scores)
    if not finite.any() or true_sizes[finite].sum() == 0:
        return {}
    scores, true_sizes = scores[finite], true_sizes[finite].astype(np.float64)
    weights = true_sizes / true_sizes.sum()

    negative = scores < 0
    return {
        "dbcv_sample_weighted": result.get("dbcv"),
        "dbcv": float((weights * scores).sum()),
        "dbcv_unweighted_mean": float(scores.mean()),
        "dbcv_cluster_min": float(scores.min()),
        "dbcv_cluster_p10": float(np.percentile(scores, 10)),
        "dbcv_cluster_median": float(np.median(scores)),
        "dbcv_cluster_max": float(scores.max()),
        # A cluster scoring below 0 is not density-separated from its neighbours. The share
        # of POINTS in such clusters is the honest headline: a hundred bad micro-clusters
        # matter less than one bad cluster holding a third of the data.
        "dbcv_frac_clusters_negative": float(negative.mean()),
        "dbcv_point_share_negative": float(weights[negative].sum()),
    }


def score_kdbcv(space: np.ndarray, labels: np.ndarray, sample: np.ndarray) -> dict:
    """DBCV via k-DBCV, which uses a KD-tree for the intercluster distances.

    Same interface and same dict shape as ``rescore_dbcv.score_cell`` so the two are
    interchangeable, and measured to agree with it to within 0.001 (k=200: hdbscan +0.8644 vs
    k-DBCV +0.8635). It is dramatically faster where it matters, because the cost that hurts
    us is the O(k^2) loop over cluster PAIRS and that is exactly what the KD-tree removes:
    4x at k=50, 15x at k=200, **42x at k=400**. At k=1,330 with 400 points per cluster --
    532,000 points -- it returns in 68 s, which is what makes an accurate score affordable on
    the real grid at all.

    Kaufman Lab, Columbia: https://github.com/Kaufman-Lab-Columbia/k-DBCV

    Two sharp edges handled here. ``DBCV_score`` returns ``(aggregate, per_cluster)`` always,
    and it signals "not enough clusters" / "all noise" by returning an aggregate of **-1** --
    which is also a perfectly legal DBCV value. So those two conditions are checked here
    BEFORE calling, and a -1 that comes back afterwards is reported as a real score rather
    than silently reinterpreted as a failure.
    """
    from kDBCV import DBCV_score

    X = np.asarray(space[sample], dtype=np.float64)
    y = np.asarray(labels[sample])
    clustered = y != -1
    noise_fraction = round(float(1.0 - clustered.mean()), 4)
    X, y = X[clustered], y[clustered]
    if y.size == 0:
        return {"dbcv": None, "dbcv_note": "every sampled row was noise"}

    unique, inverse = np.unique(y, return_inverse=True)
    counts = np.bincount(inverse, minlength=unique.size)
    keep = counts >= MIN_PER_CLUSTER
    dropped = int(counts[~keep].sum())
    kept_rows = keep[inverse]
    X = X[kept_rows]
    kept_ids, dense = np.unique(inverse[kept_rows], return_inverse=True)

    detail = {
        "scored_rows": int(dense.size),
        "scored_clusters": int(keep.sum()),
        "dropped_clusters": int((~keep).sum()),
        "dropped_point_fraction": round(dropped / max(1, int(counts.sum())), 4),
        "noise_fraction_in_sample": noise_fraction,
    }
    # Pre-empt the -1 sentinel so it can never be confused with a genuine score.
    if detail["scored_clusters"] < 2:
        return {"dbcv": None,
                "dbcv_note": f"only {detail['scored_clusters']} cluster had "
                             f"{MIN_PER_CLUSTER}+ sampled points", **detail}

    # k-DBCV also returns the -1 sentinel when its memory estimate exceeds mem_cutoff. That
    # cannot be pre-empted by counting clusters, so predict the same quantity it does:
    # the intracluster step materialises an n_i x n_i distance matrix for the LARGEST
    # cluster, and refuses at ((max^2 * 8) / 1024^3) * 8 GB.
    largest = int(np.bincount(dense).max())
    predicted_gb = ((largest ** 2) * 8) / 1024 ** 3 * 8
    detail["dbcv_largest_cluster"] = largest
    detail["dbcv_predicted_gb"] = round(predicted_gb, 3)
    if predicted_gb > KDBCV_MEMORY_CUTOFF_GB:
        return {"dbcv": None,
                "dbcv_note": f"largest sampled cluster {largest:,} points needs "
                             f"~{predicted_gb:.1f} GB, over the {KDBCV_MEMORY_CUTOFF_GB} GB "
                             f"cutoff; lower per_cluster_rows", **detail}

    aggregate, per_cluster = DBCV_score(X, dense.astype(np.int32), ind_clust_scores=True,
                                        mem_cutoff=KDBCV_MEMORY_CUTOFF_GB)
    # On any refusal it returns (-1, -1) or (-1, None) -- and -1 is a legal DBCV value, so
    # the aggregate alone cannot be trusted. With ind_clust_scores=True a real result is an
    # ARRAY of per-cluster scores; a scalar or None means the sentinel.
    scores = None if per_cluster is None else np.asarray(per_cluster).ravel()
    if scores is None or scores.ndim == 0 or scores.size != detail["scored_clusters"]:
        return {"dbcv": None,
                "dbcv_note": "k-DBCV returned its -1 sentinel (declined to score)", **detail}

    aggregate = float(aggregate)
    detail["per_cluster_dbcv"] = [float(v) for v in scores]
    detail["scored_cluster_ids"] = [int(v) for v in unique[keep]]
    return {"dbcv": None if not np.isfinite(aggregate) else aggregate, **detail}


def _to_host(array):
    for attribute in ("to_numpy", "get"):
        method = getattr(array, attribute, None)
        if callable(method):
            return np.asarray(method())
    return np.asarray(array)


def summarise(space: np.ndarray, labels: np.ndarray, probabilities: np.ndarray | None = None,
              clusterer=None, *, connectivity_rows: int = DEFAULT_CONNECTIVITY_ROWS,
              dbcv_rows: int = DEFAULT_DBCV_ROWS, dbcv_max_clusters: int | None = None,
              dbcv_stratified: bool = True,
              dbcv_per_cluster: int | None = DEFAULT_DBCV_PER_CLUSTER,
              dbcv_backend: str = "auto", seed: int = 0) -> dict:
    """Every geometry metric for one clustering. No ranking, no combination.

    ``dbcv_max_clusters`` defaults to None (no ceiling) now that stratified sampling keeps
    every cluster scoreable. Pass an int to restore the old refusal.

    ``dbcv_per_cluster`` fixes the points drawn from every cluster, which is what makes the
    score comparable between models. Set to None for a fixed total ``dbcv_rows`` budget.
    """
    record: dict = {
        "n_clusters": int(labels.max()) + 1 if (labels >= 0).any() else 0,
        "noise_fraction": float((labels < 0).mean()),
    }
    if probabilities is not None:
        record.update(probability_stats(probabilities, labels))
    if clusterer is not None:
        record.update(persistence_stats(clusterer))
        record.update(relative_validity(clusterer))
    record.update(connectivity(space, labels, max_rows=connectivity_rows, seed=seed))
    record.update(dbcv(space, labels, max_rows=dbcv_rows, max_clusters=dbcv_max_clusters,
                       stratified=dbcv_stratified, per_cluster_rows=dbcv_per_cluster,
                       backend=dbcv_backend, seed=seed))
    return record
