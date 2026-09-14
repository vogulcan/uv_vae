#!/usr/bin/env python
"""Exact `approximate_predict`, with the brute-force kNN swapped for an indexed one.

Why this exists
---------------
The scaling sweep measured cuML's ``approximate_predict`` at 4.35 s per M-probe-rows per
M-fit-rows -- dead linear in the fit-set size across a 50x range (500K -> 25M). That constant
is the signature of a brute-force neighbour search: every query point compared against every
fit point. At 25M fit rows, labelling the 157.5M-row cohort costs ~4 hours, 87% of a cell.

But only ONE step of approximate_predict scales with the fit set, and it is not intrinsic to
HDBSCAN. From the reference implementation (hdbscan/prediction.py) the algorithm is:

    1. query the 2*min_samples nearest fit points          <-- the only O(N_fit) step
    2. mr = max(core_dist(neighbour), core_dist_est(query), dist) for each neighbour, where
       core_dist_est(query) is the (min_samples)-th raw neighbour distance
    3. pick the neighbour minimising mr; lambda = 1 / mr
    4. walk up the condensed tree from that neighbour's leaf while lambda_val >= lambda
    5. label = cluster_map[cluster] (else -1); prob = min(max_lambda, lambda) / max_lambda

Steps 2-5 are O(tree depth) and independent of N_fit. Replacing step 1 with an indexed
neighbour search -- leaving 2-5 byte-identical -- reproduces the output EXACTLY while
removing the term that scales.

cuML already ships the index: ``NearestNeighbors(algorithm='rbc')``, Cayton's Random Ball
Cover. It partitions the space and prunes with the triangle inequality, answering queries in
O(sqrt n) after an O(n sqrt n) build, and it is EXACT -- not an approximation. It supports
euclidean/l2 (only haversine carries the <=3-dimension restriction) and the embedding here is
2-D, the regime it was designed for. At 25M fit rows sqrt(n) ~ 5,000 versus n = 25M.

Exactness is a claim to check, not assert. ``--validate`` re-labels the saved eval probe and
compares against the ``approximate_predict`` labels the sweep already wrote, so agreement is
measured against real ground truth before anything depends on it.

    # 1. does the fitted model expose what this needs? (seconds -- run this first)
    python umap_hdbscan_sweep/fast_predict.py --inspect --model models/fit25000000.joblib

    # 2. exactness against the saved eval-probe labels
    python umap_hdbscan_sweep/fast_predict.py --validate \
        --model models/fit25000000.joblib --coords coords.npy \
        --fit-indices labels/fit25000000_indices.npy \
        --probe-indices eval_probe_indices.npy \
        --reference labels/fit25000000_evalprobe_labels.npy

    # 3. label the full cohort
    python umap_hdbscan_sweep/fast_predict.py --apply \
        --model models/fit25000000.joblib --coords coords.npy \
        --fit-indices labels/fit25000000_indices.npy --out cohort_labels.npy

The tree-descent core is pure numpy and is tested against hdbscan's own ``approximate_predict``
in tests/test_fast_predict.py -- that part needs no GPU.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

DBL_MAX = np.finfo(np.float64).max


def log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}", flush=True)


# ── the tables the descent needs, densified into arrays ────────────────────────

@dataclass
class PredictionTables:
    """Everything from the fitted model that steps 2-5 touch, as flat arrays.

    The reference implementation carries dicts and re-scans the raw tree per point. Indexing
    by cluster id into arrays instead is what lets the whole descent run vectorised over
    millions of query points at once.
    """

    core_distances: np.ndarray      # (n_fit,)   core distance of every fit point
    point_parent: np.ndarray        # (n_fit,)   condensed-tree cluster each point falls in
    point_lambda: np.ndarray        # (n_fit,)   lambda_val of that point's tree row
    parent_of: np.ndarray           # (n_clusters+1,) cluster -> parent cluster
    lambda_of: np.ndarray           # (n_clusters+1,) cluster -> lambda_val (-inf if absent)
    cluster_label: np.ndarray       # (n_clusters+1,) cluster -> final label, -1 if unselected
    max_lambda: np.ndarray          # (n_clusters+1,) cluster -> max lambda in that cluster
    tree_root: int
    min_samples: int
    n_fit: int


def resolve_min_samples(clusterer, min_samples: int | None = None) -> int:
    """The min_samples the tables will use, without building them.

    Exposed because core distances can be recomputed instead of read off the model, and the
    recompute needs this value -- but it would otherwise only be available *after*
    build_tables, which is exactly the call that fails when they are missing.
    """
    return int(min_samples or getattr(clusterer, "min_samples", None)
               or clusterer.min_cluster_size)


def build_tables(clusterer, n_fit: int, min_samples: int | None = None,
                 core_distances: np.ndarray | None = None) -> PredictionTables:
    """Densify a fitted HDBSCAN (cuML or CPU) into PredictionTables.

    Raises with a specific message when an attribute is missing rather than failing deep in
    the descent, because "which internals does this cuML build expose" is the one thing that
    cannot be settled off the GPU box.

    ``core_distances`` overrides what the model carries. Supply it for a model that never
    stored them -- see ``recompute_core_distances``.
    """
    raw_tree = _raw_condensed_tree(clusterer)
    min_samples = resolve_min_samples(clusterer, min_samples)

    if core_distances is None:
        core_distances = _core_distances(clusterer, n_fit)
    else:
        core_distances = np.asarray(core_distances, dtype=np.float64).ravel()
    if core_distances.shape[0] != n_fit:
        raise ValueError(
            f"core_distances has {core_distances.shape[0]} entries but the fit set has "
            f"{n_fit} rows -- the --fit-indices file does not match this model"
        )

    parents = raw_tree["parent"].astype(np.int64)
    children = raw_tree["child"].astype(np.int64)
    lambdas = raw_tree["lambda_val"].astype(np.float64)
    child_sizes = raw_tree["child_size"].astype(np.int64)

    # Rows whose child is an individual point (child_size == 1) give each point its leaf
    # cluster; rows whose child is itself a cluster form the cluster tree walked in step 4.
    is_point = children < n_fit
    point_parent = np.zeros(n_fit, dtype=np.int64)
    point_lambda = np.zeros(n_fit, dtype=np.float64)
    point_parent[children[is_point]] = parents[is_point]
    point_lambda[children[is_point]] = lambdas[is_point]

    is_cluster_row = child_sizes > 1
    max_cluster = int(max(parents.max(), children.max())) + 1

    parent_of = np.arange(max_cluster, dtype=np.int64)
    lambda_of = np.full(max_cluster, -np.inf, dtype=np.float64)
    parent_of[children[is_cluster_row]] = parents[is_cluster_row]
    lambda_of[children[is_cluster_row]] = lambdas[is_cluster_row]

    # Each cluster's own max lambda = the largest lambda_val among rows having it as parent,
    # which is exactly what PredictionData.__init__ computes. Non-finite lambdas are skipped:
    # lambda = 1/distance and cuML computes it in float32, so coincident points -- unavoidable
    # in a 25M-row 2-D embedding -- overflow, and one of them would set a whole cluster's scale
    # to inf and drive every probability in it to zero.
    own_max_lambda = np.zeros(max_cluster, dtype=np.float64)
    np.maximum.at(own_max_lambda, parents, np.where(np.isfinite(lambdas), lambdas, 0.0))

    cluster_label = _cluster_labels(clusterer, raw_tree, max_cluster, parent_of,
                                    is_cluster_row, parents, children)

    # A cluster below a selected one inherits the SELECTED ancestor's max lambda, not its own
    # (PredictionData.__init__: `self.max_lambdas[sub_cluster] = self.max_lambdas[cluster]`);
    # walking to the topmost ancestor carrying the same label is that mapping without the
    # recursion.
    #
    # Derived from the tree every time rather than read from prediction_data_.max_lambdas.
    # That dict is a cache of this same derivation, and a joblib-loaded cuML model regenerates
    # it full of FLT_MAX, which leaves labels correct but makes prob = lambda/3.4e38 ~ 0
    # everywhere. Deriving is what PredictionData itself does, so this IS the reference path.
    max_lambda = own_max_lambda[_selected_ancestor(cluster_label, parent_of)]

    return PredictionTables(
        core_distances=core_distances.astype(np.float64),
        point_parent=point_parent,
        point_lambda=point_lambda,
        parent_of=parent_of,
        lambda_of=lambda_of,
        cluster_label=cluster_label,
        max_lambda=max_lambda,
        tree_root=int(parents[is_cluster_row].min()) if is_cluster_row.any()
        else int(parents.min()),
        min_samples=min_samples,
        n_fit=n_fit,
    )


def _selected_ancestor(cluster_label: np.ndarray, parent_of: np.ndarray) -> np.ndarray:
    """For each cluster, the topmost ancestor carrying the same label -- i.e. the selected
    cluster it descends from. Unlabelled clusters map to themselves (their max lambda is
    never read, since probability is forced to 0 wherever the label is -1).

    Pointer-jumping instead of recursion: one pass per tree level, over all clusters at once.
    """
    top = np.arange(cluster_label.shape[0], dtype=np.int64)
    for _ in range(64):
        parent = parent_of[top]
        can_rise = (cluster_label[top] >= 0) & (parent != top) & (
            cluster_label[parent] == cluster_label[top])
        if not can_rise.any():
            break
        top[can_rise] = parent[can_rise]
    return top


def _raw_condensed_tree(clusterer) -> np.ndarray:
    tree = getattr(clusterer, "condensed_tree_", None)
    if tree is None:
        raise AttributeError(
            "fitted model has no `condensed_tree_`. Without the condensed tree the lambda "
            "descent (steps 4-5) cannot be reproduced and only approximate label transfer "
            "is possible. Run with --inspect to see what this model does expose."
        )
    for attribute in ("_raw_tree", "to_numpy"):
        value = getattr(tree, attribute, None)
        if value is None:
            continue
        return value() if callable(value) else value
    if isinstance(tree, np.ndarray):
        return tree
    try:  # cuML returns a cudf DataFrame from to_pandas()/to_numpy() in some versions
        return tree.to_pandas().to_records(index=False)
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        raise AttributeError(
            f"could not get a raw condensed tree out of {type(tree)!r}: {exc}"
        ) from exc


def _core_distances(clusterer, n_fit: int) -> np.ndarray:
    prediction_data = getattr(clusterer, "prediction_data_", None)
    for holder, attribute in ((prediction_data, "core_distances"),
                              (clusterer, "core_distances_"),
                              (clusterer, "core_distances")):
        if holder is None:
            continue
        value = getattr(holder, attribute, None)
        if value is not None:
            return _to_host(value).ravel()
    raise AttributeError(
        "fitted model exposes no core distances (looked at prediction_data_.core_distances, "
        "core_distances_, core_distances).\n"
        "  Most likely cause: the model was constructed without prediction_data=True. Both "
        "cuML and the CPU hdbscan package populate core distances only when asked for them "
        "at construction, so refitting with prediction_data=True is the real fix and is what "
        "hdbscan_param_sweep._fit_hdbscan already does.\n"
        "  If refitting is not an option, they can be recomputed as the min_samples-th "
        "nearest-neighbour distance inside the fit set: pass --recompute-core-distances to "
        "this script's CLI, or call fast_predict.recompute_core_distances(fit_coords, "
        "min_samples) and hand the result to build_tables(core_distances=...)."
    )


def _cluster_labels(clusterer, raw_tree, max_cluster, parent_of,
                    is_cluster_row, parents, children) -> np.ndarray:
    """cluster id -> final label, with selected clusters' descendants inheriting the label.

    The reference builds this by walking `_clusters_below` per selected cluster; here the
    selected set is taken from the model and descendants inherit by iterated pointer-jumping,
    which is the same mapping without the recursion.
    """
    labels = np.full(max_cluster, -1, dtype=np.int64)

    prediction_data = getattr(clusterer, "prediction_data_", None)
    cluster_map = getattr(prediction_data, "cluster_map", None) if prediction_data else None
    if cluster_map:
        for cluster, label in cluster_map.items():
            labels[int(cluster)] = int(label)
        return labels

    # No prediction_data_ (the cuML case): recover the selected clusters from the fit labels.
    # Each fit point's leaf cluster maps to that point's final label; propagating up gives
    # every selected cluster and its descendants.
    fit_labels = _to_host(clusterer.labels_).ravel()
    is_point = children < len(fit_labels)
    point_leaf = np.zeros(len(fit_labels), dtype=np.int64)
    point_leaf[children[is_point]] = parents[is_point]

    clustered = fit_labels >= 0
    labels[point_leaf[clustered]] = fit_labels[clustered]

    # Push each labelled cluster's label up to ancestors that carry the same label, so a
    # query landing on an ancestor resolves the same way the reference's cluster_map does.
    for _ in range(64):
        unlabelled = np.nonzero(labels < 0)[0]
        if unlabelled.size == 0:
            break
        inherited = labels[parent_of[unlabelled]]
        if not (inherited >= 0).any():
            break
        labels[unlabelled] = inherited
    return labels


def _to_host(array) -> np.ndarray:
    for attribute in ("to_numpy", "get"):
        method = getattr(array, attribute, None)
        if callable(method):
            return np.asarray(method())
    return np.asarray(array)


# ── neighbour search backends ──────────────────────────────────────────────────

def knn_query(fit_coords: np.ndarray, query_coords: np.ndarray, k: int,
              backend: str = "rbc", batch_rows: int = 5_000_000,
              index=None) -> tuple[np.ndarray, np.ndarray]:
    """k nearest fit points for each query point. All backends here are EXACT.

    'rbc'     cuML Random Ball Cover -- O(sqrt n) queries, exact, 2-D euclidean is its
              intended regime. This is the one that removes the O(N_fit) term.
    'brute'   cuML brute force -- what approximate_predict does internally. Reference.
    'sklearn' CPU KDTree -- what the CPU hdbscan reference uses. For local testing.
    """
    if backend == "sklearn":
        from sklearn.neighbors import KDTree
        tree = index if index is not None else KDTree(fit_coords)
        return tree.query(query_coords, k=k)

    from cuml.neighbors import NearestNeighbors
    if index is None:
        index = NearestNeighbors(n_neighbors=k, algorithm=backend, metric="euclidean")
        index.fit(fit_coords)

    distance_chunks, index_chunks = [], []
    for start in range(0, query_coords.shape[0], batch_rows):
        chunk = query_coords[start:start + batch_rows]
        distances, indices = index.kneighbors(chunk, n_neighbors=k)
        distance_chunks.append(_to_host(distances).astype(np.float64))
        index_chunks.append(_to_host(indices).astype(np.int64))
    return np.concatenate(distance_chunks), np.concatenate(index_chunks)


def recompute_core_distances(fit_coords: np.ndarray, min_samples: int,
                             backend: str = "rbc", batch_rows: int = 5_000_000,
                             index=None) -> np.ndarray:
    """Core distances derived from the fit set, for a model that never stored them.

    ``k = min_samples`` neighbours, taking the last column. The self-match sits at position 0
    at distance 0, so this is the ``(min_samples - 1)``-th *true* neighbour -- which looks
    like an off-by-one and is not: hdbscan counts the point itself among its min_samples.
    Verified against ``prediction_data_.core_distances`` on a 1,200-point three-blob fit at
    min_samples=5, where ``k = min_samples`` reproduces the model's own array exactly
    (max |diff| = 0.0) and ``k = min_samples + 1`` is wrong by up to 0.44.

    This is a genuine substitute, not an approximation: it is the same quantity the fit
    computes. It is only separate because the fit does not always keep it.
    """
    distances, _ = knn_query(fit_coords, fit_coords, min_samples,
                             backend=backend, batch_rows=batch_rows, index=index)
    return distances[:, -1].astype(np.float64)


def build_index(fit_coords: np.ndarray, k: int, backend: str):
    if backend == "sklearn":
        from sklearn.neighbors import KDTree
        return KDTree(fit_coords)
    from cuml.neighbors import NearestNeighbors
    index = NearestNeighbors(n_neighbors=k, algorithm=backend, metric="euclidean")
    index.fit(fit_coords)
    return index


# ── steps 2-5, vectorised ──────────────────────────────────────────────────────

def predict_from_neighbours(tables: PredictionTables, neighbour_distances: np.ndarray,
                            neighbour_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Steps 2-5 of approximate_predict, over all query points at once.

    Mirrors hdbscan's `_find_neighbor_and_lambda` + `_find_cluster_and_probability` exactly,
    including the detail that the query point's own core distance is estimated as the
    (min_samples)-th raw neighbour distance -- index min_samples into the 2*min_samples
    neighbours, not the last one.
    """
    min_samples = tables.min_samples
    n_query = neighbour_indices.shape[0]

    # step 2: mutual reachability to each candidate neighbour
    point_core = neighbour_distances[:, min_samples][:, None]
    mr = np.maximum(tables.core_distances[neighbour_indices], point_core)
    np.maximum(mr, neighbour_distances, out=mr)

    # step 3: the neighbour minimising mutual reachability (NOT raw distance)
    best = mr.argmin(axis=1)
    rows = np.arange(n_query)
    mr_min = mr[rows, best]
    nearest = neighbour_indices[rows, best]
    lambda_ = np.where(mr_min > 0.0, 1.0 / np.where(mr_min > 0.0, mr_min, 1.0), DBL_MAX)

    # step 4: walk up the cluster tree while the edge's lambda_val >= the point's lambda.
    # The reference loops per point; here every point that can still step takes one step per
    # iteration, so the loop runs once per tree level rather than once per point.
    potential = tables.point_parent[nearest].copy()
    walking = np.nonzero(tables.point_lambda[nearest] > lambda_)[0]
    while walking.size:
        current = potential[walking]
        can_step = (current > tables.tree_root) & (tables.lambda_of[current] >= lambda_[walking])
        walking = walking[can_step]
        if walking.size == 0:
            break
        potential[walking] = tables.parent_of[potential[walking]]

    # step 5
    labels = tables.cluster_label[potential]
    max_lambda = tables.max_lambda[potential]
    clamped = np.minimum(max_lambda, lambda_)
    with np.errstate(divide="ignore", invalid="ignore"):
        probabilities = np.where(max_lambda > 0.0, clamped / max_lambda, 1.0)
    probabilities = np.where(labels >= 0, probabilities, 0.0)
    return labels.astype(np.int32), probabilities.astype(np.float32)


def predict(tables: PredictionTables, fit_coords: np.ndarray, query_coords: np.ndarray,
            backend: str = "rbc", batch_rows: int = 5_000_000,
            index=None, progress_every: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Label query points, batching the WHOLE pipeline rather than only the kNN.

    The neighbour arrays are the memory driver, not the coordinates: k = 2*min_samples
    neighbours per point at 8 bytes each, times distances + indices, times another copy for
    the mutual-reachability matrix. At min_samples=5 that is ~240 bytes per query row, so
    labelling the 132.5M held rows in one shot would want ~32 GB of host RAM before anything
    is written. Batching end-to-end holds it at batch_rows * 240 bytes (~1.2 GB at 5M) and
    keeps only the int32/float32 outputs, which are 8 bytes per row.
    """
    k = 2 * tables.min_samples
    if index is None:
        index = build_index(fit_coords, k, backend)

    n_query = query_coords.shape[0]
    label_chunks, probability_chunks = [], []
    for start in range(0, n_query, batch_rows):
        chunk = query_coords[start:start + batch_rows]
        distances, indices = knn_query(fit_coords, chunk, k, backend=backend,
                                       batch_rows=batch_rows, index=index)
        chunk_labels, chunk_probabilities = predict_from_neighbours(tables, distances, indices)
        label_chunks.append(chunk_labels)
        probability_chunks.append(chunk_probabilities)
        del distances, indices
        if progress_every and (start // batch_rows) % progress_every == 0:
            log(f"    {min(start + batch_rows, n_query):,} / {n_query:,} rows")
    return np.concatenate(label_chunks), np.concatenate(probability_chunks)


# ── CLI ────────────────────────────────────────────────────────────────────────

def load_model(path: Path):
    import joblib
    return joblib.load(path)


def do_inspect(model) -> None:
    """Report which internals this build exposes. Run before anything long.

    Note: accessing `prediction_data_` on a cuML model can itself be slow the first time --
    if prediction data did not survive a joblib round-trip (its C++/GPU state is not always
    picklable across cuML versions), the attribute access silently regenerates it from
    scratch over the whole fit set. That is real GPU work, not a hang; watch `nvidia-smi`.
    """
    print(f"model type: {type(model)}")
    interesting = ["labels_", "probabilities_", "condensed_tree_", "prediction_data_",
                   "core_distances_", "core_distances", "min_samples", "min_cluster_size",
                   "single_linkage_tree_", "minimum_spanning_tree_"]
    for name in interesting:
        value = getattr(model, name, None)
        if value is None:
            print(f"  {name:26} MISSING")
            continue
        shape = getattr(value, "shape", None)
        print(f"  {name:26} {type(value).__name__}"
              + (f" shape={shape}" if shape is not None else ""))

    tree = getattr(model, "condensed_tree_", None)
    if tree is not None:
        print(f"\n  condensed_tree_ attributes: "
              f"{[a for a in dir(tree) if not a.startswith('__')][:12]}")
    prediction_data = getattr(model, "prediction_data_", None)
    if prediction_data is not None:
        print(f"  prediction_data_ attributes: "
              f"{[a for a in dir(prediction_data) if not a.startswith('__')][:12]}")
        # This is the actual first place _core_distances() looks -- if it is here, the
        # top-level core_distances_/core_distances MISSING above does not matter.
        nested_core = getattr(prediction_data, "core_distances", None)
        nested_map = getattr(prediction_data, "cluster_map", None)
        if nested_core is not None:
            shape = getattr(nested_core, "shape", None)
            print(f"  prediction_data_.core_distances   present"
                  + (f" shape={shape}" if shape is not None else ""))
        if nested_map is not None:
            print(f"  prediction_data_.cluster_map      present ({len(nested_map)} entries)")

    have_core = (prediction_data is not None
                and getattr(prediction_data, "core_distances", None) is not None) \
        or getattr(model, "core_distances_", None) is not None \
        or getattr(model, "core_distances", None) is not None
    have_tree = tree is not None
    print(f"\nWhat this script needs: condensed_tree_ (raw) [{'ok' if have_tree else 'MISSING'}], "
          f"core distances [{'ok' if have_core else 'MISSING'}], labels_.")
    if not have_core:
        print("Core distances missing everywhere. The model was most likely fit without "
              "prediction_data=True -- refitting with it is the real fix. Failing that, "
              "pass --recompute-core-distances to derive them from the fit set (they are "
              "just the min_samples-th NN distance inside it).")


def do_diagnose(tables: PredictionTables, fit_coords, query_coords, args) -> None:
    """Everything worth knowing, from one model load.

    Splits the search space in one shot. 'brute' runs the SAME steps 2-5 over different
    neighbours, so if rbc and brute agree with each other but both differ from cuML, the
    neighbour search is exonerated and the divergence is in steps 2-5 (or in cuML's own
    algorithm differing from the CPU reference this reproduces). If they differ from each
    other, the index is at fault.
    """
    mapped = tables.cluster_label >= 0
    print("\n-- prediction tables ------------------------------------------")
    print(f"  mapped tree clusters  : {int(mapped.sum()):,}")
    print(f"  distinct labels       : {int(np.unique(tables.cluster_label[mapped]).size):,}")
    if mapped.any():
        values = tables.max_lambda[mapped]
        print(f"  max_lambda > 0        : {float((values > 0).mean()) * 100:.2f}% of mapped")
        positive = values[values > 0]
        if positive.size:
            print(f"  max_lambda percentiles: "
                  + ", ".join(f"p{p}={np.percentile(positive, p):.4g}"
                              for p in (1, 25, 50, 75, 99)))
    print(f"  core_distances        : min={tables.core_distances.min():.4g} "
          f"median={np.median(tables.core_distances):.4g} "
          f"max={tables.core_distances.max():.4g} "
          f"zeros={int((tables.core_distances == 0).sum()):,}")
    print(f"  tree_root             : {tables.tree_root:,}  (n_fit={tables.n_fit:,})")

    results = {}
    for backend in ("rbc", "brute"):
        try:
            started = perf_counter()
            index = build_index(fit_coords, 2 * tables.min_samples, backend)
            build_seconds = perf_counter() - started
            started = perf_counter()
            labels, probabilities = predict(tables, fit_coords, query_coords,
                                            backend=backend, batch_rows=args.batch_rows,
                                            index=index)
            elapsed = perf_counter() - started
            rate = elapsed / (query_coords.shape[0] / 1e6)
            results[backend] = (labels, probabilities)
            print(f"\n  {backend:6} index {build_seconds:6.1f}s  label {elapsed:7.1f}s  "
                  f"{rate:7.2f} s/M -> {rate * 157.5 / 60:.1f} min for 157.5M")
        except Exception as exc:  # noqa: BLE001 - a failing backend is itself the finding
            print(f"\n  {backend:6} FAILED: {type(exc).__name__}: {str(exc)[:300]}")

    if len(results) == 2:
        rbc_labels, rbc_probabilities = results["rbc"]
        brute_labels, brute_probabilities = results["brute"]
        same = rbc_labels == brute_labels
        print(f"\n  rbc vs brute labels   : {same.mean() * 100:.4f}% identical "
              f"({int((~same).sum()):,} differ)")
        print(f"  rbc vs brute max|dp|  : "
              f"{np.abs(rbc_probabilities - brute_probabilities).max():.3e}")
        if same.all():
            print("  -> the index is EXONERATED: RBC returns neighbours giving identical "
                  "output to brute force.")
            print("     Any gap to cuML is in steps 2-5, or cuML's algorithm differs from "
                  "the CPU reference.")
        else:
            print("  -> RBC and brute disagree: the neighbour search is the suspect.")

    if not args.reference:
        return
    reference = np.load(args.reference)
    reference_probabilities = (np.load(args.reference_probabilities)
                               if args.reference_probabilities else None)
    for backend, (labels, probabilities) in results.items():
        agree = labels == reference
        print(f"\n-- {backend} vs cuML reference ----------------------------------")
        print(f"  exact label agreement : {agree.mean() * 100:.4f}%")
        mine_noise, ref_noise = labels < 0, reference < 0
        print(f"  mine clustered/ref -1 : {float((~mine_noise & ref_noise).mean()) * 100:.4f}%")
        print(f"  mine -1/ref clustered : {float((mine_noise & ~ref_noise).mean()) * 100:.4f}%")
        both = ~mine_noise & ~ref_noise
        if both.any():
            disagree_both = float((labels[both] != reference[both]).mean())
            print(f"  both clustered, differ: {disagree_both * 100:.4f}% "
                  "(cluster identity, not the noise gate)")
        if reference_probabilities is not None:
            delta = np.abs(probabilities - reference_probabilities)
            print(f"  max|dp| {delta.max():.3e}   mean|dp| {delta.mean():.3e}")
            print(f"  mine  prob mean {probabilities.mean():.4f}  "
                  f"frac==1.0 {float((probabilities == 1.0).mean()):.4f}  "
                  f"frac==0.0 {float((probabilities == 0.0).mean()):.4f}")
            print(f"  cuML  prob mean {reference_probabilities.mean():.4f}  "
                  f"frac==1.0 {float((reference_probabilities == 1.0).mean()):.4f}  "
                  f"frac==0.0 {float((reference_probabilities == 0.0).mean()):.4f}")


def do_determinism_check(tables: PredictionTables, fit_coords: np.ndarray,
                         query_coords: np.ndarray, n_runs: int,
                         batch_rows: int) -> None:
    """Build the RBC index and label the probe N times; compare hashes across runs.

    RBC selects its representatives at random; cuML exposes no random_state for
    NearestNeighbors. This measures whether the implementation happens to be
    deterministic in practice (fixed seed internally, or deterministic GPU kernel).

    The hash printed at the end can be compared across separate invocations:
        python fast_predict.py --determinism-check 3 ... | tail -1
    Run it twice in different terminals; if the hashes match, the index is
    reproducible across processes too.
    """
    import hashlib

    k = 2 * tables.min_samples
    hashes = []
    log(f"determinism check: {n_runs} runs over {query_coords.shape[0]:,} query rows")
    for i in range(n_runs):
        t0 = perf_counter()
        index = build_index(fit_coords, k, "rbc")
        build_s = perf_counter() - t0
        t0 = perf_counter()
        labels, _ = predict(tables, fit_coords, query_coords,
                            backend="rbc", batch_rows=batch_rows, index=index)
        label_s = perf_counter() - t0
        h = hashlib.md5(labels.tobytes()).hexdigest()
        hashes.append(h)
        clustered_pct = float((labels >= 0).mean()) * 100
        log(f"  run {i + 1}/{n_runs}: hash={h}  build={build_s:.1f}s  "
            f"label={label_s:.1f}s  clustered={clustered_pct:.2f}%")

    distinct = len(set(hashes))
    if distinct == 1:
        log(f"DETERMINISTIC within this process: all {n_runs} runs produced hash={hashes[0]}")
        log(f"PROCESS hash={hashes[0]}")
    else:
        log(f"NON-DETERMINISTIC: {distinct} distinct hashes across {n_runs} runs")
        for i, h in enumerate(hashes):
            log(f"  run {i + 1}: {h}")
        log("PROCESS hash=UNSTABLE")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect", action="store_true", help="report model internals, exit")
    mode.add_argument("--validate", action="store_true",
                      help="relabel the eval probe and compare against saved reference labels")
    mode.add_argument("--apply", action="store_true", help="label rows and write them out")
    mode.add_argument("--determinism-check", type=int, metavar="N", default=0,
                      help="build the index and label the probe N times from scratch, then "
                           "compare. Prints a hash of the labels so runs in SEPARATE "
                           "processes can be compared too -- RBC picks its representatives "
                           "at random and cuML exposes no seed for it")
    mode.add_argument("--diagnose", action="store_true",
                      help="one model load, every question: table health, rbc-vs-brute "
                           "(isolates the neighbour search from steps 2-5), and a breakdown "
                           "of how the labels differ from the reference")

    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--coords", type=Path, help="coords.npy for the whole cohort")
    parser.add_argument("--fit-indices", type=Path, help="rows the model was fit on")
    parser.add_argument("--probe-indices", type=Path, help="rows to label (default: all held)")
    parser.add_argument("--reference", type=Path, help="approximate_predict labels to check")
    parser.add_argument("--reference-probabilities", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--backend", default="rbc", choices=["rbc", "brute", "sklearn"])
    parser.add_argument("--batch-rows", type=int, default=5_000_000)
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--recompute-core-distances", action="store_true",
                        help="derive core distances from the fit set instead of the model")
    args = parser.parse_args()

    log(f"loading {args.model}")
    model = load_model(args.model)

    if args.inspect:
        do_inspect(model)
        return

    if not (args.coords and args.fit_indices):
        raise SystemExit("--coords and --fit-indices are required for --validate/--apply")

    coords = np.load(args.coords, mmap_mode="r")
    fit_idx = np.load(args.fit_indices)
    fit_coords = np.array(coords[fit_idx], dtype=np.float32)
    log(f"fit set {fit_coords.shape[0]:,} x {fit_coords.shape[1]}")

    # First touch of prediction_data_ on a joblib-loaded cuML model regenerates it on the GPU
    # (its C++/GPU state does not always pickle), which on 25M rows is minutes of real work
    # with no output. Say so, otherwise it reads as a hang.
    # Recomputed BEFORE build_tables, not after. build_tables raises when the model carries
    # no core distances, so a flag applied to the tables it returns can only ever patch a
    # model that did not need patching -- for the model the flag exists to rescue, the call
    # that consumes it never returns. This ordering is what makes the suggestion in that
    # error message actually reachable.
    override = None
    if args.recompute_core_distances:
        min_samples = resolve_min_samples(model, args.min_samples)
        log(f"recomputing core distances inside the fit set (min_samples={min_samples}) ...")
        override = recompute_core_distances(fit_coords, min_samples,
                                            backend=args.backend,
                                            batch_rows=args.batch_rows)

    log("building prediction tables (a cuML model regenerates prediction data on first "
        "access after a joblib load -- minutes at 25M rows; nvidia-smi will show it working)")
    tables = build_tables(model, n_fit=fit_coords.shape[0], min_samples=args.min_samples,
                          core_distances=override)
    log(f"tables built: min_samples={tables.min_samples}, "
        f"{int((tables.cluster_label >= 0).sum()):,} mapped clusters, "
        f"tree_root={tables.tree_root}")

    if args.probe_indices:
        query_idx = np.load(args.probe_indices)
    else:
        mask = np.ones(coords.shape[0], dtype=bool)
        mask[fit_idx] = False
        query_idx = np.nonzero(mask)[0]
    query_coords = np.array(coords[query_idx], dtype=np.float32)
    log(f"labelling {query_coords.shape[0]:,} rows via backend={args.backend}")

    if args.diagnose:
        do_diagnose(tables, fit_coords, query_coords, args)
        return

    if args.determinism_check:
        do_determinism_check(tables, fit_coords, query_coords,
                             n_runs=args.determinism_check,
                             batch_rows=args.batch_rows)
        return

    started = perf_counter()
    index = build_index(fit_coords, 2 * tables.min_samples, args.backend)
    build_seconds = perf_counter() - started
    log(f"  index built in {build_seconds:.1f}s")

    started = perf_counter()
    labels, probabilities = predict(tables, fit_coords, query_coords,
                                    backend=args.backend, batch_rows=args.batch_rows,
                                    index=index)
    elapsed = perf_counter() - started
    rate = elapsed / (query_coords.shape[0] / 1e6)
    log(f"  labelled in {elapsed:.1f}s = {rate:.2f} s/M "
        f"-> {rate * 157.5 / 60:.1f} min for the 157.5M cohort")

    if args.validate:
        if not args.reference:
            raise SystemExit("--validate needs --reference")
        reference = np.load(args.reference)
        if reference.shape != labels.shape:
            raise SystemExit(f"reference has {reference.shape} labels, got {labels.shape}")
        agree = labels == reference
        print(f"\nexact label agreement : {agree.mean() * 100:.4f}% "
              f"({int((~agree).sum()):,} of {labels.size:,} differ)")

        both = (labels >= 0) & (reference >= 0)
        print(f"  clustered in both     : {both.mean() * 100:.2f}%")
        print(f"  noise here only       : {float(((labels < 0) & (reference >= 0)).mean()) * 100:.4f}%")
        print(f"  noise reference only  : {float(((labels >= 0) & (reference < 0)).mean()) * 100:.4f}%")
        if (~agree).any():
            from cross_size_ari import adjusted_rand, contingency
            counts, rows, cols = contingency(labels, reference)
            print(f"  ARI vs reference      : {adjusted_rand(counts, rows, cols):.6f}")
            disagreeing = np.nonzero(~agree)[0][:10]
            print(f"  first disagreements   : "
                  f"{[(int(i), int(labels[i]), int(reference[i])) for i in disagreeing]}")
        if args.reference_probabilities:
            reference_probabilities = np.load(args.reference_probabilities)
            delta = np.abs(probabilities - reference_probabilities)
            print(f"  max |prob delta|      : {delta.max():.3e}  "
                  f"(mean {delta.mean():.3e})")
        summary = args.model.parent / f"{args.model.stem}_fastpredict_validation.json"
        summary.write_text(json.dumps({
            "backend": args.backend,
            "rows": int(labels.size),
            "exact_agreement": float(agree.mean()),
            "seconds": round(elapsed, 2),
            "seconds_per_million": round(rate, 3),
            "index_build_seconds": round(build_seconds, 2),
        }, indent=2))
        print(f"\nwrote {summary}")

    if args.apply:
        out = args.out or Path("fast_predict_labels.npy")
        np.save(out, labels)
        np.save(out.with_name(out.stem + "_probabilities.npy"), probabilities)
        np.save(out.with_name(out.stem + "_indices.npy"), query_idx)
        log(f"wrote {out} ({labels.size:,} labels, "
            f"{float((labels < 0).mean()) * 100:.2f}% noise)")


if __name__ == "__main__":
    main()
