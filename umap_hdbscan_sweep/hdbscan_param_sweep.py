#!/usr/bin/env python
"""HDBSCAN parameter sweep: fit, label the full cohort, save the labels.

What this does and does not do
------------------------------
It fits HDBSCAN at each grid point, labels all 157.5M cohort rows through
``fast_predict``'s RBC backend, and writes ``cohort_labels.npy``. That is the expensive,
GPU-bound part and the part that has to be swept.

It deliberately stops there. Signature assignment, the SBS96 count matrix, the cluster-size
tables and the profile plots all already exist in
``uv_vae/scripts/run_variant_cluster_pipeline.py`` -- ``annotate_trinuc_counts``,
``write_cluster_sbs96_matrix``, ``run_sigprofiler_assignment``, ``query_cluster_stats``,
``dominant_signature_columns``. Those are the versions every earlier result in this repo was
produced with, so a second implementation here would be a second answer to compare against,
not a shortcut. Point the pipeline at a saved ``cohort_labels.npy`` when signatures are wanted.

The confound this sweep exists to break
---------------------------------------
``scaling_results.json`` swept fit size with ``min_cluster_size = max(50, N * 1e-4)`` --
proportional. Cluster count fell 1338 -> 442 and noise fell 32.6% -> 5.6% from 500K to 25M,
but mcs rose 50x over the same range, so **nothing in that run separates "more data resolves
coarser structure" from "mcs got 50x bigger"**. Here mcs is an independent axis held fixed
across fit sizes.

Why the labelling is affordable at all
--------------------------------------
``approximate_predict`` is brute force -- 109 s per million probe rows against a 25M fit set,
so ~4.8 h per cell. ``fast_predict.py`` swaps in cuML's Random Ball Cover for the neighbour
search and does the same labelling in 5.3 min, 54x. That is what makes a per-cell full-cohort
labelling a sweep step rather than a shortlist step.

Cost model (measured, from scaling_results.json -- fit time is O(N^1.9) and dominates):

    500K  7 s      1M  24 s      5M  470 s     10M  1,870 s
    15M  4,562 s   25M  11,684 s              50M  OOM at 46,450 s

min_samples=15 at 25M OOMed on a 47 GB card, so the 25M row of the grid is min_samples=5 only.

    # print the grid and its projected cost, run nothing
    python umap_hdbscan_sweep/hdbscan_param_sweep.py --dry-run \
        --coords coords.npy --output-dir <out>

    # the sweep (resumable: finished cells are skipped)
    python umap_hdbscan_sweep/hdbscan_param_sweep.py \
        --coords coords.npy --output-dir <out>

    # summarise finished cells
    python umap_hdbscan_sweep/hdbscan_param_sweep.py --aggregate --output-dir <out>

Every cell writes ``<out>/cells/<label>/metrics.json`` and ``cohort_labels.npy`` the moment it
finishes. A crash loses at most the cell in flight.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

# Walk up to the directory that holds uv_vae/ rather than counting levels: these scripts
# get reorganised into subfolders (umap/, hdbscan/, tests/), and a hard-coded parents[N]
# silently resolves to the wrong root the moment one moves, failing on `import uv_vae`.
REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "uv_vae").is_dir()),
                 Path(__file__).resolve().parents[1])
for candidate in (REPO_ROOT / "uv_vae", REPO_ROOT, REPO_ROOT / "uv_vae" / "scripts",
                  Path(__file__).resolve().parent):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np
import polars as pl


def log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}", flush=True)


# ── the grid ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Cell:
    fit_rows: int
    min_cluster_size: int
    min_samples: int
    cluster_selection_method: str = "eom"
    cluster_selection_epsilon: float = 0.0

    def label(self) -> str:
        name = (f"fit{self.fit_rows}_mcs{self.min_cluster_size}"
                f"_ms{self.min_samples}_{self.cluster_selection_method}")
        if self.cluster_selection_epsilon:
            name += f"_eps{self.cluster_selection_epsilon}"
        return name


# Measured fit seconds from scaling_results.json, for the cost projection. Interpolated in
# log-log between the measured points (the curve is a clean power law, ~O(N^1.9)).
MEASURED_FIT_SECONDS = {
    500_000: 7.2, 1_000_000: 23.6, 5_000_000: 469.9,
    10_000_000: 1869.6, 15_000_000: 4562.1, 25_000_000: 11684.0,
}


def projected_fit_seconds(fit_rows: int, min_samples: int) -> float:
    sizes = np.array(sorted(MEASURED_FIT_SECONDS))
    seconds = np.array([MEASURED_FIT_SECONDS[size] for size in sizes])
    estimate = float(np.exp(np.interp(np.log(fit_rows), np.log(sizes), np.log(seconds))))
    # min_samples enlarges the kNN graph; ms=15 at 25M OOMed, and the ms=15 probe that did
    # run took 11,714 s against 11,684 s at ms=5, so the time penalty is small even though
    # the memory penalty is not.
    return estimate * (1.0 + 0.02 * (min_samples - 5))


def build_grid(fit_sizes: list[int], min_cluster_sizes: list[int], min_samples: list[int],
               methods: list[str], epsilons: list[float],
               max_min_samples_at: dict[int, int]) -> list[Cell]:
    """Grid ordered so that the cheapest cells run first.

    Cheapest-first is deliberate: it front-loads the information. If the 500K and 1M rows
    show mcs doing nothing (which HDBSCAN_SWEEP_PLAN.md Finding 1 predicts), that is known
    within minutes rather than after the 3.25 h 25M fits.
    """
    cells = []
    for fit_rows in fit_sizes:
        ceiling = max_min_samples_at.get(fit_rows)
        for samples in min_samples:
            if ceiling is not None and samples > ceiling:
                continue
            for mcs in min_cluster_sizes:
                for method in methods:
                    for epsilon in epsilons:
                        cells.append(Cell(fit_rows, mcs, samples, method, epsilon))
    return sorted(cells, key=lambda cell: (cell.fit_rows, cell.min_samples,
                                           cell.min_cluster_size))


# ── one cell ───────────────────────────────────────────────────────────────────

def build_sbs96_index(context_path: Path, row_order: list[str], cache_path: Path) -> np.ndarray:
    """Collapse each cohort row's context to one int8: its position in the 96-channel order.

    Done ONCE and cached (157 MB), because the alternative is paying it per cell 28 times.
    The canonicalisation itself is imported from ``stage3_apply_full`` -- existing repo code,
    documented and tested as identical to ``rvcp.annotate_trinuc_counts`` -- rather than
    written a third time.

    Streamed by row group so peak memory is one group, not 157.5M strings.
    """
    import pyarrow.parquet as pq

    from stage3_apply_full import sbs96_expr

    position = {label: index for index, label in enumerate(row_order)}
    parquet_file = pq.ParquetFile(context_path)
    total = parquet_file.metadata.num_rows
    log(f"building SBS96 index over {total:,} rows (one-off, cached)")

    index = np.empty(total, dtype=np.int8)
    written = 0
    for group in range(parquet_file.num_row_groups):
        frame = pl.from_arrow(parquet_file.read_row_group(
            group, columns=["REF", "ALT", "X_PREV1", "X_NEXT1"]))
        labels = frame.select(sbs96_expr().alias("sbs96"))["sbs96"].to_numpy()
        codes = np.fromiter((position.get(label, -1) for label in labels),
                            dtype=np.int8, count=len(labels))
        index[written:written + codes.size] = codes
        written += codes.size

    if written != total:
        raise RuntimeError(f"read {written:,} rows but metadata said {total:,}")
    np.save(cache_path, index)
    log(f"  {float((index >= 0).mean()) * 100:.2f}% of rows carry an SBS96 context")
    return index


def load_or_build_index(context_path: Path, row_order: list[str], output_dir: Path) -> np.ndarray:
    cache_path = output_dir / "sbs96_index.npy"
    if cache_path.exists():
        index = np.load(cache_path, mmap_mode="r")
        log(f"SBS96 index cache hit: {index.shape[0]:,} rows")
        return index
    return build_sbs96_index(Path(context_path), row_order, cache_path)


def sbs96_counts(sbs96_index: np.ndarray, labels: np.ndarray, n_clusters: int,
                 chunk_rows: int = 20_000_000) -> np.ndarray:
    """96 x n_clusters counts over the whole cohort.

    ``bincount`` on a flattened (channel, cluster) index rather than the ``np.add.at`` that
    ``stage3_apply_full`` uses: np.add.at is unbuffered and runs at roughly a million updates
    per second, which is minutes per cell over 157.5M rows. Same arithmetic, ~50x faster.

    Chunked because the flattened index is int64 -- one 157.5M-row array is 1.3 GB of scratch
    nothing reads back.
    """
    counts = np.zeros(96 * n_clusters, dtype=np.int64)
    for start in range(0, labels.shape[0], chunk_rows):
        stop = min(start + chunk_rows, labels.shape[0])
        channels = np.asarray(sbs96_index[start:stop]).astype(np.int64)
        clusters = np.asarray(labels[start:stop]).astype(np.int64)
        keep = (channels >= 0) & (clusters >= 0) & (clusters < n_clusters)
        if not keep.any():
            continue
        counts += np.bincount(channels[keep] * n_clusters + clusters[keep],
                              minlength=counts.size)
    return counts.reshape(96, n_clusters)


def run_cell(cell: Cell, coords: np.ndarray, cell_dir: Path, args,
             sbs96_index: np.ndarray | None = None, row_order: list[str] | None = None,
             signature_database: Path | None = None) -> dict:
    """Fit, label the whole cohort, save the labels, then score it. Writes as it goes."""
    import fast_predict

    cell_dir.mkdir(parents=True, exist_ok=True)
    record: dict = {"cell": asdict(cell), "label": cell.label(),
                    "started": datetime.now(timezone.utc).isoformat()}
    total_rows = coords.shape[0]

    rng = np.random.default_rng(args.seed)
    fit_indices = np.sort(rng.choice(total_rows, size=cell.fit_rows, replace=False))
    fit_coords = np.asarray(coords[fit_indices], dtype=np.float32)

    log(f"  fitting HDBSCAN on {cell.fit_rows:,} rows "
        f"(mcs={cell.min_cluster_size}, ms={cell.min_samples}, "
        f"{cell.cluster_selection_method}, eps={cell.cluster_selection_epsilon})")
    started = perf_counter()
    clusterer, fit_backend = _fit_hdbscan(fit_coords, cell, args)
    record["fit_seconds"] = round(perf_counter() - started, 1)
    # The backend that RAN, plus what was asked for. 'auto' resolves to cuML whenever cuML
    # imports, so the requested value alone does not identify the implementation -- and the
    # two produce materially different clusterings (measured: all 24 cells of this grid
    # differ, cuML from 7.3% below to 22.6% above the CPU cluster count). Recording only the
    # request is what let a cuML sweep sit in a directory named ..._cpu and be published as a
    # CPU/GPU agreement result.
    record["cluster_backend"] = fit_backend
    record["cluster_backend_requested"] = args.cluster_backend
    record["dbcv_backend_requested"] = args.dbcv_backend
    fit_labels = np.asarray(_to_host(clusterer.labels_)).astype(np.int32)
    n_clusters = int(fit_labels.max()) + 1 if (fit_labels >= 0).any() else 0
    record["n_clusters_fit"] = n_clusters
    record["fit_noise_fraction"] = float((fit_labels < 0).mean())
    log(f"    {record['fit_seconds']}s, {n_clusters:,} clusters, "
        f"{record['fit_noise_fraction'] * 100:.2f}% noise on the fit set "
        f"[backend={fit_backend}]")
    if n_clusters == 0:
        record["error"] = "no non-noise clusters"
        (cell_dir / "metrics.json").write_text(json.dumps(record, indent=2))
        return record

    # Full-cohort labels: the fit rows keep the labels the fit gave them, held rows come from
    # the RBC path. Re-predicting the fit rows would be both slower and less accurate --
    # approximate_predict is an approximation OF the fit, so where the fit's own answer
    # exists it is the better one.
    log(f"  labelling all {total_rows:,} rows (RBC)")
    started = perf_counter()
    tables = fast_predict.build_tables(clusterer, n_fit=cell.fit_rows,
                                       min_samples=cell.min_samples)
    held_mask = np.ones(total_rows, dtype=bool)
    held_mask[fit_indices] = False
    held_indices = np.nonzero(held_mask)[0]

    labels = np.empty(total_rows, dtype=np.int32)
    labels[fit_indices] = fit_labels
    index = fast_predict.build_index(fit_coords, 2 * tables.min_samples, args.backend)
    held_labels, held_probabilities = fast_predict.predict(
        tables, fit_coords, np.asarray(coords[held_indices], dtype=np.float32),
        backend=args.backend, batch_rows=args.batch_rows, index=index)
    labels[held_indices] = held_labels
    record["label_seconds"] = round(perf_counter() - started, 1)
    record["held_mean_probability"] = float(held_probabilities.mean())

    # Kept, not dropped after taking the mean. The per-row values are a by-product of a
    # labelling pass that has already happened, and are 630 MB against the fit's hours --
    # but without them run_variant_cluster_pipeline's cluster_probability plot cannot be
    # drawn and its profile panels have no mean/median p to report, and the only way to get
    # them back is to re-fit the cell from scratch.
    if not args.skip_probabilities:
        probabilities = np.empty(total_rows, dtype=np.float32)
        fit_probabilities = getattr(clusterer, "probabilities_", None)
        probabilities[fit_indices] = (
            np.ones(len(fit_indices), dtype=np.float32) if fit_probabilities is None
            else np.asarray(_to_host(fit_probabilities), dtype=np.float32)
        )
        probabilities[held_indices] = held_probabilities
        np.save(cell_dir / "cohort_probabilities.npy", probabilities)
        record["cohort_probabilities"] = str(cell_dir / "cohort_probabilities.npy")
        del probabilities
    del held_labels, held_probabilities

    np.save(cell_dir / "cohort_labels.npy", labels)
    record["cohort_labels"] = str(cell_dir / "cohort_labels.npy")
    record["cohort_n_clusters"] = int(labels.max()) + 1 if (labels >= 0).any() else 0
    record["cohort_noise_fraction"] = float((labels < 0).mean())
    log(f"    {record['label_seconds']}s, "
        f"{record['cohort_noise_fraction'] * 100:.2f}% cohort noise")

    if not args.skip_geometry:
        _score_geometry(record, fit_coords, fit_labels, clusterer, args)

    if not args.skip_model_artefacts:
        _save_model_artefacts(record, clusterer, cell_dir, args, fit_indices)

    if sbs96_index is not None:
        _score_cell(record, cell_dir, sbs96_index, labels, row_order,
                    signature_database, args)

    record["finished"] = datetime.now(timezone.utc).isoformat()
    (cell_dir / "metrics.json").write_text(json.dumps(record, indent=2))
    return record


def _save_model_artefacts(record: dict, clusterer, cell_dir: Path, args,
                          fit_indices: np.ndarray | None = None) -> None:
    """Persist the small clusterer arrays that a re-fit would otherwise be needed to recover.

    The fit is the entire cost of this sweep (3.2 h at 25M), and the clusterer object is
    discarded when the process exits -- so anything derived from it and not written here is
    only obtainable by paying that cost again. These four are all cheap:

    * ``cluster_persistence.npy`` -- the RAW per-cluster array. ``metrics.json`` keeps only
      mean/median/min/max/sum, which cannot answer "which clusters are ephemeral".
    * ``outlier_scores.npy`` -- GLOSH per-point outlier strength over the fit set. Not
      derivable from labels and probabilities; float32 halves it to ~100 MB at 25M rows.
    * ``exemplars.npz`` -- the most persistent points of each cluster, the natural
      representative for interpreting or plotting a cluster.
    * ``relative_validity`` -- already a scalar inside ``geometry``; noted here for
      completeness since it comes from the same object.

    Each artefact is written under its own try/except: a backend that does not expose one
    (cuML does not implement the full CPU attribute set) must not cost the others, and must
    certainly not cost the fit.
    """
    saved, missing = {}, []

    for name, attribute, dtype in (("cluster_persistence", "cluster_persistence_", np.float64),
                                   ("outlier_scores", "outlier_scores_", np.float32)):
        try:
            values = getattr(clusterer, attribute, None)
            if values is None:
                missing.append(attribute)
                continue
            array = np.asarray(_to_host(values), dtype=dtype)
            path = cell_dir / f"{name}.npy"
            np.save(path, array)
            saved[name] = {"path": str(path), "shape": list(array.shape)}
        except Exception as exc:  # noqa: BLE001 - diagnostics must never cost the fit
            missing.append(f"{attribute} ({type(exc).__name__})")

    try:
        exemplars = getattr(clusterer, "exemplars_", None)
        if exemplars is None:
            missing.append("exemplars_")
        else:
            # A ragged list, one array per cluster -- npz keeps them separate rather than
            # forcing a padded rectangle that would silently invent points.
            blocks = {f"cluster_{i}": np.asarray(_to_host(block), dtype=np.float32)
                      for i, block in enumerate(exemplars)}
            path = cell_dir / "exemplars.npz"
            np.savez_compressed(path, **blocks)
            saved["exemplars"] = {"path": str(path), "n_clusters": len(blocks)}
    except Exception as exc:  # noqa: BLE001
        missing.append(f"exemplars_ ({type(exc).__name__})")

    try:
        import joblib
        pkl_path = cell_dir / "hdbscan_model.pkl"
        joblib.dump(clusterer, pkl_path)
        saved["hdbscan_model"] = {"path": str(pkl_path)}
    except Exception as exc:  # noqa: BLE001
        missing.append(f"hdbscan_model ({type(exc).__name__})")

    # The rows the model was fit on, in fit order. Downstream prediction (fast_predict's
    # tables index into the fit set positionally) needs the EXACT set, and re-deriving it
    # from (seed, fit_rows) only agrees while both scripts keep the same derivation and the
    # same coords length -- a silent divergence would relabel every read against the wrong
    # neighbours. Cheap to store (8 MB at 1M rows) next to the model it belongs to.
    if fit_indices is not None:
        try:
            path = cell_dir / "fit_indices.npy"
            np.save(path, np.asarray(fit_indices, dtype=np.int64))
            saved["fit_indices"] = {"path": str(path), "shape": [int(len(fit_indices))]}
        except Exception as exc:  # noqa: BLE001
            missing.append(f"fit_indices ({type(exc).__name__})")

    record["model_artefacts"] = saved
    if missing:
        record["model_artefacts_missing"] = missing
    log(f"    saved {', '.join(saved) or 'nothing'}"
        + (f"; unavailable: {', '.join(missing)}" if missing else ""))


def _score_geometry(record: dict, fit_coords: np.ndarray, fit_labels: np.ndarray,
                    clusterer, args) -> None:
    """DBCV, probabilities, persistence, connectivity -- on the FIT set.

    The fit set, not the cohort: DBCV is O(n^2)-ish and connectivity needs a kNN, neither of
    which is affordable over 157.5M rows, and the geometry question is about the model the
    fit produced. Failures are swallowed for the same reason scoring is -- these are
    diagnostics, not something worth losing hours of fitting over.
    """
    try:
        import cluster_quality
        started = perf_counter()
        probabilities = getattr(clusterer, "probabilities_", None)
        record["geometry"] = cluster_quality.summarise(
            fit_coords, fit_labels,
            probabilities=None if probabilities is None else _to_host(probabilities),
            clusterer=clusterer,
            connectivity_rows=args.connectivity_rows,
            dbcv_rows=args.dbcv_rows,
            dbcv_max_clusters=args.dbcv_max_clusters,
            dbcv_stratified=not args.dbcv_uniform,
            dbcv_per_cluster=args.dbcv_per_cluster or None,
            dbcv_backend=args.dbcv_backend,
            seed=args.seed)
        record["geometry_seconds"] = round(perf_counter() - started, 1)
        geometry = record["geometry"]
        relative = geometry.get("relative_validity")
        log(f"    DBCV {geometry.get('dbcv')} "
            f"({geometry.get('dbcv_points_per_cluster_median')} pts/cluster), "
            f"rel_validity {'n/a' if relative is None else f'{relative:.4f}'}, "
            f"connectivity {geometry.get('connectivity_mean', float('nan')):.3f}, "
            f"prob mean {geometry.get('prob_mean', float('nan')):.3f}")
    except Exception as exc:  # noqa: BLE001
        record["geometry_error"] = f"{type(exc).__name__}: {exc}"
        log(f"  geometry metrics failed: {type(exc).__name__}: {str(exc)[:200]}")


def _score_cell(record: dict, cell_dir: Path, sbs96_index: np.ndarray, labels: np.ndarray,
                row_order: list[str], signature_database: Path, args) -> None:
    """SBS96 matrix -> SigProfiler -> metrics.

    The matrix is built from the cached int8 index rather than re-reading context.parquet per
    cell. SigProfiler itself is ``rvcp.Analyzer.cosmic_fit`` -- the same CPU call every other
    stage in this repo makes, and the same one stage3_apply_full uses. There is no GPU
    SigProfiler and it would not help: cosmic_fit only ever sees a 96 x n_clusters matrix.

    Scoring failures are recorded and swallowed. The labelling above is the expensive part
    (up to 3.25 h of fitting); losing it because SigProfiler tripped over one cell would be
    the worst possible trade, and the saved cohort_labels.npy can always be rescored later.
    """
    import assignment_metrics
    import run_variant_cluster_pipeline as rvcp
    import spectrum_metrics

    try:
        n_clusters = int(labels.max()) + 1
        log("  building SBS96 counts (cached index)")
        started = perf_counter()
        counts = sbs96_counts(sbs96_index, labels, n_clusters)
        present = np.nonzero(counts.sum(axis=0) > 0)[0]
        cluster_columns = [f"cluster_{int(c)}" for c in present]
        counts = counts[:, present]
        record["counts_seconds"] = round(perf_counter() - started, 1)
        record["sbs96_total_mutations"] = int(counts.sum())

        sigprof_root = cell_dir / (f"sigprofilerassignment_uv_only_"
                                   f"{args.genome_build.lower()}_v{args.cosmic_version}")
        input_dir = sigprof_root / "input"
        output_sigprof = sigprof_root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_sigprof.mkdir(parents=True, exist_ok=True)

        matrix_path = input_dir / "cluster_sbs96_matrix.tsv"
        matrix = {"Type": row_order}
        for position, name in enumerate(cluster_columns):
            matrix[name] = counts[:, position].tolist()
        pl.DataFrame(matrix).write_csv(matrix_path, separator="\t")

        log(f"  SigProfiler on {len(cluster_columns):,} clusters "
            f"({int(counts.sum()):,} mutations)")
        started = perf_counter()
        rvcp.Analyzer.cosmic_fit(
            samples=str(matrix_path), output=str(output_sigprof),
            signature_database=str(signature_database),
            genome_build=args.genome_build, cosmic_version=float(args.cosmic_version),
            make_plots=False, collapse_to_SBS96=True, connected_sigs=False, verbose=False,
            input_type="matrix", context_type="96", export_probabilities=True,
            sample_reconstruction_plots=False, cpu=args.sigprofiler_cpu,
            add_background_signatures=False,
        )
        record["sigprofiler_seconds"] = round(perf_counter() - started, 1)
        record["sbs96_matrix"] = str(matrix_path)

        stats_path = (output_sigprof / "Assignment_Solution" / "Solution_Stats"
                      / "Assignment_Solution_Samples_Stats.txt")
        record["assignment"] = assignment_metrics.summarise(stats_path)
        record["spectrum"] = spectrum_metrics.summarise(matrix_path)
        log(f"    cosine mean {record['assignment']['cosine_mean']:.3f}, "
            f"L1% mean {record['assignment'].get('l1_pct_mean', float('nan')):.1f}, "
            f"top-channel median {record['spectrum']['top_channel_share_median']:.3f}")
    except Exception as exc:  # noqa: BLE001 - never lose a labelled cohort over scoring
        record["scoring_error"] = f"{type(exc).__name__}: {exc}"
        record["scoring_traceback"] = traceback.format_exc()
        log(f"  SCORING FAILED (labels kept): {type(exc).__name__}: {str(exc)[:200]}")


def _fit_hdbscan(fit_coords: np.ndarray, cell: Cell, args):
    """Fit, asking for the MST so relative_validity_ is available.

    ``gen_min_span_tree=True`` retains the mutual-reachability MST the clustering already
    builds, which is what populates ``relative_validity_`` -- a DBCV approximation over ALL
    fit points rather than a subsample, for almost no extra cost.

    **cuML's HDBSCAN does not implement ``outlier_scores_`` or ``exemplars_`` at all, and
    its ``cluster_persistence_`` is degenerate** -- measured on the real 500K/mcs1000/ms15
    cell as exactly 1.0 for every cluster, where a CPU fit on equivalent data gave a real
    0.597-0.653 spread. This is not an argument cuML rejects (``gen_min_span_tree`` is
    accepted without error); the attributes simply are not populated. There is no flag that
    recovers them from a GPU fit -- only ``--cluster-backend cpu`` does, by fitting with the
    CPU ``hdbscan`` package instead, which costs real wall time: CPU HDBSCAN has not been
    benchmarked against this project's 2-D embedding at scale, so time it on your smallest
    cell before requesting it on anything past a few million fit rows.

    Returns ``(clusterer, backend)`` where ``backend`` is the backend that actually ran, not
    the one requested. Those differ whenever ``--cluster-backend auto`` is left at its default
    and cuML happens to be importable, which is exactly how a sweep written to a directory
    named ``param_sweep_refit_cpu`` came to hold cuML fits. The caller records the resolved
    value so the artefacts can never again misrepresent which implementation produced them.
    """
    common = dict(min_cluster_size=cell.min_cluster_size, min_samples=cell.min_samples,
                  cluster_selection_method=cell.cluster_selection_method,
                  cluster_selection_epsilon=cell.cluster_selection_epsilon,
                  metric="euclidean", prediction_data=True)

    use_cpu = args.cluster_backend == "cpu"
    if not use_cpu and args.cluster_backend == "auto":
        try:
            import cuml  # noqa: F401
        except ImportError:
            use_cpu = True

    if use_cpu:
        import hdbscan
        build, extra, backend = hdbscan.HDBSCAN, {"core_dist_n_jobs": args.threads}, "cpu"
    else:
        from cuml.cluster import HDBSCAN as CumlHDBSCAN
        build, extra, backend = CumlHDBSCAN, {}, "cuml"

    if not args.no_min_span_tree:
        try:
            return build(**common, **extra, gen_min_span_tree=True).fit(fit_coords), backend
        except TypeError:
            log("    backend rejected gen_min_span_tree; relative_validity_ unavailable")
    return build(**common, **extra).fit(fit_coords), backend


def _to_host(array):
    for attribute in ("to_numpy", "get"):
        method = getattr(array, attribute, None)
        if callable(method):
            return np.asarray(method())
    return np.asarray(array)


# ── aggregate ──────────────────────────────────────────────────────────────────

def aggregate(output_dir: Path) -> pl.DataFrame:
    """Every finished cell: clustering shape, spectrum concentration, SigProfiler fit.

    ``top_chan`` leads because it is the gate. The 2026-08-04 runs measured a median cluster
    at 0.87 of a single trinucleotide channel, and no COSMIC profile is that concentrated --
    so a cell above ~0.5 there has already lost the signature question and its cosine is
    reporting channel identity rather than fit quality.
    """
    rows = []
    for metrics_path in sorted((output_dir / "cells").glob("*/metrics.json")):
        payload = json.loads(metrics_path.read_text())
        if "error" in payload:
            continue
        cell = payload["cell"]
        assignment = payload.get("assignment", {})
        spectrum = payload.get("spectrum", {})
        geometry = payload.get("geometry", {})
        nan = float("nan")
        rows.append({
            "label": payload["label"],
            "fit_rows": cell["fit_rows"],
            "mcs": cell["min_cluster_size"],
            "ms": cell["min_samples"],
            "method": cell["cluster_selection_method"],
            "eps": cell["cluster_selection_epsilon"],
            "n_clusters": payload.get("cohort_n_clusters", 0),
            "noise_pct": round(payload.get("cohort_noise_fraction", 0) * 100, 2),
            # Which implementation actually produced this row. Sits next to n_clusters
            # because the two backends disagree on it for every cell of this grid, so the
            # count is meaningless without knowing which one ran. "?" marks a cell written
            # before this column existed -- infer those from persistence: cuML reports
            # exactly 1.0 for every cluster, a CPU fit reports a real distribution.
            "backend": payload.get("cluster_backend", "?"),
            "dbcv_backend": geometry.get("dbcv_backend", "?"),
            "top_chan": round(spectrum.get("top_channel_share_median", nan), 3),
            "tc>0.5": round(spectrum.get("frac_above_05", nan), 3),
            "tc>0.8": round(spectrum.get("frac_above_08", nan), 3),
            # Geometry. Read connect_mean WITH n_clusters -- it rises monotonically as
            # clusters merge, so a high value alone says nothing.
            "dbcv": geometry.get("dbcv"),
            "connect": round(geometry.get("connectivity_mean", nan), 3),
            "prob_mean": round(geometry.get("prob_mean", nan), 3),
            "prob>0.8": round(geometry.get("prob_frac_above_08", nan), 3),
            "persist": round(geometry.get("persistence_mean", nan), 4),
            "cos_mean": round(assignment.get("cosine_mean", nan), 4),
            "cos_wmean": round(assignment.get("cosine_weighted_mean", nan), 4),
            "l1%_mean": round(assignment.get("l1_pct_mean", nan), 2),
            "l2%_mean": round(assignment.get("l2_pct_mean", nan), 2),
            "mut%>0.8": round(100 * assignment.get("mutation_share_above_08", nan), 1),
            "fit_s": payload.get("fit_seconds", 0),
            "label_s": payload.get("label_seconds", 0),
        })
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).sort("top_chan", nulls_last=True)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--coords", type=Path, help="coords.npy, 157.5M x 2")
    parser.add_argument("--context", type=Path,
                        help="<embed_dir>/context.parquet, row-aligned with coords.npy. "
                             "Without it the sweep only labels and saves; with it each cell "
                             "also runs SigProfiler and scores itself.")
    parser.add_argument("--genome-build", default="GRCh38")
    parser.add_argument("--cosmic-version", default="3.5")
    parser.add_argument("--sigprofiler-cpu", type=int, default=16)
    parser.add_argument("--build-index-only", action="store_true",
                        help="build the SBS96 index cache and exit")

    parser.add_argument("--skip-geometry", action="store_true",
                        help="skip DBCV/connectivity/probability/persistence metrics")
    parser.add_argument("--connectivity-rows", type=int, default=200_000)
    parser.add_argument("--dbcv-rows", type=int, default=25_000)
    parser.add_argument("--dbcv-max-clusters", type=int, default=None,
                        help="refuse DBCV above this many clusters. Default None: stratified "
                             "sampling keeps every cluster scoreable, so the old ceiling of "
                             "500 (which left 17 of 28 cells unscored) is no longer needed.")
    parser.add_argument("--dbcv-uniform", action="store_true",
                        help="sample DBCV rows uniformly instead of stratified by cluster "
                             "(the pre-2026-08-13 behaviour; starves small clusters)")
    parser.add_argument("--dbcv-backend", default="auto",
                        choices=["auto", "kdbcv", "hdbscan"],
                        help="auto prefers k-DBCV (KD-tree; 42x faster at k=400, agrees "
                             "with hdbscan to 0.001) and falls back if it is not installed")
    parser.add_argument("--dbcv-per-cluster", type=int, default=400,
                        help="points drawn from EVERY cluster. Fixing this rather than the "
                             "total is what makes DBCV comparable between cells: the bias "
                             "of a sampled DBCV depends on per-cluster resolution. 0 falls "
                             "back to a fixed --dbcv-rows total budget.")
    parser.add_argument("--no-min-span-tree", action="store_true",
                        help="skip gen_min_span_tree=True, losing relative_validity_")
    parser.add_argument("--cluster-backend", default="auto", choices=["auto", "cuml", "cpu"],
                        help="'auto' prefers cuML when importable (fast, but does not "
                             "populate outlier_scores_/exemplars_ and reports a degenerate "
                             "cluster_persistence_ -- measured as exactly 1.0 on the real "
                             "data). 'cpu' forces the hdbscan package, which gives all four "
                             "requested artefacts but is unbenchmarked at this project's "
                             "scale -- time it on your smallest cell first.")
    parser.add_argument("--skip-model-artefacts", action="store_true",
                        help="do not save cluster_persistence.npy, outlier_scores.npy and "
                             "exemplars.npz. They are small (outlier scores dominate at "
                             "~100 MB/cell at 25M fit rows) and are NOT recoverable without "
                             "re-fitting, which is the whole cost of the sweep.")
    parser.add_argument("--skip-probabilities", action="store_true",
                        help="do not write cohort_probabilities.npy (630 MB/cell at 157.5M "
                             "rows). Without it there is no per-row membership probability, "
                             "so run_variant_cluster_pipeline's cluster_probability plot "
                             "cannot be drawn and profile panels report 'mean p = nan' -- "
                             "and recovering them means re-fitting the cell.")

    parser.add_argument("--dry-run", action="store_true",
                        help="print the grid and its projected cost, run nothing")
    parser.add_argument("--aggregate", action="store_true",
                        help="summarise finished cells and exit")

    parser.add_argument("--fit-sizes", default="500000,1000000,5000000,25000000")
    parser.add_argument("--min-cluster-sizes", default="250,500,1000,2500",
                        help="held FIXED across fit sizes -- that is the point of this sweep")
    parser.add_argument("--min-samples", default="5,15")
    parser.add_argument("--methods", default="eom")
    parser.add_argument("--epsilons", default="0.0")
    parser.add_argument("--max-ms-at-25m", type=int, default=5,
                        help="min_samples ceiling at 25M rows; ms=15 OOMed on a 47 GB card")

    parser.add_argument("--backend", default="rbc", choices=["rbc", "brute", "sklearn"])
    parser.add_argument("--batch-rows", type=int, default=5_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--gpu-budget-gb", type=float, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--continue-after-failure", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate:
        table = aggregate(output_dir)
        if table.is_empty():
            log("no finished cells yet")
            return 0
        with pl.Config(tbl_rows=200, tbl_cols=30, tbl_width_chars=250):
            print(table)
        table.write_csv(output_dir / "sweep_ranking.csv")
        log(f"wrote {output_dir / 'sweep_ranking.csv'}")
        return 0

    # The signature database also fixes the SBS96 row order, so it is built before the index.
    sbs96_index = row_order = signature_database = None
    if args.context or args.build_index_only:
        import run_variant_cluster_pipeline as rvcp

        signature_dir = output_dir / "signature_db"
        signature_dir.mkdir(parents=True, exist_ok=True)
        signature_database = signature_dir / f"uv_only_SBS_{args.genome_build}.tsv"
        row_order = rvcp.write_uv_only_signature_database(
            args.genome_build, args.cosmic_version, signature_database)
        sbs96_index = load_or_build_index(args.context, row_order, output_dir)
        if args.build_index_only:
            return 0

    fit_sizes = [int(v) for v in args.fit_sizes.split(",") if v]
    grid = build_grid(
        fit_sizes=fit_sizes,
        min_cluster_sizes=[int(v) for v in args.min_cluster_sizes.split(",") if v],
        min_samples=[int(v) for v in args.min_samples.split(",") if v],
        methods=[v for v in args.methods.split(",") if v],
        epsilons=[float(v) for v in args.epsilons.split(",") if v],
        max_min_samples_at={25_000_000: args.max_ms_at_25m, 50_000_000: 0},
    )

    cells_dir = output_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    done = {path.parent.name for path in cells_dir.glob("*/metrics.json")} if args.resume else set()
    pending = [cell for cell in grid if cell.label() not in done]

    fit_total = sum(projected_fit_seconds(c.fit_rows, c.min_samples) for c in pending)
    # 5.3 min RBC labelling, plus ~2 min counts + SigProfiler when scoring is on.
    overhead = len(pending) * (5.3 * 60 + (120 if args.context else 0))
    log(f"grid: {len(grid)} cells, {len(done)} already done, {len(pending)} to run")
    log(f"projected: {fit_total / 3600:.1f} h fitting + {overhead / 3600:.1f} h "
        f"labelling{'/scoring' if args.context else ''} "
        f"= {(fit_total + overhead) / 3600:.1f} h total")
    for cell in pending:
        log(f"    {cell.label():48} ~{projected_fit_seconds(cell.fit_rows, cell.min_samples) / 60:6.1f} min fit")
    if args.dry_run:
        return 0

    if not args.coords:
        raise SystemExit("--coords is required to run the sweep")
    coords = np.load(args.coords, mmap_mode="r")
    log(f"coords: {coords.shape[0]:,} x {coords.shape[1]}")
    if sbs96_index is not None:
        if sbs96_index.shape[0] != coords.shape[0]:
            raise SystemExit(
                f"SBS96 index has {sbs96_index.shape[0]:,} rows but coords has "
                f"{coords.shape[0]:,} -- they must come from the same stage-1 run")
        log(f"scoring on: {args.context} (SigProfiler per cell)")
    else:
        log("no --context: cells will be labelled and saved but NOT signature-scored")

    if args.gpu_budget_gb:
        try:
            import sweep_core
            sweep_core.apply_gpu_budget("sweep", budget_gb=args.gpu_budget_gb)
        except Exception as exc:  # noqa: BLE001
            log(f"gpu budget not applied: {exc}")

    (output_dir / "sweep_config.json").write_text(json.dumps({
        "grid": [asdict(c) for c in grid], "seed": args.seed, "backend": args.backend,
        # "backend" above is the LABELLING backend (rbc/brute/sklearn). These two are the
        # clustering and DBCV backends, recorded separately because they decide what the
        # numbers mean. Both are the requested values; the resolved cluster backend is in
        # each cell's metrics.json under "cluster_backend", since 'auto' is only resolved
        # per fit.
        "cluster_backend_requested": args.cluster_backend,
        "dbcv_backend_requested": args.dbcv_backend,
        "started": datetime.now(timezone.utc).isoformat(),
    }, indent=2))

    for position, cell in enumerate(pending, start=1):
        log(f"[{position}/{len(pending)}] {cell.label()}")
        cell_dir = cells_dir / cell.label()
        try:
            run_cell(cell, coords, cell_dir, args, sbs96_index=sbs96_index,
                     row_order=row_order, signature_database=signature_database)
        except Exception as exc:  # noqa: BLE001 - a failed cell must not lose the sweep
            cell_dir.mkdir(parents=True, exist_ok=True)
            (cell_dir / "failed.json").write_text(json.dumps({
                "cell": asdict(cell), "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }, indent=2))
            log(f"  FAILED: {type(exc).__name__}: {str(exc)[:300]}")
            if not args.continue_after_failure:
                raise

    table = aggregate(output_dir)
    if not table.is_empty():
        table.write_csv(output_dir / "sweep_ranking.csv")
        with pl.Config(tbl_rows=200, tbl_cols=30, tbl_width_chars=250):
            print(table)
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
