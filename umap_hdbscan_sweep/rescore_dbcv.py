"""Post-hoc DBCV for a finished stage-2 sweep, on CPU.

cuML's HDBSCAN does not expose ``relative_validity_``, so every cell fitted on the GPU
records ``dbcv: null`` and ``stage2_sweep``'s "best cell" selection -- which ranks by
DBCV -- comes back empty. This recovers a DBCV-equivalent score *without refitting
anything*, using ``hdbscan.validity.validity_index``, which scores an existing
``(X, labels)`` pair rather than deriving the score from a fit's MST.

Three facts drive the design:

* **It cannot be exact.** ``validity_index`` compares every pair of clusters, so its
  cost is O(S^2) in the number of scored points regardless of how many clusters there
  are. 157.5M rows is not reachable by any margin; the score has to come from a sample.
* **The sample must be shared.** A per-cell sample would make the scores incomparable,
  which defeats the purpose. ``agreement_index`` is already a fixed 200k uniform draw
  shared by every cell, reproducible from ``--seed``, and every cell has persisted its
  labels for exactly those rows in ``agreement_labels.npy``. That is the sample.
* **Coordinates are shared within a UMAP config.** All 15 HDBSCAN cells under one UMAP
  directory index the same embedding, so coordinates are read once per config (16 reads
  rather than 240) with column projection, not once per cell.

The result is a *relative* score for ranking cells against one another under an
identical sampling rule -- not an absolute DBCV of the full 157.5M-row clustering.
Report it as such.

**n_components=16 cells are skipped by default.** ``write_analysis_parquet`` only
persists coordinates for 2-D embeddings, so a 16-D cell has no stored space to score.
``--reload-umap-models`` will reload each ``umap_model.joblib`` and re-transform the
sampled latents instead, which needs the same cuML build that wrote them (the pickles
are version-dependent -- see ``stage2_sweep.persist_model``).

Usage:
    python umap_hdbscan_sweep/rescore_dbcv.py \
        --sweep-root uv_vae/runs/<run>/stage2_sweep \
        --embed-dir  uv_vae/runs/<run>/stage1_embed
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

# Cost is quadratic in this, and memory is quadratic in the LARGEST CLUSTER's share of
# it. 25k with a cluster holding 30% peaks around 450 MB and scores in ~10s; 50k with
# the same shape is 4x both. Raise it only if the ranking looks unstable.
DEFAULT_SAMPLE_ROWS = 25_000

# A cluster's internal core distance divides by (points - 1), so a cluster holding one
# sampled point divides by zero and validity_index raises outright (measured: 2,000
# clusters over 12,500 sampled points fails). Clusters this small in the SAMPLE are
# folded into noise so the cell still yields a score for its substantial clusters --
# with dropped_clusters/dropped_point_fraction recorded, because a score computed after
# discarding most of a fragmented clustering is not comparable to one that kept all of
# its clusters, and the caller has to be able to see that.
MIN_SAMPLED_POINTS_PER_CLUSTER = 4


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover DBCV for a GPU-fitted stage-2 sweep")
    parser.add_argument("--sweep-root", required=True, help="stage2_sweep directory")
    parser.add_argument("--embed-dir", required=True, help="stage1_embed (for latent.npy)")
    parser.add_argument("--output", default=None, help="default: <sweep-root>/dbcv_rescore.json")
    parser.add_argument("--seed", type=int, default=42,
                        help="must match the sweep's --seed; regenerates agreement_index")
    parser.add_argument("--agreement-rows", type=int, default=200_000,
                        help="must match the sweep's --agreement-rows")
    parser.add_argument("--sample-rows", type=int, default=DEFAULT_SAMPLE_ROWS,
                        help=f"rows actually scored, drawn from the agreement sample (default {DEFAULT_SAMPLE_ROWS:,})")
    parser.add_argument("--top-n", type=int, default=0,
                        help="only rescore the N best cells by silhouette (0 = all)")
    parser.add_argument("--max-clusters", type=int, default=500,
                        help="skip cells with more clusters than this; validity_index "
                             "compares every PAIR of clusters in python, so its cost is "
                             "quadratic in cluster count (measured: 400 clusters = 17s, "
                             "5,709 clusters = ~1 hour for a single cell)")
    parser.add_argument("--reload-umap-models", action="store_true",
                        help="score n_components!=2 cells by reloading umap_model.joblib")
    return parser.parse_args()


def load_cells(sweep_root: Path) -> list[dict]:
    """Every successfully-fitted cell in the sweep, newest layout: <umap>/<hdbscan>/metrics.json."""
    cells = []
    for metrics_path in sorted(sweep_root.glob("*/*/metrics.json")):
        try:
            payload = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            continue
        if "error" in payload:  # a cell that OOMed has no labels to score
            continue
        payload["_metrics_path"] = str(metrics_path)
        cells.append(payload)
    return cells


def agreement_positions(total_rows: int, agreement_rows: int, seed: int) -> np.ndarray:
    """Recreate stage2_sweep's agreement_index exactly.

    Must stay byte-identical to the sweep's own construction, or the labels in
    agreement_labels.npy would be paired with the wrong coordinates.
    """
    rows = min(agreement_rows, total_rows)
    return np.sort(np.random.default_rng(seed).choice(total_rows, size=rows, replace=False))


def read_2d_coordinates(analysis_path: Path, positions: np.ndarray) -> np.ndarray:
    """The umap_1/umap_2 columns at ``positions``.

    ``scan_parquet(...).select(...)`` projects to two float columns, so the ~40 wide
    context columns in analysis.parquet are never read off disk. Taking the whole
    column and then indexing beats a row-wise filter: the positions are spread across
    every row group, so a predicate would not prune anything and would cost a lot more.
    """
    frame = pl.scan_parquet(analysis_path).select("umap_1", "umap_2").collect()
    coordinates = frame.to_numpy()
    return np.ascontiguousarray(coordinates[positions], dtype=np.float64)


def transform_latents(model_path: Path, latent_path: Path, positions: np.ndarray) -> np.ndarray:
    """Fallback for n_components != 2: push the sampled latents through the saved model."""
    import joblib

    model = joblib.load(model_path)
    latent = np.load(latent_path, mmap_mode="r")
    sampled = np.ascontiguousarray(latent[positions], dtype=np.float32)
    embedded = model.transform(sampled)
    if hasattr(embedded, "get"):  # cuML returns a device array
        embedded = embedded.get()
    return np.ascontiguousarray(np.asarray(embedded), dtype=np.float64)


def score_cell(space: np.ndarray, labels: np.ndarray, sample: np.ndarray,
               per_cluster: bool = False) -> dict:
    """DBCV for one cell's labels over the shared sample.

    Noise is dropped rather than scored: DBCV is defined over clustered points, and
    hdbscan's validity_index would otherwise treat -1 as an ordinary cluster. Labels
    are made contiguous afterwards because the implementation indexes clusters by
    position, not by id.

    ``per_cluster=True`` additionally returns the per-cluster validity array and the
    original cluster ids it corresponds to. Callers that sample non-uniformly need both:
    the aggregate validity_index returns is weighted by the *sampled* cluster sizes, which
    only match the true sizes under uniform sampling. With the ids and the per-cluster
    scores a caller can reweight by the true sizes and recover the correct aggregate.
    """
    from hdbscan.validity import validity_index

    X = space[sample]
    y = labels[sample]

    clustered = y != -1
    noise_fraction = round(float(1.0 - clustered.mean()), 4)
    X, y = X[clustered], y[clustered]
    if y.size == 0:
        return {"dbcv": None, "note": "every sampled row was noise"}

    unique, inverse = np.unique(y, return_inverse=True)
    counts = np.bincount(inverse, minlength=unique.size)
    keep = counts >= MIN_SAMPLED_POINTS_PER_CLUSTER
    dropped_points = int(counts[~keep].sum())

    kept_rows = keep[inverse]
    X = X[kept_rows]
    _, y = np.unique(inverse[kept_rows], return_inverse=True)

    detail = {
        "scored_rows": int(y.size),
        "scored_clusters": int(keep.sum()),
        "dropped_clusters": int((~keep).sum()),
        "dropped_point_fraction": round(dropped_points / max(1, int(counts.sum())), 4),
        "noise_fraction_in_sample": noise_fraction,
    }
    if detail["scored_clusters"] < 2:
        return {"dbcv": None,
                "note": f"only {detail['scored_clusters']} cluster had "
                        f"{MIN_SAMPLED_POINTS_PER_CLUSTER}+ sampled points",
                **detail}

    if not per_cluster:
        score = validity_index(X, y.astype(np.int64, copy=False), metric="euclidean")
        return {"dbcv": None if not np.isfinite(score) else float(score), **detail}

    score, per_cluster_scores = validity_index(
        X, y.astype(np.int64, copy=False), metric="euclidean", per_cluster_scores=True)
    detail["per_cluster_dbcv"] = [float(v) for v in np.asarray(per_cluster_scores)]
    # The ids the caller knew these clusters by, in the same order as the scores above.
    detail["scored_cluster_ids"] = [int(v) for v in unique[keep]]
    return {"dbcv": None if not np.isfinite(score) else float(score), **detail}


def main() -> int:
    started = perf_counter()
    args = parse_args()

    sweep_root = Path(args.sweep_root).resolve()
    embed_dir = Path(args.embed_dir).resolve()
    output_path = Path(args.output).resolve() if args.output else sweep_root / "dbcv_rescore.json"

    cells = load_cells(sweep_root)
    if not cells:
        raise SystemExit(f"no completed cells under {sweep_root}")
    log(f"Found {len(cells)} completed cells")

    if args.top_n:
        # Silhouette is the only quality signal the GPU sweep produced, so it is what
        # shortlists candidates for the expensive score.
        cells.sort(key=lambda c: c["metrics"].get("silhouette") or -np.inf, reverse=True)
        cells = cells[:args.top_n]
        log(f"Rescoring the top {len(cells)} by silhouette")

    # Applied BEFORE grouping, so a UMAP config whose cells are all too fragmented never
    # triggers its coordinate read -- which is the expensive half of this script.
    results: list[dict] = []
    eligible = []
    for cell in cells:
        n_clusters = cell["metrics"].get("n_clusters") or 0
        if n_clusters > args.max_clusters:
            results.append({
                "cell": cell["cell"], "dbcv": None, "n_clusters": n_clusters,
                "silhouette": cell["metrics"].get("silhouette"),
                "noise_fraction": cell["metrics"].get("noise_fraction"),
                "note": f"skipped: {n_clusters:,} clusters exceeds --max-clusters="
                        f"{args.max_clusters:,}",
            })
        else:
            eligible.append(cell)
    if results:
        log(f"Skipping {len(results)} cells over --max-clusters={args.max_clusters:,}")
    if not eligible:
        raise SystemExit(
            f"every cell exceeds --max-clusters={args.max_clusters:,}; raise it if you "
            f"are willing to spend roughly (k/400)^2 x 17s per cell"
        )
    cells = eligible

    total_rows = int(cells[0]["space_shape"][0])
    positions = agreement_positions(total_rows, args.agreement_rows, args.seed)
    sample = np.random.default_rng(args.seed + 1).choice(
        positions.size, size=min(args.sample_rows, positions.size), replace=False
    )
    log(f"Scoring {sample.size:,} rows drawn from the {positions.size:,}-row agreement sample")

    # Group by UMAP config: the embedding is identical across a config's 15 cells, so
    # it is read once and reused. This is the difference between 16 reads and 240.
    by_config: dict[str, list[dict]] = {}
    for cell in cells:
        by_config.setdefault(cell["cell"].split("/")[0], []).append(cell)

    for config_name, config_cells in sorted(by_config.items()):
        n_components = int(config_cells[0]["umap"].get("n_components", 2))
        first = config_cells[0]

        try:
            if n_components == 2:
                space = read_2d_coordinates(Path(first["analysis_path"]), positions)
            elif args.reload_umap_models:
                space = transform_latents(
                    sweep_root / config_name / "umap_model.joblib",
                    embed_dir / "latent.npy",
                    positions,
                )
            else:
                log(f"[{config_name}] skipped: n_components={n_components} stores no "
                    f"coordinates (pass --reload-umap-models to score it anyway)")
                for cell in config_cells:
                    results.append({"cell": cell["cell"], "dbcv": None,
                                    "note": f"n_components={n_components}; no stored coordinates"})
                continue
        except Exception as exc:
            log(f"[{config_name}] coordinates unavailable: {type(exc).__name__}: {exc}")
            for cell in config_cells:
                results.append({"cell": cell["cell"], "dbcv": None,
                                "note": f"{type(exc).__name__}: {exc}"})
            continue

        for cell in config_cells:
            cell_started = perf_counter()
            labels = np.load(cell["agreement_labels_path"])
            try:
                scored = score_cell(space, labels, sample)
            except Exception as exc:
                scored = {"dbcv": None, "note": f"{type(exc).__name__}: {exc}"}
            scored.update({
                "cell": cell["cell"],
                "silhouette": cell["metrics"].get("silhouette"),
                "n_clusters": cell["metrics"].get("n_clusters"),
                "noise_fraction": cell["metrics"].get("noise_fraction"),
                "seconds": round(perf_counter() - cell_started, 1),
            })
            results.append(scored)
            log(f"  [{cell['cell']}] dbcv={scored['dbcv']} "
                f"(sil={scored['silhouette']}, k={scored['n_clusters']}, {scored['seconds']}s)")

    ranked = sorted(
        (r for r in results if r.get("dbcv") is not None),
        key=lambda r: r["dbcv"], reverse=True,
    )
    payload = {
        "sweep_root": str(sweep_root),
        "scored_rows": int(sample.size),
        "agreement_rows": int(positions.size),
        "seed": args.seed,
        "note": (
            "Relative DBCV over a fixed shared subsample -- comparable across cells, "
            "not an absolute DBCV of the full clustering."
        ),
        "best_cell": ranked[0]["cell"] if ranked else None,
        "ranked": ranked,
        "all": results,
        "seconds": round(perf_counter() - started, 1),
    }
    output_path.write_text(json.dumps(payload, indent=2))

    log("")
    if ranked:
        log(f"Best by rescored DBCV: {ranked[0]['cell']}  dbcv={ranked[0]['dbcv']:.4f}")
        log("Top 10:")
        for row in ranked[:10]:
            log(f"  {row['dbcv']:>8.4f}  k={row['n_clusters']:<6} "
                f"noise={row['noise_fraction']:.3f}  {row['cell']}")
    else:
        log("No cell could be scored.")
    log(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
