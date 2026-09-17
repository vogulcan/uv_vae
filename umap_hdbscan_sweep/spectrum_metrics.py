#!/usr/bin/env python
"""Clustering metrics computed on SBS96 cluster spectra, with no signature reference involved.

Why this exists
---------------
Re-reading the 2026-08-04 sweeps showed the clusters are largely **single-context**: the
median cluster in ``runs/nn15_md0.0_nc2/mcs*_ms5`` puts **87 %** of its mutations in one
trinucleotide channel, and all 40 clusters sharing cosine 0.296 are T[C>T]T at 93.9 % purity.
A mutational signature is spread across many of the 96 channels, so a cluster that is 87 %
one channel cannot resemble one -- which is why cosine-to-reference was flat at 0.296 across
a 25x ``min_cluster_size`` range and identical across two different embeddings. It was
reporting channel identity, not clustering quality.

So the metrics here deliberately never touch a signature database. They ask two questions of
the spectra alone:

1. **Is each cluster a signature or a channel?** ``top_channel_share``.
2. **Are the clusters different from each other?** pairwise cosine / L1 / L2 between spectra.

Both are cheap, need only ``cluster_sbs96_matrix.tsv`` (written by
``run_variant_cluster_pipeline.write_cluster_sbs96_matrix``), and can gate a parameter choice
before any GPU-days go into SigProfiler.

Every spectrum is normalised to sum 1 before comparison. Without that, L1 and L2 measure
cluster size and nothing else -- a 400,000-mutation cluster would sit far from a 3,000-row
one no matter how identical their shapes.

    # score every matrix under a tree, newest sweep first
    python umap_hdbscan_sweep/spectrum_metrics.py \
        "uv_vae/runs/**/cluster_sbs96_matrix.tsv" --output metrics.csv

    # one cell, with the per-cluster detail
    python umap_hdbscan_sweep/spectrum_metrics.py <matrix.tsv> --per-cluster detail.csv
"""
from __future__ import annotations

import argparse
import glob
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

# Above this, a cluster is one context wearing a cluster's name rather than a spectrum.
# 0.5 is the "more than half of it is a single channel" line; 0.8 is unambiguous.
CONCENTRATION_THRESHOLDS = (0.5, 0.8)

# Pairwise work is O(n^2) in clusters and the largest cell has 5,709 of them. 4,000 keeps the
# dense n x n gram under ~130 MB and the sampling error on a mean over ~8M pairs negligible.
DEFAULT_MAX_PAIRWISE = 4_000


def load_spectra(matrix_path: Path) -> tuple[list[str], list[str], np.ndarray]:
    """Read a cluster_sbs96_matrix.tsv into (channels, cluster names, 96 x n_clusters)."""
    frame = pl.read_csv(matrix_path, separator="\t")
    channels = frame[frame.columns[0]].to_list()
    names = frame.columns[1:]
    matrix = frame.select(names).to_numpy().astype(np.float64)
    return channels, list(names), matrix


def normalise(matrix: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Columns to sum 1, dropping empty clusters.

    Empty columns are dropped rather than zero-filled: a zero column has no direction, so its
    cosine to anything is undefined and it would otherwise poison every pairwise mean.
    """
    totals = matrix.sum(axis=0)
    keep = totals > 0
    return matrix[:, keep] / totals[keep], [n for n, k in zip(names, keep) if k]


def top_channel_stats(spectra: np.ndarray, channels: list[str]) -> dict:
    """How concentrated each cluster is in its single most common channel.

    This is the metric that fails fast. A cell whose median cluster is 0.87 of one channel has
    already lost the signature question, whatever its silhouette or its cosine-to-reference
    says, because no COSMIC profile is that concentrated.
    """
    share = spectra.max(axis=0)
    dominant = Counter(channels[i] for i in spectra.argmax(axis=0))
    statistics = {
        "n_clusters": int(spectra.shape[1]),
        "top_channel_share_median": float(np.median(share)),
        "top_channel_share_mean": float(share.mean()),
        "distinct_dominant_channels": int(len(dominant)),
        # A partition with 1,150 clusters spread over 64 dominant channels is ~18 clusters
        # per channel: the clustering is subdividing WITHIN a context, not separating processes.
        "clusters_per_dominant_channel": float(spectra.shape[1] / len(dominant)) if dominant else 0.0,
        "most_common_dominant": dominant.most_common(3),
    }
    for threshold in CONCENTRATION_THRESHOLDS:
        key = str(threshold).replace(".", "")
        statistics[f"frac_above_{key}"] = float((share > threshold).mean())
    return statistics


def _upper_triangle_mean(square: np.ndarray) -> float:
    rows = square.shape[0]
    if rows < 2:
        return float("nan")
    total = float(square.sum() - np.trace(square)) / 2.0
    return total / (rows * (rows - 1) / 2.0)


def _mean_l1(points: np.ndarray, block: int = 128) -> float:
    """Mean pairwise L1, blocked over rows.

    No gram-matrix shortcut exists for L1, and the full (n, n, 96) difference tensor would be
    hundreds of GB at n=4,000. Blocking keeps peak memory at block x n x 96.
    """
    rows = points.shape[0]
    if rows < 2:
        return float("nan")
    total = 0.0
    for start in range(0, rows, block):
        stop = min(start + block, rows)
        chunk = np.abs(points[start:stop, None, :] - points[None, :, :]).sum(axis=2)
        total += float(chunk.sum())
    return (total / 2.0) / (rows * (rows - 1) / 2.0)


def pairwise_stats(spectra: np.ndarray, max_clusters: int = DEFAULT_MAX_PAIRWISE,
                   seed: int = 0) -> dict:
    """Mean pairwise cosine similarity, L1 and L2 between normalised cluster spectra.

    High cosine / low L1 means the clusters are near-duplicates of each other in mutation
    space -- the partition has many groups but few distinct spectra, which is what a
    context-driven clustering looks like when several clusters share a dominant channel.

    L1 between two distributions is twice their total variation distance, so it lands in
    [0, 2] and 2.0 means disjoint support.
    """
    points = spectra.T  # n_clusters x 96
    rows = points.shape[0]
    subsampled = rows > max_clusters
    if subsampled:
        chosen = np.random.default_rng(seed).choice(rows, size=max_clusters, replace=False)
        points = points[np.sort(chosen)]
        rows = max_clusters
    if rows < 2:
        return {"pairwise_n": rows, "pairwise_subsampled": subsampled,
                "cosine_mean": float("nan"), "l1_mean": float("nan"), "l2_mean": float("nan")}

    norms = np.linalg.norm(points, axis=1, keepdims=True)
    unit = points / norms
    gram = unit @ unit.T

    squared = (points * points).sum(axis=1)
    distances_squared = np.maximum(squared[:, None] + squared[None, :] - 2 * (points @ points.T), 0.0)

    return {
        "pairwise_n": int(rows),
        "pairwise_subsampled": bool(subsampled),
        "cosine_mean": _upper_triangle_mean(gram),
        "l1_mean": _mean_l1(points),
        "l2_mean": _upper_triangle_mean(np.sqrt(distances_squared)),
    }


def summarise(matrix_path: Path, max_clusters: int = DEFAULT_MAX_PAIRWISE,
              seed: int = 0) -> dict:
    channels, names, matrix = load_spectra(matrix_path)
    spectra, names = normalise(matrix, names)
    if spectra.shape[1] == 0:
        return {"matrix": str(matrix_path), "n_clusters": 0, "error": "no non-empty clusters"}
    record = {"matrix": str(matrix_path), "total_mutations": float(matrix.sum())}
    record.update(top_channel_stats(spectra, channels))
    record.update(pairwise_stats(spectra, max_clusters=max_clusters, seed=seed))
    return record


def per_cluster_table(matrix_path: Path) -> pl.DataFrame:
    channels, names, matrix = load_spectra(matrix_path)
    spectra, names = normalise(matrix, names)
    share = spectra.max(axis=0)
    return pl.DataFrame({
        "cluster": names,
        "mutations": matrix.sum(axis=0)[matrix.sum(axis=0) > 0],
        "top_channel": [channels[i] for i in spectra.argmax(axis=0)],
        "top_channel_share": share,
        # How many channels it takes to cover 90% of the cluster -- a spread-out spectrum
        # needs many, a context cluster needs one. The epsilon is not cosmetic: nine channels
        # of exactly 0.1 sum to 0.8999999999999999 in binary floating point, so an exact
        # comparison reports 10 and every evenly-spread cluster is overstated by one.
        "channels_for_90pct": [
            int(np.searchsorted(np.cumsum(np.sort(column)[::-1]), 0.9 - 1e-9) + 1)
            for column in spectra.T
        ],
    }).sort("top_channel_share", descending=True)


def label_for(matrix_path: Path) -> str:
    """Name a cell by the run directory the matrix sits under, not by the filename.

    Every matrix is called cluster_sbs96_matrix.tsv, so the filename alone identifies nothing.
    The cell name is three levels up: <run>/<cell>/sigprofilerassignment_.../input/<file>.
    """
    parts = matrix_path.resolve().parts
    for anchor, part in enumerate(parts):
        if part.startswith("sigprofilerassignment"):
            return "/".join(parts[max(0, anchor - 2):anchor])
    return str(matrix_path.parent)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("matrices", nargs="+",
                        help="cluster_sbs96_matrix.tsv paths or globs (** supported)")
    parser.add_argument("--output", type=Path, help="write the summary table to CSV")
    parser.add_argument("--per-cluster", type=Path,
                        help="write the per-cluster detail for the FIRST matrix to CSV")
    parser.add_argument("--max-pairwise", type=int, default=DEFAULT_MAX_PAIRWISE,
                        help="subsample clusters above this before the O(n^2) pairwise work")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths: list[Path] = []
    for pattern in args.matrices:
        expanded = [Path(p) for p in glob.glob(pattern, recursive=True)]
        paths.extend(expanded if expanded else [Path(pattern)])
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise SystemExit("no matrix files matched")

    rows = []
    for path in sorted(paths):
        record = summarise(path, max_clusters=args.max_pairwise, seed=args.seed)
        record["cell"] = label_for(path)
        rows.append(record)
        print(f"  scored {record['cell']}", flush=True)

    table = pl.DataFrame([{
        "cell": r["cell"],
        "clusters": r.get("n_clusters", 0),
        "top_chan_median": round(r.get("top_channel_share_median", float("nan")), 3),
        "frac>0.5": round(r.get("frac_above_05", float("nan")), 3),
        "frac>0.8": round(r.get("frac_above_08", float("nan")), 3),
        "dom_channels": r.get("distinct_dominant_channels", 0),
        "clus_per_chan": round(r.get("clusters_per_dominant_channel", float("nan")), 1),
        "cosine_mean": round(r.get("cosine_mean", float("nan")), 4),
        "l1_mean": round(r.get("l1_mean", float("nan")), 4),
        "l2_mean": round(r.get("l2_mean", float("nan")), 4),
        "subsampled": r.get("pairwise_subsampled", False),
    } for r in rows]).sort("top_chan_median")

    with pl.Config(tbl_rows=200, tbl_cols=20, tbl_width_chars=220):
        print(table)
    if args.output:
        table.write_csv(args.output)
        print(f"wrote {args.output}")
    if args.per_cluster:
        per_cluster_table(sorted(paths)[0]).write_csv(args.per_cluster)
        print(f"wrote {args.per_cluster}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
