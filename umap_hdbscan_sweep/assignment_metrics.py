#!/usr/bin/env python
"""Aggregate SigProfilerAssignment's per-cluster fit statistics into per-cell numbers.

SigProfiler already computes everything needed here. ``Assignment_Solution_Samples_Stats.txt``
carries, per cluster: ``Cosine Similarity``, ``L1 Norm``, ``L1_Norm_%``, ``L2 Norm``,
``L2_Norm_%``, ``KL Divergence`` and ``Correlation``. This module only reads, normalises and
averages them -- nothing is recomputed, so these numbers are the same ones every earlier run
in this repo reported.

Two things this gets right that a naive mean would not
------------------------------------------------------
**Use the percentage norms, not the raw ones.** ``L1 Norm`` is in mutation counts: a
400,000-mutation cluster has a bigger L1 than a 3,000-mutation one at identical fit quality.
Averaging raw norms across clusters therefore ranks cells by cluster SIZE, and since
``min_cluster_size`` is the swept parameter, the metric would hand the parameter back instead
of measuring anything. ``L1_Norm_%`` divides by the cluster's total mutations and is
comparable across sizes. Both are reported; the percentage is the one to rank on.

**Weight by mutations as well as by cluster.** A plain mean over clusters gives a 3,000-row
cluster the same vote as a 400,000-row one. The 2026-08-04 sweep showed cluster counts moving
8-fold (544 -> 68) while the mutation share they carried stayed at ~10.5 %, so the
cluster-mean and the mutation-weighted mean can tell opposite stories. Both are emitted.

Note that ``L1_Norm_%`` routinely exceeds 100 % -- cluster_0 of the mcs1000 cell sits at
123.8 % with cosine 0.876. The reconstruction error can be larger than the mutation total
because the fitted spectrum is not constrained to the same total, so treat it as an unbounded
error measure, not a fraction.

    python umap_hdbscan_sweep/assignment_metrics.py \
        "uv_vae/runs/**/Assignment_Solution_Samples_Stats.txt" --output assignment.csv
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import polars as pl

COSINE_THRESHOLDS = (0.7, 0.8)

# Substring -> canonical name. Matched case-insensitively so both the packaged COSMIC spelling
# and any renamed variant resolve, and so a future SigProfiler column rename fails loudly at
# the lookup rather than silently averaging the wrong column.
COLUMN_PATTERNS = {
    "sample": "sample names",
    "mutations": "total mutations",
    "cosine": "cosine similarity",
    "l1_pct": "l1_norm_%",
    "l2_pct": "l2_norm_%",
    "l1_abs": "l1 norm",
    "l2_abs": "l2 norm",
    "kl": "kl divergence",
    "correlation": "correlation",
}


def _find_column(columns: list[str], wanted: str) -> str | None:
    lowered = {c.lower(): c for c in columns}
    if wanted in lowered:
        return lowered[wanted]
    # "l1 norm" must not match "l1_norm_%", so exact-first then prefix without the percent.
    for lower, original in lowered.items():
        if lower.replace(" ", "_") == wanted.replace(" ", "_"):
            return original
    return None


def _as_float(series: pl.Series) -> np.ndarray:
    """Parse a stats column to float, tolerating the trailing '%' on the norm percentages."""
    if series.dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32):
        return series.to_numpy().astype(np.float64)
    cleaned = series.cast(pl.String).str.strip_chars().str.strip_suffix("%")
    return cleaned.cast(pl.Float64, strict=False).to_numpy().astype(np.float64)


def load_stats(stats_path: Path) -> dict[str, np.ndarray | list[str]]:
    frame = pl.read_csv(stats_path, separator="\t")
    resolved: dict[str, np.ndarray | list[str]] = {}
    for key, pattern in COLUMN_PATTERNS.items():
        column = _find_column(frame.columns, pattern)
        if column is None:
            continue
        resolved[key] = (frame[column].to_list() if key == "sample"
                         else _as_float(frame[column]))
    if "cosine" not in resolved:
        raise ValueError(f"no cosine column in {stats_path} (columns: {frame.columns})")
    return resolved


def _weighted(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    return float((values * weights).sum() / total) if total else float("nan")


def summarise(stats_path: Path) -> dict:
    stats = load_stats(stats_path)
    cosine = stats["cosine"]
    mutations = stats.get("mutations")
    if mutations is None:
        mutations = np.ones_like(cosine)

    record: dict = {
        "stats": str(stats_path),
        "n_clusters": int(cosine.size),
        "total_mutations": float(mutations.sum()),
        "cosine_mean": float(cosine.mean()),
        "cosine_median": float(np.median(cosine)),
        "cosine_weighted_mean": _weighted(cosine, mutations),
    }
    for key, label in [("l1_pct", "l1_pct"), ("l2_pct", "l2_pct"),
                       ("l1_abs", "l1_abs"), ("l2_abs", "l2_abs"),
                       ("kl", "kl"), ("correlation", "correlation")]:
        if key not in stats:
            continue
        values = stats[key]
        record[f"{label}_mean"] = float(np.nanmean(values))
        record[f"{label}_median"] = float(np.nanmedian(values))
        if key in ("l1_pct", "l2_pct"):
            record[f"{label}_weighted_mean"] = _weighted(np.nan_to_num(values), mutations)

    # L2 re-based on total mutations instead of ||observed||_2.
    #
    # SigProfiler's L2_Norm_% is ||obs - recon||_2 / ||obs||_2, which is a principled relative
    # error but has a SHAPE-DEPENDENT denominator: ||obs||_2 is 1.0x total mutations for a
    # one-hot spectrum and total/sqrt(k) for one spread over k channels. Measured on real
    # cells it is 0.879 x total for a concentrated cluster set and 0.507 x total for a spread
    # one -- a 1.73x swing driven purely by shape, which is larger than the entire spread of
    # L2_Norm_% across cells. That is why L2_Norm_% ranks cells OPPOSITE to L1_Norm_%
    # (rho -0.819 vs +0.818 against cluster concentration): it rewards concentrated spectra
    # for being concentrated.
    #
    # ||obs||_1 is total mutations for any shape (counts are non-negative), so L1_Norm_% is
    # already unconfounded. Dividing L2 by the same total restores agreement (rho +0.956).
    if "l2_abs" in stats:
        with np.errstate(divide="ignore", invalid="ignore"):
            rebased = np.where(mutations > 0, 100.0 * stats["l2_abs"] / mutations, np.nan)
        record["l2_over_total_pct_mean"] = float(np.nanmean(rebased))
        record["l2_over_total_pct_median"] = float(np.nanmedian(rebased))
        record["l2_over_total_pct_weighted_mean"] = _weighted(np.nan_to_num(rebased), mutations)

    for threshold in COSINE_THRESHOLDS:
        key = str(threshold).replace(".", "")
        above = cosine > threshold
        record[f"clusters_above_{key}"] = int(above.sum())
        record[f"frac_clusters_above_{key}"] = float(above.mean())
        # The share of the COHORT that is confidently fit -- the number that stayed flat at
        # ~10.5% across a 25x mcs range while the cluster count moved 8-fold.
        record[f"mutation_share_above_{key}"] = _weighted(above.astype(np.float64), mutations)
    return record


def label_for(stats_path: Path) -> str:
    parts = stats_path.resolve().parts
    for anchor, part in enumerate(parts):
        if part.startswith("sigprofilerassignment"):
            return "/".join(parts[max(0, anchor - 2):anchor])
    return str(stats_path.parent)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stats", nargs="+",
                        help="Assignment_Solution_Samples_Stats.txt paths or globs")
    parser.add_argument("--output", type=Path, help="write the summary table to CSV")
    return parser.parse_args()


def main() -> int:
    paths: list[Path] = []
    args = parse_args()
    for pattern in args.stats:
        expanded = [Path(p) for p in glob.glob(pattern, recursive=True)]
        paths.extend(expanded if expanded else [Path(pattern)])
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise SystemExit("no stats files matched")

    rows = []
    for path in sorted(paths):
        record = summarise(path)
        record["cell"] = label_for(path)
        rows.append(record)

    table = pl.DataFrame([{
        "cell": r["cell"],
        "clusters": r["n_clusters"],
        "cos_mean": round(r["cosine_mean"], 4),
        "cos_median": round(r["cosine_median"], 4),
        "cos_wmean": round(r["cosine_weighted_mean"], 4),
        "l1%_mean": round(r.get("l1_pct_mean", float("nan")), 2),
        "l1%_wmean": round(r.get("l1_pct_weighted_mean", float("nan")), 2),
        "l2%_mean": round(r.get("l2_pct_mean", float("nan")), 2),
        "l2/tot%": round(r.get("l2_over_total_pct_mean", float("nan")), 2),
        "kl_mean": round(r.get("kl_mean", float("nan")), 3),
        "mut%>0.7": round(100 * r["mutation_share_above_07"], 1),
        "mut%>0.8": round(100 * r["mutation_share_above_08"], 1),
    } for r in rows]).sort("cos_wmean", descending=True)

    with pl.Config(tbl_rows=200, tbl_cols=20, tbl_width_chars=230):
        print(table)
    if args.output:
        table.write_csv(args.output)
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
