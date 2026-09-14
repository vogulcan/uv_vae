#!/usr/bin/env python
"""Cross-size agreement between HDBSCAN fits of different sizes, on the shared eval probe.

The scaling sweep labels one fixed 2M-row eval probe with every fitted model. Because the
probe rows are identical across sizes and are in no fit set, the labellings are directly
comparable -- that is the whole reason the probe exists. This script turns those saved
labels into an answer to "how large does the fit set actually need to be?".

Why ARI is not enough on its own
--------------------------------
min_cluster_size scales proportionally in the sweep (N * 1e-4), so a bigger fit set is asked
for a *coarser* clustering by construction: 856 clusters at 5M, 442 at 25M. If those 442 are
clean unions of the 856, nothing is unstable -- the coarse run is a strict coarsening of the
fine one -- yet ARI drops a lot purely from the merging. ARI cannot tell "clean merge" from
"boundaries moved", and only the second is a stability problem.

So each pair also gets two directional numbers:

    purity(fine -> coarse)  fraction of points whose coarse label is the modal coarse label
                            of their fine cluster. ~1.0 means every fine cluster sits inside
                            a single coarse cluster: a clean merge.
    purity(coarse -> fine)  the reverse. Expected to be < 1 whenever clusters merged.

Read them as a pair:
    fine->coarse ~1, coarse->fine <1   clean hierarchical coarsening. The extra rows did not
                                       move any boundary, they merged clusters the smaller
                                       fit had over-split. Converged.
    fine->coarse ~1 and coarse->fine ~1  the two runs agree outright.
    fine->coarse < ~0.95               real disagreement: boundaries shifted, points changed
                                       cluster in a way no merge explains. Not converged.

Usage
-----
    python umap_hdbscan_sweep/cross_size_ari.py \
        --labels-dir umap_hdbscan_sweep/umap_tests/hdbscan_scaling/labels

    # confident points only (probabilities are saved alongside the labels)
    python umap_hdbscan_sweep/cross_size_ari.py --min-probability 0.5

Labels are small (2M int32 = 8 MB per size), so this runs anywhere -- no GPU, no cuML. Pull
the labels/ directory off miletus and run it locally.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

LABEL_PATTERN = re.compile(r"^fit(\d+)(.*)_evalprobe_labels\.npy$")


# ── contingency and the metrics built on it ────────────────────────────────────

def contingency(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Joint counts of two labellings of the same points.

    Returns (counts, row_sums, col_sums) with labels densified to 0..k-1 first, so arbitrary
    and non-contiguous cluster ids (and -1 for noise) all work. Built by encoding each pair
    as a single integer and counting -- for 2M points and a few thousand clusters per side
    that is a couple of passes over the array, well under a second.
    """
    a_codes, a_inv = np.unique(a, return_inverse=True)
    b_codes, b_inv = np.unique(b, return_inverse=True)
    n_a, n_b = a_codes.size, b_codes.size

    flat = a_inv.astype(np.int64) * n_b + b_inv.astype(np.int64)
    counts = np.bincount(flat, minlength=n_a * n_b).reshape(n_a, n_b)
    return counts, counts.sum(axis=1), counts.sum(axis=0)


def _comb2(x: np.ndarray | np.int64) -> np.ndarray | np.int64:
    return x * (x - 1) // 2


def adjusted_rand(counts: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> float:
    """Standard ARI from a contingency table. 0 = chance, 1 = identical."""
    n = int(counts.sum())
    if n < 2:
        return float("nan")
    sum_cells = float(_comb2(counts.astype(np.int64)).sum())
    sum_rows = float(_comb2(rows.astype(np.int64)).sum())
    sum_cols = float(_comb2(cols.astype(np.int64)).sum())
    total = float(_comb2(np.int64(n)))
    expected = sum_rows * sum_cols / total
    maximum = 0.5 * (sum_rows + sum_cols)
    if maximum == expected:
        return 1.0
    return (sum_cells - expected) / (maximum - expected)


def _entropy(sums: np.ndarray, n: int) -> float:
    p = sums[sums > 0].astype(np.float64) / n
    return float(-(p * np.log(p)).sum())


def mutual_information(counts: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> float:
    n = int(counts.sum())
    nz_r, nz_c = np.nonzero(counts)
    nz = counts[nz_r, nz_c].astype(np.float64)
    outer = rows[nz_r].astype(np.float64) * cols[nz_c].astype(np.float64)
    return float((nz / n * (np.log(nz * n) - np.log(outer))).sum())


def homogeneity_completeness(counts, rows, cols) -> tuple[float, float]:
    """Rows are treated as the 'true' side, columns as the 'predicted' side.

    Matches sklearn's convention: homogeneity = MI / H(true), completeness = MI / H(pred).
    """
    n = int(counts.sum())
    h_rows, h_cols = _entropy(rows, n), _entropy(cols, n)
    mi = mutual_information(counts, rows, cols)
    homogeneity = 1.0 if h_rows == 0 else mi / h_rows
    completeness = 1.0 if h_cols == 0 else mi / h_cols
    return homogeneity, completeness


def variation_of_information(counts, rows, cols) -> tuple[float, float, float]:
    """Meila's VI = H(A|B) + H(B|A), returned with its two halves.

    VI is the principled version of what purity measures here, and the reason it is the right
    metric for this comparison is its behaviour under refinement: it is a true metric on the
    lattice of partitions, and when one clustering is a refinement of the other the two are in
    a covering relation where VI collapses to |H(A) - H(B)| (Meila 2003, 2007).

    Concretely, the conditional term is the exact test:

        H(coarse | fine) == 0   <=>   every fine cluster lies wholly inside one coarse
                                      cluster, i.e. the coarse clustering is a clean merge

    so the fine->coarse direction going to zero is not a heuristic threshold but an identity.
    Units are nats; VI is bounded above by log(n) and is NOT normalised, so compare VI values
    only across pairs on the same probe (which is exactly what this script does).
    """
    n = int(counts.sum())
    h_rows, h_cols = _entropy(rows, n), _entropy(cols, n)
    mi = mutual_information(counts, rows, cols)
    h_a_given_b = max(h_rows - mi, 0.0)
    h_b_given_a = max(h_cols - mi, 0.0)
    return h_a_given_b + h_b_given_a, h_a_given_b, h_b_given_a


def mapping_purity(counts: np.ndarray, rows: np.ndarray) -> float:
    """Fraction of points whose column label is the modal column label of their row cluster.

    This is the "can I predict B from A with a lookup table" number. It is asymmetric on
    purpose: a clean merge gives purity(fine->coarse) = 1 while purity(coarse->fine) < 1.
    """
    n = int(counts.sum())
    if n == 0:
        return float("nan")
    return float(counts.max(axis=1).sum()) / n


# Hennig (2007) reads a cluster's maximum Jaccard across resamples as: above 0.75 the cluster
# is a stable, reproducible pattern; below 0.5 it is "dissolved" and should not be
# interpreted; between the two it is a real but uncertain pattern.
JACCARD_STABLE = 0.75
JACCARD_DISSOLVED = 0.5


def cluster_jaccard(counts: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                    chunk: int = 4096) -> dict:
    """For each cluster in A, the Jaccard of its best match in B.

    Hennig (2007), "Cluster-wise assessment of cluster stability", Comput. Stat. Data Anal.
    52(1):258-271. The global scores above collapse a whole partition to one number; this
    keeps per-cluster resolution, so the answer is "412 of 584 clusters reproduce" rather
    than "ARI 0.31" -- and it says WHICH ones, since a few large unstable clusters and a
    long tail of unstable micro-clusters give the same ARI.

    Hennig resamples and matches each original cluster against the resampled clustering. Here
    the two sides are two independent fits, so the comparison is symmetric and the caller
    runs it in both directions.

    Reported by cluster count AND by point share, because they answer different questions:
    a hundred unstable micro-clusters barely move the point share, while one unstable cluster
    holding a third of the data barely moves the count.
    """
    n_a = counts.shape[0]
    if n_a == 0 or counts.sum() == 0:
        return {}

    best = np.empty(n_a, dtype=np.float64)
    for start in range(0, n_a, chunk):
        block = counts[start:start + chunk].astype(np.float64)
        # |A n B| / |A u B|, and the union is |A| + |B| - |A n B|
        union = rows[start:start + chunk, None] + cols[None, :] - block
        with np.errstate(divide="ignore", invalid="ignore"):
            best[start:start + chunk] = np.nanmax(np.where(union > 0, block / union, 0.0),
                                                  axis=1)

    weights = rows.astype(np.float64) / max(1.0, float(rows.sum()))
    stable, dissolved = best >= JACCARD_STABLE, best < JACCARD_DISSOLVED
    return {
        "jaccard_mean": round(float(best.mean()), 4),
        "jaccard_median": round(float(np.median(best)), 4),
        "jaccard_p10": round(float(np.percentile(best, 10)), 4),
        "clusters_stable": int(stable.sum()),
        "clusters_dissolved": int(dissolved.sum()),
        "frac_clusters_stable": round(float(stable.mean()), 4),
        "frac_clusters_dissolved": round(float(dissolved.mean()), 4),
        # The mass-weighted view. This is the number to quote.
        "point_share_stable": round(float(weights[stable].sum()), 4),
        "point_share_dissolved": round(float(weights[dissolved].sum()), 4),
    }


# ── loading ────────────────────────────────────────────────────────────────────

def discover(labels_dir: Path) -> list[dict]:
    found = []
    for path in sorted(labels_dir.glob("fit*_evalprobe_labels.npy")):
        match = LABEL_PATTERN.match(path.name)
        if not match:
            continue
        size, tag = int(match.group(1)), match.group(2)
        probability_path = path.with_name(path.name.replace("_labels.npy", "_probabilities.npy"))
        found.append({
            "fit_rows": size,
            "tag": tag,
            "name": f"{size / 1e6:g}M{tag}",
            "labels_path": path,
            "probabilities_path": probability_path if probability_path.exists() else None,
        })
    found.sort(key=lambda entry: (entry["fit_rows"], entry["tag"]))
    return found


def load(entry: dict) -> np.ndarray:
    return np.load(entry["labels_path"])


# ── report ─────────────────────────────────────────────────────────────────────

def compare(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None) -> dict:
    """One pair, under three views of the noise label."""
    if mask is not None:
        a, b = a[mask], b[mask]

    result: dict = {"rows_compared": int(a.size)}
    if a.size == 0:
        return result

    counts, rows, cols = contingency(a, b)
    result["ari_with_noise"] = round(adjusted_rand(counts, rows, cols), 4)

    # Noise is not a cluster -- it is the absence of one. A point that is noise in the small
    # fit and clustered in the large one is not a disagreement about cluster structure, it is
    # the larger fit having enough density to resolve it. Both views are reported.
    both = (a >= 0) & (b >= 0)
    result["both_clustered_fraction"] = round(float(both.mean()), 4)
    result["noise_a_only"] = round(float(((a < 0) & (b >= 0)).mean()), 4)
    result["noise_b_only"] = round(float(((a >= 0) & (b < 0)).mean()), 4)
    result["noise_both"] = round(float(((a < 0) & (b < 0)).mean()), 4)

    if both.sum() >= 2:
        c2, r2, k2 = contingency(a[both], b[both])
        result["ari"] = round(adjusted_rand(c2, r2, k2), 4)
        homogeneity, completeness = homogeneity_completeness(c2, r2, k2)
        result["homogeneity_a_to_b"] = round(homogeneity, 4)
        result["completeness_a_to_b"] = round(completeness, 4)
        result["purity_a_to_b"] = round(mapping_purity(c2, r2), 4)
        result["purity_b_to_a"] = round(mapping_purity(c2.T, k2), 4)
        vi, h_a_given_b, h_b_given_a = variation_of_information(c2, r2, k2)
        result["variation_of_information"] = round(vi, 4)
        result["h_b_given_a"] = round(h_b_given_a, 4)   # 0 <=> b is a clean merge of a
        result["h_a_given_b"] = round(h_a_given_b, 4)
        result["clusters_a"] = int(np.unique(a[both]).size)
        result["clusters_b"] = int(np.unique(b[both]).size)

        # Per-cluster reproducibility, both directions. Asymmetric on purpose: when b is a
        # coarsening of a, every a-cluster has a good match in b but not the reverse, and
        # the gap between the two directions is what says so.
        for direction, (cc, rr, kk) in (("a_to_b", (c2, r2, k2)),
                                        ("b_to_a", (c2.T, k2, r2))):
            for key, value in cluster_jaccard(cc, rr, kk).items():
                result[f"{key}_{direction}"] = value
    return result


def main() -> None:
    default_labels = Path(__file__).resolve().parent / "umap_tests" / "hdbscan_scaling" / "labels"
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels-dir", type=Path, default=default_labels)
    parser.add_argument("--output", type=Path, default=None,
                        help="JSON destination (default: cross_size_agreement.json beside labels/)")
    parser.add_argument("--min-probability", type=float, default=0.0,
                        help="restrict to probe rows whose probability is >= this in BOTH "
                             "labellings; disagreement at low probability is expected and "
                             "uninteresting, so this isolates whether confident points move")
    args = parser.parse_args()

    labels_dir: Path = args.labels_dir
    if not labels_dir.is_dir():
        raise SystemExit(f"no such directory: {labels_dir}")

    entries = discover(labels_dir)
    if len(entries) < 2:
        raise SystemExit(
            f"found {len(entries)} eval-probe labelling(s) in {labels_dir}; need at least 2. "
            "Cells whose predict step OOMed have no eval-probe labels by design."
        )

    print(f"eval-probe labellings found in {labels_dir}:")
    labellings, sizes = {}, {}
    for entry in entries:
        labels = load(entry)
        labellings[entry["name"]] = labels
        sizes[entry["name"]] = labels.size
        clustered = labels[labels >= 0]
        print(f"  {entry['name']:>10}  {labels.size:>10,} rows  "
              f"{np.unique(clustered).size:>6,} clusters  "
              f"{float((labels < 0).mean()) * 100:5.2f}% noise")

    if len(set(sizes.values())) != 1:
        raise SystemExit(
            "eval-probe labellings have different lengths: "
            f"{sizes}. They must all label the same probe rows -- a mismatch means the probe "
            "draw changed mid-sweep, and no cross-size comparison over them is meaningful."
        )

    names = [entry["name"] for entry in entries]

    mask = None
    if args.min_probability > 0:
        mask = np.ones(next(iter(labellings.values())).size, dtype=bool)
        missing = []
        for entry in entries:
            if entry["probabilities_path"] is None:
                missing.append(entry["name"])
                continue
            mask &= np.load(entry["probabilities_path"]) >= args.min_probability
        if missing:
            print(f"\nwarning: no probabilities saved for {', '.join(missing)}; "
                  "those are not filtered")
        print(f"\nprobability filter >= {args.min_probability}: "
              f"{mask.sum():,} of {mask.size:,} rows kept ({mask.mean() * 100:.1f}%)")

    pairs = {}
    for i, a_name in enumerate(names):
        for b_name in names[i + 1:]:
            pairs[f"{a_name} vs {b_name}"] = compare(
                labellings[a_name], labellings[b_name], mask)

    def matrix(key: str, title: str) -> None:
        print(f"\n{title}")
        header = "".join(f"{n:>10}" for n in names)
        print(f"{'':>10}{header}")
        for a_name in names:
            row = f"{a_name:>10}"
            for b_name in names:
                if a_name == b_name:
                    row += f"{'-':>10}"
                    continue
                key_forward, key_back = f"{a_name} vs {b_name}", f"{b_name} vs {a_name}"
                record = pairs.get(key_forward) or pairs.get(key_back)
                value = record.get(key) if record else None
                row += f"{value:>10.3f}" if value is not None else f"{'':>10}"
            print(row)

    matrix("ari", "ARI, noise excluded (points clustered in both)")

    # The ladder is the actual decision: each fit size against the next one up. If the last
    # rung is a clean coarsening, the larger fit bought nothing but hours.
    print("\nconsecutive ladder -- each size vs the next one up")
    print(f"{'pair':>22}{'ARI':>8}{'fine->coarse':>15}{'H(coarse|fine)':>16}"
          f"{'VI':>8}{'both clust.':>13}")
    for a_name, b_name in zip(names, names[1:]):
        record = pairs[f"{a_name} vs {b_name}"]
        if "ari" not in record:
            continue
        print(f"{a_name + ' -> ' + b_name:>22}"
              f"{record['ari']:>8.3f}"
              f"{record['purity_a_to_b']:>15.3f}"
              f"{record['h_b_given_a']:>16.4f}"
              f"{record['variation_of_information']:>8.3f}"
              f"{record['both_clustered_fraction'] * 100:>12.1f}%")

    print("\n  H(coarse|fine) = 0 : identity, not a threshold -- every cluster of the smaller")
    print("                       fit lies wholly inside one cluster of the larger. A clean")
    print("                       coarsening: the extra rows merged, they did not re-cut.")
    print("  H(coarse|fine) > 0 : boundaries genuinely moved. Read alongside fine->coarse,")
    print("                       which says how much of the mass moved (nats vs fraction).")
    print("  VI (nats)          : Meila's metric, H(a|b)+H(b|a). Comparable across the rows")
    print("                       of this table only -- it is unnormalised.")

    output = args.output or labels_dir.parent / "cross_size_agreement.json"
    output.write_text(json.dumps({
        "labels_dir": str(labels_dir),
        "min_probability": args.min_probability,
        "rows_total": int(next(iter(labellings.values())).size),
        "rows_after_filter": int(mask.sum()) if mask is not None else None,
        "labellings": [
            {"name": entry["name"], "fit_rows": entry["fit_rows"],
             "clusters": int(np.unique(labellings[entry["name"]][labellings[entry["name"]] >= 0]).size),
             "noise_fraction": round(float((labellings[entry["name"]] < 0).mean()), 5)}
            for entry in entries
        ],
        "pairs": pairs,
    }, indent=2))
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
