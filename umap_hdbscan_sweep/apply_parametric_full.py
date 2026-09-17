"""Push all 157.5 M deduplicated rows through a swept parametric encoder, then look at it.

The sweep scores encoders on a 500 K probe. That is the right size for measuring, and the
wrong size for *seeing*: a probe is 0.3% of the cohort, so anything rare enough to matter --
a small cluster, a thin filament, a satellite island -- is either invisible or indistinguishable
from noise at that sampling rate. This script does the thing the encoder was built to make
cheap: one forward pass over every row.

That is the whole argument for the parametric approach in one number. ``cuml.transform`` on
the full cohort was measured at 54.4 minutes; the encoder is a per-row function, so the same
work is a memory-bandwidth-bound streaming pass -- 10 GB of float32 read once. Nothing about
the output depends on batch size or ordering.

What comes out
--------------

Three artefacts per cell, in ``<output>/<cell>/``:

``density.png``     every row, as a log-scaled 2-D histogram. Not a scatter -- 157 M points
                    overplot into a solid blob, and the interesting question (is this space
                    separable?) is precisely a question about *density*, which a scatter
                    destroys and a histogram preserves. Log scale because occupancy spans
                    several orders of magnitude and a linear ramp shows only the largest island.
``clusters.png``    HDBSCAN labels over a subsample, noise in grey. Answers the question the
                    density plot raises: are the visible basins actually recovered as clusters?
``separability.json``  the numbers behind that picture -- cluster count, noise fraction, how
                    much mass the largest cluster holds, silhouette.

Two honesty notes
-----------------

**Clustering is on a subsample, embedding is not.** cuML's HDBSCAN does not survive much past
~5 M rows on this card, so ``--cluster-rows`` caps it. The density image is the full cohort;
the labels are an estimate from a sample of it. They answer different questions and the JSON
records which rows each used.

**Without ``--encoder-dir`` this refits and retrains.** The sweep discards its encoders unless
told otherwise, so by default this reconstructs one: same rows, same seed, same config, same
objective. That is reproducible in every respect except GPU non-determinism in training, which
was measured at ~0.002 on kNN-overlap between two runs of an identical cell -- immaterial for
looking at the shape of the space, but it does mean this is *an* encoder from that cell rather
than bitwise the one the sweep scored. Run the sweep with ``--encoder-dir`` to remove the
caveat entirely.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import sweep_core as core
import parametric_umap as pu
import parametric_sweep as ps


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


# ── choosing which cells to apply ──────────────────────────────────────────────

def select_cells(payload: dict, args) -> list[str]:
    """Successful cells, ranked by the metric that survived the sweep's own scrutiny.

    Ranking is on ``rnx_ratio`` -- how faithful the encoder's own map is to the 16-D latent,
    relative to cuML's map of the same rows -- and deliberately not on kNN-overlap against
    ``transform()``. The sweep established that distillation fidelity tops out far below the
    0.4141 determinism floor and cannot reach it at any attainable size, so ranking on it
    would just sort by which cell got closest to an unreachable target. ``rnx_ratio`` asks
    the question that stays meaningful: is this a good map, whether or not it is cuML's map.
    """
    summaries = payload["summaries"]
    usable = []
    for key, summary in summaries.items():
        if summary.get("failed_replicates") or summary.get("failure"):
            continue
        ratio = summary.get("rnx_ratio")
        if ratio is None:
            continue
        usable.append((key, ratio))

    if args.cells:
        wanted = [c.strip() for c in args.cells.split(",") if c.strip()]
        missing = [c for c in wanted if c not in summaries]
        if missing:
            raise SystemExit(f"no such cell(s) in the sweep JSON: {', '.join(missing)}")
        return wanted

    usable.sort(key=lambda item: item[1], reverse=True)
    if args.all:
        return [key for key, _ in usable]
    return [key for key, _ in usable[:args.top]]


def config_from_payload(payload: dict, label: str, seed: int) -> core.UmapConfig:
    spec = dict(payload["graph_configs"][label])
    spec["seed"] = seed
    return core.UmapConfig(**spec)


# ── rebuilding the encoder ─────────────────────────────────────────────────────

def load_or_train_encoder(payload, fit_rows, config, mode, latent, args, use_gpu, device):
    """Return a ParametricUmap for this cell, from disk if the sweep saved one."""
    import torch

    latent_dim = latent.shape[1]

    if args.encoder_dir:
        path = Path(args.encoder_dir) / f"{fit_rows}__{config.label()}__{mode}__seed{config.seed}.pt"
        if path.exists():
            blob = torch.load(path, map_location=device)
            encoder = pu.ParametricEncoder(
                latent_dim, output_dim=2, hidden=tuple(args.hidden)).to(device)
            encoder.load_state_dict(blob["state_dict"])
            log(f"    loaded saved encoder {path.name}")
            return pu.ParametricUmap(encoder=encoder, device=device, mode=mode), 0.0
        log(f"    no saved encoder at {path.name} -- refitting and retraining")

    # Reproduce the sweep's own row draw exactly: same probe seed, same nested prefixes,
    # same replicate. Anything else would embed the cohort through a different fit.
    total_rows = latent.shape[0]
    _, pool = ps.draw_probe(total_rows, payload["probe_rows"], args.seed)
    fit_sets = ps.draw_nested_fit_sets(pool, payload["sizes"], args.seed, args.replicate)
    fit_index = fit_sets[fit_rows]

    started = perf_counter()
    fit_latent = np.ascontiguousarray(latent[fit_index])
    fitted = core.fit_umap(fit_latent, config)
    log(f"    UMAP fit on {fit_rows:,} rows: {perf_counter() - started:.0f}s")

    fit_embedding = np.ascontiguousarray(fitted.embedding)
    needs_graph = mode in ("umap", "hybrid")
    edges = pu.extract_edges(fitted.model.graph_) if needs_graph else None
    negative_sample_rate = int(getattr(fitted.model, "negative_sample_rate", 5))
    del fitted
    if use_gpu:
        try:
            import cupy

            cupy.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    torch.manual_seed(config.seed)
    encoder = pu.ParametricEncoder(latent_dim, output_dim=2, hidden=tuple(args.hidden)).to(device)
    X = torch.from_numpy(fit_latent).to(device)
    Y = torch.from_numpy(fit_embedding).to(device)
    del fit_latent

    started = perf_counter()
    if mode in ("regress", "hybrid"):
        pu.train_regression(encoder, X, Y, steps=args.regress_steps,
                            batch_size=args.batch_size, learning_rate=args.learning_rate,
                            device=device)
    if mode in ("umap", "hybrid"):
        head_np, tail_np, weight_np = edges
        if len(head_np) > args.max_edges:
            keep = np.random.default_rng(config.seed).choice(
                len(head_np), size=args.max_edges, replace=False)
            keep.sort()
            head_np, tail_np, weight_np = head_np[keep], tail_np[keep], weight_np[keep]
        a, b = pu.ab_params(config.min_dist, spread=config.spread)
        pu.train_umap_loss(
            encoder, X,
            torch.from_numpy(head_np).to(device),
            torch.from_numpy(tail_np).to(device),
            # float64 on purpose: a float32 running sum saturates near 1.7e7 and most of a
            # large graph would never be drawn. Same reason as in train_umap_loss.
            torch.cumsum(torch.from_numpy(weight_np).to(device).double(), 0),
            a=a, b=b, steps=args.umap_steps, batch_size=args.batch_size,
            learning_rate=args.umap_learning_rate if mode == "hybrid" else args.learning_rate,
            negative_sample_rate=negative_sample_rate,
            repulsion_strength=config.repulsion_strength, device=device)
    train_seconds = perf_counter() - started
    log(f"    encoder ({mode}) trained: {train_seconds:.0f}s")

    del X, Y
    if use_gpu:
        torch.cuda.empty_cache()
    return pu.ParametricUmap(encoder=encoder, device=device, mode=mode), round(train_seconds, 1)


# ── the full pass ──────────────────────────────────────────────────────────────

def embed_cohort(model, latent, args, extent):
    """Stream every row through the encoder, accumulating a 2-D histogram.

    Coordinates are never all held at once unless ``--save-coords`` asks for them: the
    histogram is associative, so each batch scatter-adds and is dropped. Peak memory is one
    batch plus the bin grid, whatever the cohort size.
    """
    total = latent.shape[0]
    bins = args.bins
    hist = np.zeros((bins, bins), dtype=np.int64)
    (xlo, xhi), (ylo, yhi) = extent
    edges_x = np.linspace(xlo, xhi, bins + 1)
    edges_y = np.linspace(ylo, yhi, bins + 1)

    kept = [] if args.save_coords else None
    finite_total = 0
    clipped_total = 0
    started = perf_counter()
    for start in range(0, total, args.batch_rows):
        stop = min(start + args.batch_rows, total)
        chunk = np.ascontiguousarray(latent[start:stop], dtype=np.float32)
        coords = model.transform(chunk, batch_size=args.infer_batch_size)

        good = np.isfinite(coords).all(axis=1)
        coords = coords[good]
        finite_total += int(good.sum())

        if kept is not None:
            kept.append(coords.astype(np.float32))

        # The extent comes from a subsample, so a small tail of rows lands outside it.
        # histogram2d would drop those silently, leaving the image showing fewer rows than
        # the caption claims. Clip them onto the border instead and count them: the total
        # is then conserved, and the count says how much of the map the extent is cutting off.
        outside = (
            (coords[:, 0] < xlo) | (coords[:, 0] > xhi)
            | (coords[:, 1] < ylo) | (coords[:, 1] > yhi)
        )
        clipped_total += int(outside.sum())
        if outside.any():
            coords = coords.copy()
            np.clip(coords[:, 0], xlo, xhi, out=coords[:, 0])
            np.clip(coords[:, 1], ylo, yhi, out=coords[:, 1])

        block, _, _ = np.histogram2d(coords[:, 0], coords[:, 1], bins=[edges_x, edges_y])
        hist += block.astype(np.int64)

        if (start // args.batch_rows) % 10 == 0:
            done = stop / total
            elapsed = perf_counter() - started
            log(f"      {stop:,}/{total:,} ({done * 100:.0f}%) "
                f"eta {elapsed / max(done, 1e-9) * (1 - done) / 60:.1f} min")

    seconds = perf_counter() - started
    log(f"    full cohort embedded: {finite_total:,}/{total:,} finite in {seconds / 60:.1f} min"
        + (f" ({clipped_total:,} clipped to the extent border)" if clipped_total else ""))
    coords_all = np.concatenate(kept) if kept else None
    return hist, (edges_x, edges_y), finite_total, clipped_total, round(seconds, 1), coords_all


def probe_extent(model, latent, args):
    """Robust plot bounds from a subsample.

    Percentile rather than min/max: a handful of rows land far outside the body of the map
    and a min/max extent would spend the whole grid on empty space to accommodate them.
    """
    rng = np.random.default_rng(args.seed)
    index = np.sort(rng.choice(latent.shape[0], size=min(args.extent_rows, latent.shape[0]),
                               replace=False))
    coords = model.transform(np.ascontiguousarray(latent[index]), batch_size=args.infer_batch_size)
    coords = coords[np.isfinite(coords).all(axis=1)]
    lo, hi = args.extent_percentile, 100.0 - args.extent_percentile
    return (
        (float(np.percentile(coords[:, 0], lo)), float(np.percentile(coords[:, 0], hi))),
        (float(np.percentile(coords[:, 1], lo)), float(np.percentile(coords[:, 1], hi))),
    ), coords


# ── separability ───────────────────────────────────────────────────────────────

def assess_separability(model, latent, args, use_gpu):
    """HDBSCAN on a subsample, plus the statistics that say whether the split is real.

    A cluster count on its own is not evidence of structure -- HDBSCAN will always return
    something. The three numbers that matter are how much mass falls in noise, how much
    falls in the single largest cluster (a "clustering" that is one giant blob plus dust is
    not a clustering), and silhouette on the assignment.
    """
    rng = np.random.default_rng(args.seed + 7)
    n = min(args.cluster_rows, latent.shape[0])
    index = np.sort(rng.choice(latent.shape[0], size=n, replace=False))
    coords = model.transform(np.ascontiguousarray(latent[index]), batch_size=args.infer_batch_size)
    good = np.isfinite(coords).all(axis=1)
    coords = coords[good]

    started = perf_counter()
    config = core.HdbscanConfig(min_cluster_size=args.min_cluster_size,
                                min_samples=args.min_samples)
    labels = core.fit_hdbscan(coords, config, use_gpu=use_gpu).labels
    seconds = perf_counter() - started

    labels = np.asarray(labels)
    real = labels[labels >= 0]
    counts = np.bincount(real) if real.size else np.array([], dtype=np.int64)
    counts = np.sort(counts)[::-1]
    noise_fraction = float(np.mean(labels == -1))

    silhouette = None
    if counts.size >= 2:
        try:
            from sklearn.metrics import silhouette_score

            sub = rng.choice(len(coords), size=min(args.silhouette_rows, len(coords)),
                             replace=False)
            if len(set(labels[sub].tolist())) >= 2:
                silhouette = round(float(silhouette_score(coords[sub], labels[sub])), 5)
        except Exception as error:
            log(f"    silhouette skipped: {error}")

    return {
        "cluster_rows": int(len(coords)),
        "n_clusters": int(counts.size),
        "noise_fraction": round(noise_fraction, 5),
        "largest_cluster_fraction": round(float(counts[0] / len(coords)), 5) if counts.size else None,
        "top10_cluster_sizes": [int(c) for c in counts[:10]],
        "clustered_fraction": round(float(real.size / len(coords)), 5),
        "silhouette": silhouette,
        "hdbscan_seconds": round(seconds, 1),
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
    }, coords, labels


# ── pictures ───────────────────────────────────────────────────────────────────

def plot_density(hist, edges, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    edges_x, edges_y = edges
    # Empty bins must be masked, not zero: LogNorm has no value for 0, and leaving them in
    # drags the colour floor down so the sparse structure between islands disappears.
    shown = np.ma.masked_where(hist.T == 0, hist.T)

    figure, axes = plt.subplots(figsize=(11, 10), dpi=160)
    mesh = axes.pcolormesh(edges_x, edges_y, shown, norm=LogNorm(), cmap="magma", shading="auto")
    figure.colorbar(mesh, ax=axes, label="rows per bin (log)")
    axes.set_title(title, fontsize=10)
    axes.set_xlabel("dim 1")
    axes.set_ylabel("dim 2")
    axes.set_aspect("equal")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def plot_clusters(coords, labels, path, title, max_points=400_000, seed=42):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    if len(coords) > max_points:
        pick = rng.choice(len(coords), size=max_points, replace=False)
        coords, labels = coords[pick], labels[pick]

    figure, axes = plt.subplots(figsize=(11, 10), dpi=160)
    noise = labels == -1
    axes.scatter(coords[noise, 0], coords[noise, 1], s=0.4, c="lightgrey",
                 linewidths=0, label=f"noise ({noise.mean() * 100:.0f}%)")
    real = ~noise
    if real.any():
        axes.scatter(coords[real, 0], coords[real, 1], s=0.4, c=labels[real],
                     cmap="tab20", linewidths=0)
    axes.set_title(title, fontsize=10)
    axes.set_xlabel("dim 1")
    axes.set_ylabel("dim 2")
    axes.set_aspect("equal")
    axes.legend(markerscale=20, loc="upper right", fontsize=8)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


# ── entry point ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed the full cohort through swept parametric encoders and picture it")
    parser.add_argument("--sweep-json", required=True,
                        help="parametric_sweep.json or parametric_sweep_partial.json")
    parser.add_argument("--embed-dir", required=True, help="stage 1 output holding latent.npy")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--encoder-dir", default=None,
                        help="encoders saved by parametric_sweep.py --encoder-dir; without "
                             "this each cell is refit and retrained")

    parser.add_argument("--top", type=int, default=4,
                        help="how many cells to apply, best rnx_ratio first")
    parser.add_argument("--all", action="store_true", help="apply every successful cell")
    parser.add_argument("--cells", default=None,
                        help="explicit comma-separated cell keys, e.g. "
                             "'10000000|nn15_md0.25_nc2|umap'")
    parser.add_argument("--replicate", type=int, default=0,
                        help="which replicate's row draw and seed to reproduce")

    parser.add_argument("--bins", type=int, default=2000, help="density grid resolution")
    parser.add_argument("--extent-rows", type=int, default=2_000_000)
    parser.add_argument("--extent-percentile", type=float, default=0.05)
    parser.add_argument("--batch-rows", type=int, default=5_000_000)
    parser.add_argument("--infer-batch-size", type=int, default=2_000_000)
    parser.add_argument("--save-coords", action="store_true",
                        help="also write the full 157.5M x 2 float32 array (~1.3 GB per cell)")

    parser.add_argument("--density-only", action="store_true",
                        help="just the embedding: write density.png and stop. Skips HDBSCAN "
                             "and clusters.png. Use this to look at the space before "
                             "committing to a clusterer's opinion of it")
    parser.add_argument("--cluster-rows", type=int, default=5_000_000,
                        help="cuML HDBSCAN does not survive much past this on one card")
    parser.add_argument("--min-cluster-size", type=int, default=250)
    parser.add_argument("--min-samples", type=int, default=25)
    parser.add_argument("--silhouette-rows", type=int, default=50_000)

    parser.add_argument("--hidden", default="256,256,128")
    parser.add_argument("--regress-steps", type=int, default=15_000)
    parser.add_argument("--umap-steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--umap-learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-edges", type=int, default=150_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-budget-gb", type=float, default=None)
    parser.add_argument("--rmm-share", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.hidden = tuple(int(p) for p in args.hidden.split(",") if p.strip())

    payload = json.loads(Path(args.sweep_json).read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_gpu = core.gpu_available()
    if use_gpu and args.gpu_budget_gb is not None:
        core.apply_gpu_budget("apply", budget_gb=args.gpu_budget_gb, rmm_share=args.rmm_share)

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    latent_path = Path(args.embed_dir) / "latent.npy"
    latent = np.load(latent_path, mmap_mode="r")
    log(f"latent.npy: {latent.shape[0]:,} rows x {latent.shape[1]} dims, device {device}")

    if latent.shape[0] != payload["total_rows"]:
        log(f"WARNING: latent has {latent.shape[0]:,} rows but the sweep recorded "
            f"{payload['total_rows']:,}. Row draws will not match the sweep's.")

    selected = select_cells(payload, args)
    log(f"applying {len(selected)} cell(s): {', '.join(selected)}")

    index = []
    for position, key in enumerate(selected, 1):
        fit_rows_text, label, mode = key.split("|")
        fit_rows = int(fit_rows_text)
        config = config_from_payload(payload, label, args.seed + args.replicate)
        cell_dir = output_dir / key.replace("|", "__")
        cell_dir.mkdir(parents=True, exist_ok=True)
        log(f"=== [{position}/{len(selected)}] {key} ===")

        model, train_seconds = load_or_train_encoder(
            payload, fit_rows, config, mode, latent, args, use_gpu, device)

        extent, _ = probe_extent(model, latent, args)
        log(f"    extent x {extent[0][0]:.2f}..{extent[0][1]:.2f}  "
            f"y {extent[1][0]:.2f}..{extent[1][1]:.2f}")

        hist, edges, finite, clipped, embed_seconds, coords_all = embed_cohort(
            model, latent, args, extent)
        if coords_all is not None:
            np.save(cell_dir / "coords.npy", coords_all)
            log(f"    wrote coords.npy ({coords_all.nbytes / 1e9:.2f} GB)")
            del coords_all

        summary = payload["summaries"][key]
        title = (f"{key}   rnx_ratio {summary.get('rnx_ratio')}   "
                 f"{finite:,} rows")
        plot_density(hist, edges, cell_dir / "density.png", f"full cohort density -- {title}")
        log(f"    wrote density.png")

        if args.density_only:
            separability = None
        else:
            separability, coords_sub, labels_sub = assess_separability(
                model, latent, args, use_gpu)
            plot_clusters(coords_sub, labels_sub, cell_dir / "clusters.png",
                          f"HDBSCAN on {separability['cluster_rows']:,} rows -- {key}",
                          seed=args.seed)
            log(f"    wrote clusters.png -- {separability['n_clusters']} clusters, "
                f"noise {separability['noise_fraction'] * 100:.1f}%, "
                f"largest {(separability['largest_cluster_fraction'] or 0) * 100:.1f}%")

        record = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "cell": key,
            "fit_rows": fit_rows,
            "umap_config": payload["graph_configs"][label],
            "mode": mode,
            "replicate": args.replicate,
            "encoder_source": "loaded" if train_seconds == 0.0 else "refit_and_retrained",
            "cohort_rows": int(latent.shape[0]),
            "finite_rows": int(finite),
            "clipped_to_extent_rows": int(clipped),
            "timing": {"encoder_train_seconds": train_seconds,
                       "full_embed_seconds": embed_seconds},
            "sweep_scores": {
                "rnx_ratio": summary.get("rnx_ratio"),
                "knn_overlap_k15": summary.get("distillation", {}).get("knn_overlap_k15"),
                "vs_cuml_adjusted_rand": summary.get("clustering", {}).get(
                    "vs_cuml_adjusted_rand"),
                "reproducibility": summary.get("reproducibility"),
            },
            "separability": separability,
            "density_extent": {"x": list(extent[0]), "y": list(extent[1]),
                               "bins": args.bins},
        }
        (cell_dir / ("embedding.json" if args.density_only else "separability.json")).write_text(
            json.dumps(record, indent=2))
        index.append(record)

        del model
        if use_gpu:
            torch.cuda.empty_cache()

    (output_dir / "index.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "sweep_json": str(args.sweep_json),
        "cells": index,
    }, indent=2))

    print("\n" + "=" * 96)
    if args.density_only:
        print("FULL-COHORT EMBEDDING -- density only, no clustering")
        print("=" * 96)
        print(f"  {'cell':<42}{'rows':>15}{'clipped':>10}{'embed_min':>11}")
        for record in index:
            print(f"  {record['cell']:<42}{record['finite_rows']:>15,}"
                  f"{record['clipped_to_extent_rows']:>10,}"
                  f"{record['timing']['full_embed_seconds'] / 60:>11.1f}")
        print()
        print("  density.png is every row as a log-scaled histogram. Look for basins separated")
        print("  by genuinely empty space -- a low-density bridge between two lobes means a")
        print("  clusterer will have to choose, and different ones will choose differently.")
        print("  Rerun without --density-only to get HDBSCAN's answer.")
    else:
        print("FULL-COHORT APPLY -- is the 2-D space separable?")
        print("=" * 96)
        print(f"  {'cell':<42}{'clusters':>9}{'noise':>8}{'largest':>9}{'silh':>8}{'embed_min':>11}")
        for record in index:
            s = record["separability"]
            print(f"  {record['cell']:<42}{s['n_clusters']:>9}"
                  f"{s['noise_fraction'] * 100:>7.1f}%"
                  f"{(s['largest_cluster_fraction'] or 0) * 100:>8.1f}%"
                  f"{(s['silhouette'] if s['silhouette'] is not None else float('nan')):>8.3f}"
                  f"{record['timing']['full_embed_seconds'] / 60:>11.1f}")
        print()
        print("  Read it as: many clusters with low noise and a largest-cluster share well under")
        print("  half means the space genuinely splits. One dominant cluster plus high noise means")
        print("  HDBSCAN found a blob and some dust, whatever the cluster count says.")
    print("=" * 96 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
