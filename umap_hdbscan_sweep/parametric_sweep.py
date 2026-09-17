"""Sweep 3: parametric UMAP across fit size x graph parameters x training objective.

Sweeps 1 and 2 asked which *non-parametric* UMAP settings are best. Both inherit a problem
neither can fix: UMAP's output depends on which rows were drawn for the fit, and the only
way to remove that dependence is to fit on everything, which does not fit on this card.
Gate 1 measured the size of the problem directly -- two fits of the same 5 M rows differing
only in seed agree at **kNN-overlap 0.4141**. That number is the floor on how reproducible
anything built this way can be.

A parametric encoder changes the shape of the problem rather than the size of it. The
network is a *function*, so once trained, embedding a row is a deterministic forward pass:
the same read gets the same coordinates on every future run, and the whole cohort can be
embedded without ``transform()``. The row-draw dependence does not vanish -- it moves into
training, where it is asked to do something much weaker, namely produce the same *function*
rather than the same *point positions*. Whether it actually does is measurable, and family E
below is what measures it.

Nothing here re-implements the trainer. ``parametric_umap.py`` already holds the encoder,
both objectives and the loss port; this drives it over a grid and scores the output.


What is measured, in six families
---------------------------------

**A. Distillation fidelity** -- does the network reproduce the UMAP fit it was trained on,
on rows neither of them saw? ``knn_overlap`` at k = 15 / 50 / 200 against cuML's own
``transform()`` of the same held-out probe. Read it against two reference lines, both
measured on this data, not assumed:

    0.4141   Gate 1 determinism floor: two different-seed cuML fits scored against each
             other. At or above this the encoder disagrees with ``transform()`` no more
             than re-running UMAP disagrees with itself, which is the honest bar.
    0.4400   aUMAP, the kNN-interpolation alternative, already measured and rejected.
    0.0554   regress-only parametric at min_dist=0.0, already measured -- a total failure
             that still produced a *small coordinate error*, which is why coordinate error
             is reported here as a diagnostic and never ranked on.

**B. Intrinsic map quality** -- is the encoder's own 2-D output a faithful picture of the
16-D latent, judged without reference to cuML at all? Trustworthiness, continuity, RNX AUC
and Q_NX from ``umap_metrics``, computed on the encoder's coordinates *and* on cuML's, so
the pair can be compared. ``rnx_ratio = rnx_net / rnx_cuml`` is the number this family
exists for: at 1.0 the encoder's map is exactly as faithful to the input space as UMAP's
own, **even where the two disagree about coordinates**. Family A can only ever say how
closely the encoder imitates one particular stochastic fit; this says whether it needs to.

**C. Downstream clustering** -- HDBSCAN on the encoder's embedding versus HDBSCAN on
cuML's, compared by ARI/NMI/AMI/Jaccard, with cluster count and noise fraction for both.
``parametric_umap``'s own docstring makes the point this family answers: neighbourhood
overlap is a proxy for whether clustering moved, not a substitute for checking.

**D. Generalisation gap** -- the extrapolation question, stated as a number. The same
fidelity metrics are computed twice, once on rows the encoder trained on and once on the
held-out probe; ``gap = in_sample - out_of_sample``. Near zero means the encoder learned a
function that transfers. A large gap means it memorised its training rows, which is exactly
the failure more data is supposed to fix -- so this is the metric the size axis is really
sweeping, and it is what makes "more rows help" a measurement rather than an assumption.

**E. Cross-replicate reproducibility** -- the reason any of this is being built. Each
replicate redraws the fit rows *and* every seed, trains an independent encoder, and embeds
the **same** probe. Reported as mean pairwise kNN-overlap and mean pairwise HDBSCAN ARI
between replicates. These are directly comparable to sweep 1's non-parametric seed ARI
(0.25 at min_dist=0.0 up to 0.63 at 0.25) and to the 0.4141 floor. If the parametric numbers
do not clear the non-parametric ones, the encoder has bought speed and nothing else, and
that is a publishable answer rather than a failed run.

**F. Cost** -- fit, train and inference time, with inference extrapolated to the full cohort.


Design notes
------------

**Fit sets are nested across sizes.** Within a replicate the 5 M set contains the 2 M set,
so a change between sizes is more data and never different data. Across replicates
everything is redrawn, which is what family E needs.

**The UMAP fit is shared by every training objective.** ``regress``, ``umap`` and ``hybrid``
all learn from one fitted graph, so the expensive half is paid once per
(size, min_dist, n_neighbors) cell rather than once per mode. That is what keeps a
three-objective grid affordable.

**Step budget is a curve, not an axis.** ``--checkpoint-every`` traces kNN-overlap during
training on a small subsample, so "were 30k steps enough" is answered from the shape of that
curve instead of by re-running the grid at several budgets.

**Edge lists are capped.** A 25 M-row graph at ``n_neighbors=50`` carries ~6e8 undirected
edges; head, tail and a float64 cumulative weight would be ~13 GB of device memory on top of
cuML's own footprint. ``--max-edges`` takes a uniform random subset above that, which leaves
the within-subset sampling distribution proportional to edge weight exactly as before.

**regress is kept despite being known-bad.** It scored 0.055 at min_dist=0.0 and costs ~10 s,
and the interesting question is whether that failure is specific to min_dist=0.0 -- where
UMAP packs each island so tightly that an MSE fit lands every point near its cluster centre.
At min_dist=0.25 the islands are not packed that way. A cheap control that might change sign
across the grid earns its place.

    python umap_hdbscan_sweep/parametric_sweep.py \\
        --embed-dir <stage1-output> \\
        --output-dir umap_hdbscan_sweep/umap_tests/parametric_sweep \\
        --fit-rows 2000000,5000000,10000000,25000000 \\
        --min-dist 0.0,0.1,0.25 --n-neighbors 15,30,50 \\
        --modes regress,umap,hybrid --seeds 3 --gpu-budget-gb 40

Planning a run from a completed trial, without touching the GPU::

    python umap_hdbscan_sweep/parametric_sweep.py --plan-from <trial>/parametric_sweep.json \\
        --fit-rows 2000000,5000000,10000000,25000000 \\
        --min-dist 0.0,0.1,0.25 --n-neighbors 15,30,50 --modes regress,umap,hybrid --seeds 3
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from itertools import combinations
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
import umap_metrics as metrics
import parametric_umap as pu

# Reference lines, all measured on this cohort rather than assumed. See the module docstring.
DETERMINISM_FLOOR = 0.4141      # gate1_determinism.json -> determinism_floor_knn_overlap
LATENT_PRESERVATION = 0.16398   # gate1: how much 16-D neighbourhood UMAP itself keeps at k=15
AUMAP_BASELINE = 0.440          # aumap_validation.json, already rejected
REGRESS_MD0_FAILURE = 0.05542   # parametric_umap_regress.json, the known catastrophic case

# Sweep 1's non-parametric seed ARI, for the family E comparison. Range across its 9 configs.
NONPARAMETRIC_SEED_ARI = {"min": 0.25279, "max": 0.62608, "at_md0.1": 0.41851}

FULL_COHORT_HELD_ROWS = pu.FULL_COHORT_HELD_ROWS

OVERLAP_KS = (15, 50, 200)


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def parse_ints(text: str) -> list[int]:
    return [int(float(part)) for part in text.split(",") if part.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def cell_key(fit_rows: int, umap_label: str, mode: str) -> str:
    """Exact row count in the key, never a rounded one.

    An earlier version wrote ``fit_rows // 1_000_000`` here, which collapses any two sizes
    inside the same million to one key -- and because cells are accumulated into a dict by
    this key, the two sizes would have been silently *averaged together* rather than
    reported separately. Formatting for humans belongs in the report, not in the identity.
    """
    return f"{fit_rows}|{umap_label}|{mode}"


def format_rows(fit_rows: int) -> str:
    """Human-readable row count for the printed table only."""
    if fit_rows >= 1_000_000:
        millions = fit_rows / 1e6
        return f"{millions:.0f}M" if millions == int(millions) else f"{millions:g}M"
    if fit_rows >= 1_000:
        return f"{fit_rows / 1e3:g}k"
    return str(fit_rows)


# ── row selection ──────────────────────────────────────────────────────────────

def draw_probe(total_rows: int, probe_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """One fixed probe held out of every fit, plus the pool the fit sets come from.

    Identical rows are scored for every cell and every replicate, so a difference in any
    metric is a difference in the embedding and nothing else.
    """
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(total_rows)
    return np.sort(permutation[:probe_rows]), permutation[probe_rows:]


def draw_nested_fit_sets(
    pool: np.ndarray, sizes: list[int], seed: int, replicate: int
) -> dict[int, np.ndarray]:
    """One shuffle per replicate, prefixes of it per size -- so the sizes are nested.

    Nesting isolates the size axis: the 5 M set is the 2 M set plus 3 M more rows, never a
    different 5 M rows. Across replicates the shuffle changes, which is the variation
    family E is built to measure.
    """
    rng = np.random.default_rng(seed + 1000 + replicate)
    order = rng.permutation(pool.size)
    return {size: np.sort(pool[order[:size]]) for size in sizes}


# ── metric helpers ─────────────────────────────────────────────────────────────

def overlap_profile(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    ks: tuple[int, ...],
    use_gpu: bool,
) -> dict[str, float]:
    """Mean shared-k-nearest-neighbour fraction between two embeddings of the same rows.

    ``aumap.knn_overlap`` computes this with a Python loop over ``np.intersect1d``, which is
    fine at its own scale and far too slow at 200k rows. This reuses ``umap_metrics``'
    GPU neighbour search and its batched rank lookup instead: ``neighbour_ranks`` already
    returns ``k_reference`` for entries absent from the reference list, so the shared
    fraction is just how often the returned rank is inside the window.

    One search at ``max(ks)`` serves every k, because neighbour lists come back ordered by
    distance and a prefix of width k is exactly the k-NN result.
    """
    widest = max(ks)
    nn_a = metrics.knn_indices(coords_a, widest, use_gpu=use_gpu)
    nn_b = metrics.knn_indices(coords_b, widest, use_gpu=use_gpu)
    n_points = coords_a.shape[0]
    result = {}
    for k in ks:
        ranks = metrics.neighbour_ranks(
            np.ascontiguousarray(nn_a[:, :k]),
            np.ascontiguousarray(nn_b[:, :k]),
            n_points=n_points,
        )
        result[f"knn_overlap_k{k}"] = round(float((ranks < k).mean()), 5)
    return result


def coordinate_error(true_coords: np.ndarray, approx_coords: np.ndarray) -> dict:
    """Procrustes-aligned coordinate error, in units of the reference embedding's spread.

    Reported, never ranked on. The regress-mode failure scored 0.055 embedding std here --
    which reads as excellent -- while its kNN-overlap was 0.055, i.e. the neighbourhood
    structure was destroyed. Coordinate accuracy and neighbour accuracy come apart on this
    data, and only the second one matters to HDBSCAN.
    """
    aligned = pu.procrustes_align(approx_coords, true_coords)
    errors = np.linalg.norm(true_coords - aligned, axis=1)
    embedding_std = float(np.std(true_coords))
    return {
        "mean_error_in_std": round(float(np.mean(errors)) / embedding_std, 5),
        "p95_error_in_std": round(float(np.percentile(errors, 95)) / embedding_std, 5),
        "procrustes_disparity": round(metrics.procrustes_disparity(approx_coords, true_coords), 5),
    }


def map_quality(
    high: np.ndarray, low: np.ndarray, args, use_gpu: bool
) -> dict:
    """Trustworthiness / continuity / RNX AUC of one 2-D map against the 16-D latent.

    No reference embedding involved, which is the point: this scores the encoder's output on
    its own merits, so an encoder that disagrees with cuML can still be shown to be an
    equally good map of the same data.
    """
    return metrics.local_quality(
        high, low, k=args.metric_k, k_max=args.metric_k_max, use_gpu=use_gpu
    )


def cluster_and_compare(
    net_coords: np.ndarray,
    reference_labels: np.ndarray,
    args,
    use_gpu: bool,
) -> tuple[dict, np.ndarray]:
    """HDBSCAN the encoder's embedding and compare its labels to the reference run's."""
    config = core.HdbscanConfig(
        min_cluster_size=args.min_cluster_size, min_samples=args.min_samples)
    clustered = core.fit_hdbscan(net_coords, config, use_gpu=use_gpu)
    labels = clustered.labels
    payload = {
        "n_clusters": int(len(set(labels.tolist()) - {-1})),
        "noise_fraction": round(float(np.mean(labels == -1)), 5),
        "vs_cuml": metrics.label_agreement(reference_labels, labels),
    }
    return payload, labels


# ── one (size, umap config, replicate) cell ────────────────────────────────────

def run_cell(
    fit_rows: int,
    config: core.UmapConfig,
    latent: np.ndarray,
    fit_index: np.ndarray,
    probe_latent: np.ndarray,
    tiers: dict,
    args,
    use_gpu: bool,
    device,
) -> dict:
    """Fit UMAP once, then train and score every requested objective against that one fit.

    Ordering is chosen to keep peak device memory down: everything cuML produced -- the
    embedding, the probe transform, the edge list -- is pulled to the host and the fitted
    model is dropped *before* torch allocates anything. cuML and torch would otherwise hold
    a 25 M-row graph simultaneously.
    """
    import torch

    timing = {}
    started = perf_counter()
    fit_latent = np.ascontiguousarray(latent[fit_index])
    timing["gather_seconds"] = round(perf_counter() - started, 1)

    started = perf_counter()
    fitted = core.fit_umap(fit_latent, config)
    timing["umap_fit_seconds"] = round(perf_counter() - started, 1)

    started = perf_counter()
    cuml_probe = fitted.transform(probe_latent, batch_size=args.transform_batch_size)
    timing["probe_transform_seconds"] = round(perf_counter() - started, 1)

    if not np.isfinite(cuml_probe).all():
        bad = int((~np.isfinite(cuml_probe)).any(axis=1).sum())
        log(f"    FAILED: {bad:,}/{len(cuml_probe):,} probe rows non-finite from cuML transform")
        del fitted
        return {"failed": f"{bad} non-finite probe rows from cuML transform", "timing": timing,
                "modes": {}}

    # Everything needed from cuML, on the host, before torch touches the card.
    fit_embedding = np.ascontiguousarray(fitted.embedding)
    needs_graph = any(mode in ("umap", "hybrid") for mode in args.modes)
    edges = pu.extract_edges(fitted.model.graph_) if needs_graph else None
    negative_sample_rate = (
        args.negative_sample_rate
        if args.negative_sample_rate is not None
        else int(getattr(fitted.model, "negative_sample_rate", 5))
    )
    del fitted
    if use_gpu:
        try:
            import cupy

            cupy.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    # ── reference scores for the fit being distilled ──────────────────────────
    started = perf_counter()
    cuml_map = map_quality(
        probe_latent[tiers["knn"]], cuml_probe[tiers["knn"]], args, use_gpu)
    hdbscan_config = core.HdbscanConfig(
        min_cluster_size=args.min_cluster_size, min_samples=args.min_samples)
    cuml_clustered = core.fit_hdbscan(cuml_probe[tiers["cluster"]], hdbscan_config, use_gpu=use_gpu)
    cuml_labels = cuml_clustered.labels
    timing["reference_metric_seconds"] = round(perf_counter() - started, 1)

    if cuml_map["clamped_fraction"] > 0.25:
        log(f"    note: clamped_fraction {cuml_map['clamped_fraction'] * 100:.0f}% at "
            f"k_max={cuml_map['k_max']} -- on this latent that is mostly genuine distortion "
            f"(gate 1 put latent-vs-embedding overlap at {LATENT_PRESERVATION:.3f}), not a "
            f"sampling artefact, so trustworthiness stays comparable across cells")

    # ── torch side ────────────────────────────────────────────────────────────
    latent_dim = fit_latent.shape[1]
    X = torch.from_numpy(fit_latent).to(device)
    Y = torch.from_numpy(fit_embedding).to(device)
    del fit_latent

    edge_tensors = None
    if needs_graph:
        head_np, tail_np, weight_np = edges
        n_edges = len(head_np)
        if n_edges > args.max_edges:
            # Uniform subset: within it, draws stay proportional to edge weight exactly as
            # before, so the objective is unchanged -- only the number of distinct edges
            # available to sample from falls.
            keep = np.random.default_rng(config.seed).choice(
                n_edges, size=args.max_edges, replace=False)
            keep.sort()
            head_np, tail_np, weight_np = head_np[keep], tail_np[keep], weight_np[keep]
            log(f"    edge list capped {n_edges:,} -> {len(head_np):,} (--max-edges)")
        edge_tensors = (
            torch.from_numpy(head_np).to(device),
            torch.from_numpy(tail_np).to(device),
            # float64: see train_umap_loss. A float32 running sum stops advancing past
            # ~1.7e7 and most of the graph would never be drawn.
            torch.cumsum(torch.from_numpy(weight_np).to(device).double(), 0),
        )
        del edges, head_np, tail_np, weight_np

    a, b = pu.ab_params(config.min_dist, spread=config.spread)

    # A small fixed slice used only for the mid-training curve. Kept tiny on purpose --
    # it runs many times per cell and must not cost more than the training it observes.
    curve_positions = tiers["curve"]
    curve_high = probe_latent[curve_positions]
    curve_reference = cuml_probe[curve_positions]

    # Family D scores the encoder on rows it trained on, so the tier arrives as row *ids*
    # rather than positions. Fit sets are sorted, so a fixed position range would instead
    # pick the lowest-row-index rows of every set -- the whole 2 M set at size 2 M but only
    # the earliest fifth of a 10 M one, an ever-narrower slice of the cohort as the fit
    # grows. That would confound the size axis family D exists to measure. Resolving ids
    # against this fit set scores the identical physical rows at every size instead.
    in_ids = tiers["in_sample"]
    located = np.searchsorted(fit_index, in_ids)
    inside = located < len(fit_index)
    inside[inside] &= fit_index[located[inside]] == in_ids[inside]
    in_positions = located[inside]
    if in_positions.size != in_ids.size:
        log(f"    note: {in_ids.size - in_positions.size:,}/{in_ids.size:,} in-sample rows "
            f"are not in this fit set -- family D scores the remainder")

    mode_results: dict[str, dict] = {}
    for mode in args.modes:
        torch.manual_seed(config.seed)
        encoder = pu.ParametricEncoder(
            latent_dim, output_dim=2, hidden=args.hidden).to(device)
        model = pu.ParametricUmap(encoder=encoder, device=device, mode=mode)

        curve: list[dict] = []

        def on_checkpoint(step: int, current, _mode=mode, _curve=curve):
            probe_coords = pu.ParametricUmap(
                encoder=current, device=device, mode=_mode
            ).transform(curve_high, batch_size=args.infer_batch_size)
            if not np.isfinite(probe_coords).all():
                _curve.append({"step": step, "knn_overlap_k15": None})
                return
            shared = overlap_profile(curve_reference, probe_coords, (15,), use_gpu)
            _curve.append({"step": step, "knn_overlap_k15": shared["knn_overlap_k15"]})

        hook = on_checkpoint if args.checkpoint_every else None
        started = perf_counter()
        regress_history: list[float] = []
        umap_history: list[float] = []

        if mode in ("regress", "hybrid"):
            regress_history = pu.train_regression(
                encoder, X, Y, steps=args.regress_steps, batch_size=args.batch_size,
                learning_rate=args.learning_rate, device=device,
                checkpoint_every=args.checkpoint_every, on_checkpoint=hook,
            )
        if mode in ("umap", "hybrid"):
            learning_rate = args.umap_learning_rate if mode == "hybrid" else args.learning_rate
            umap_history = pu.train_umap_loss(
                encoder, X, *edge_tensors, a=a, b=b,
                steps=args.umap_steps, batch_size=args.batch_size,
                learning_rate=learning_rate,
                negative_sample_rate=negative_sample_rate,
                repulsion_strength=config.repulsion_strength, device=device,
                checkpoint_every=args.checkpoint_every, on_checkpoint=hook,
            )
        train_seconds = perf_counter() - started

        # ── inference on the probe ────────────────────────────────────────────
        started = perf_counter()
        net_probe = model.transform(probe_latent, batch_size=args.infer_batch_size)
        infer_seconds = perf_counter() - started

        # Persist the encoder so downstream work (embedding the full cohort) does not have
        # to refit UMAP and retrain to get this exact network back. Off by default: 108
        # cells x 3 modes x ~400 KB is small, but only some runs want it.
        if args.encoder_dir is not None:
            encoder_path = (
                Path(args.encoder_dir)
                / f"{fit_rows}__{config.label()}__{mode}__seed{config.seed}.pt")
            encoder_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(encoder_path)

        if not np.isfinite(net_probe).all():
            bad = int((~np.isfinite(net_probe)).any(axis=1).sum())
            log(f"      {mode}: FAILED, {bad:,} non-finite rows")
            mode_results[mode] = {
                "failed": f"{bad} non-finite probe rows from the encoder",
                "timing": {"encoder_train_seconds": round(train_seconds, 1)},
                "_labels": None,
                "_probe_coords": None,
            }
            continue

        started = perf_counter()
        distillation = {
            **overlap_profile(
                cuml_probe[tiers["overlap"]], net_probe[tiers["overlap"]], OVERLAP_KS, use_gpu),
            **coordinate_error(
                cuml_probe[tiers["overlap"]], net_probe[tiers["overlap"]]),
        }
        net_map = map_quality(
            probe_latent[tiers["knn"]], net_probe[tiers["knn"]], args, use_gpu)
        clustering, net_labels = cluster_and_compare(
            net_probe[tiers["cluster"]], cuml_labels, args, use_gpu)

        # ── family D: the same comparison on rows the encoder trained on ──────
        if in_positions.size:
            in_latent = np.ascontiguousarray(latent[fit_index[in_positions]])
            net_in = model.transform(in_latent, batch_size=args.infer_batch_size)
            cuml_in = fit_embedding[in_positions]
            in_overlap = overlap_profile(cuml_in, net_in, (args.metric_k,), use_gpu)
            in_sample_k15 = in_overlap[f"knn_overlap_k{args.metric_k}"]
            in_map = map_quality(in_latent, net_in, args, use_gpu)
            del in_latent, net_in
        else:
            in_sample_k15, in_map = None, {}
        metric_seconds = perf_counter() - started

        out_sample_k15 = distillation.get(f"knn_overlap_k{args.metric_k}")
        generalisation = {
            "knn_overlap_in_sample": in_sample_k15,
            "knn_overlap_out_of_sample": out_sample_k15,
            "knn_overlap_gap": (
                round(in_sample_k15 - out_sample_k15, 5)
                if in_sample_k15 is not None and out_sample_k15 is not None else None
            ),
            "rnx_auc_in_sample": in_map.get("rnx_auc"),
            "rnx_auc_out_of_sample": net_map.get("rnx_auc"),
            "rnx_auc_gap": (
                round(in_map["rnx_auc"] - net_map["rnx_auc"], 5)
                if in_map.get("rnx_auc") is not None and net_map.get("rnx_auc") is not None
                else None
            ),
        }

        rnx_net, rnx_cuml = net_map.get("rnx_auc"), cuml_map.get("rnx_auc")
        mode_results[mode] = {
            "distillation": distillation,
            "map_quality_net": net_map,
            "rnx_ratio": (
                round(rnx_net / rnx_cuml, 5) if rnx_net is not None and rnx_cuml else None),
            "clustering": clustering,
            "generalisation": generalisation,
            "step_curve": curve,
            "loss_history": {"regression_mse": regress_history, "umap_ce": umap_history},
            "timing": {
                "encoder_train_seconds": round(train_seconds, 1),
                "probe_inference_seconds": round(infer_seconds, 2),
                "full_cohort_inference_minutes": round(
                    infer_seconds * FULL_COHORT_HELD_ROWS / len(probe_latent) / 60, 2),
                "metric_seconds": round(metric_seconds, 1),
            },
            "_labels": net_labels,
            "_probe_coords": net_probe[tiers["overlap"]].copy(),
        }
        log(f"      {mode}: train {train_seconds:.0f}s  "
            f"knn15 {distillation['knn_overlap_k15']:.3f}  "
            f"rnx {net_map['rnx_auc']:.4f} (cuml {cuml_map['rnx_auc']:.4f})  "
            f"ARIvs {clustering['vs_cuml']['adjusted_rand']:.3f}  "
            f"{clustering['n_clusters']} clusters")

        del net_probe

    del X, Y
    if edge_tensors is not None:
        del edge_tensors
    if use_gpu:
        torch.cuda.empty_cache()

    return {
        "timing": timing,
        "reference": {
            "map_quality_cuml": cuml_map,
            "n_clusters": int(len(set(cuml_labels.tolist()) - {-1})),
            "noise_fraction": round(float(np.mean(cuml_labels == -1)), 5),
        },
        "modes": mode_results,
        "_cuml_labels": cuml_labels,
    }


# ── aggregation ────────────────────────────────────────────────────────────────

def mean_of(values: list) -> float | None:
    usable = [v for v in values if v is not None and np.isfinite(v)]
    return round(float(np.mean(usable)), 5) if usable else None


def std_of(values: list) -> float | None:
    usable = [v for v in values if v is not None and np.isfinite(v)]
    return round(float(np.std(usable)), 5) if len(usable) > 1 else None


def summarise_mode(replicate_cells: list[dict], use_gpu: bool) -> dict:
    """Average one (size, umap config, mode) across replicates and add family E.

    Family E is the only thing here that cannot be computed from a single replicate: it
    compares independently trained encoders' embeddings of the *same* probe rows, both as
    neighbourhood overlap and as HDBSCAN label agreement.
    """
    ok = [c for c in replicate_cells if not c.get("failed")]
    summary: dict = {
        "replicates": len(replicate_cells),
        "failed_replicates": len(replicate_cells) - len(ok),
    }
    if not ok:
        summary["failure"] = replicate_cells[0].get("failed", "unknown")
        summary["reproducibility"] = None
        return summary

    for section, keys in (
        ("distillation", ("knn_overlap_k15", "knn_overlap_k50", "knn_overlap_k200",
                          "mean_error_in_std", "p95_error_in_std", "procrustes_disparity")),
        ("map_quality_net", ("trustworthiness", "continuity", "rnx_auc", "qnx_at_k",
                             "clamped_fraction")),
        ("generalisation", ("knn_overlap_in_sample", "knn_overlap_out_of_sample",
                            "knn_overlap_gap", "rnx_auc_in_sample", "rnx_auc_out_of_sample",
                            "rnx_auc_gap")),
    ):
        block = {}
        for key in keys:
            values = [c.get(section, {}).get(key) for c in ok]
            block[key] = mean_of(values)
            block[f"{key}_std"] = std_of(values)
        summary[section] = block

    summary["rnx_ratio"] = mean_of([c.get("rnx_ratio") for c in ok])
    summary["rnx_ratio_std"] = std_of([c.get("rnx_ratio") for c in ok])

    clustering = {}
    for key in ("n_clusters", "noise_fraction"):
        values = [c.get("clustering", {}).get(key) for c in ok]
        clustering[key] = mean_of(values)
        clustering[f"{key}_std"] = std_of(values)
    for key in ("adjusted_rand", "normalized_mutual_info", "adjusted_mutual_info", "jaccard"):
        values = [c.get("clustering", {}).get("vs_cuml", {}).get(key) for c in ok]
        clustering[f"vs_cuml_{key}"] = mean_of(values)
    summary["clustering"] = clustering

    timing = {}
    for key in ("encoder_train_seconds", "probe_inference_seconds",
                "full_cohort_inference_minutes", "metric_seconds"):
        timing[key] = mean_of([c.get("timing", {}).get(key) for c in ok])
    summary["timing"] = timing

    # the step curve of the first surviving replicate; they are near-identical by design
    summary["step_curve"] = ok[0].get("step_curve", [])

    # ── family E ──────────────────────────────────────────────────────────────
    usable = [c for c in ok if c.get("_probe_coords") is not None and c.get("_labels") is not None]
    if len(usable) < 2:
        summary["reproducibility"] = None
        summary["reproducibility_note"] = "needs --seeds >= 2 surviving replicates"
    else:
        overlaps, agreements = [], []
        for left, right in combinations(usable, 2):
            overlaps.append(
                overlap_profile(left["_probe_coords"], right["_probe_coords"], (15,), use_gpu)
                ["knn_overlap_k15"]
            )
            agreements.append(metrics.label_agreement(left["_labels"], right["_labels"]))
        summary["reproducibility"] = {
            "knn_overlap_k15": round(float(np.mean(overlaps)), 5),
            "knn_overlap_k15_min": round(float(np.min(overlaps)), 5),
            "adjusted_rand": round(float(np.mean([a["adjusted_rand"] for a in agreements])), 5),
            "adjusted_rand_min": round(float(np.min([a["adjusted_rand"] for a in agreements])), 5),
            "normalized_mutual_info": round(
                float(np.mean([a["normalized_mutual_info"] for a in agreements])), 5),
            "jaccard": round(float(np.mean([a["jaccard"] for a in agreements])), 5),
            "pairs": len(overlaps),
        }
    return summary


# ── planning from a trial run ──────────────────────────────────────────────────

def plan_from_trial(trial_path: Path, args) -> int:
    """Project the full grid's wall time from a completed small run.

    UMAP fit time is the only term that moves with fit size, and it moves smoothly, so two
    or more measured sizes pin a power law ``t = c * n**alpha`` that extrapolates the rest.
    Everything else -- encoder training, inference, metrics -- is fixed-step work whose cost
    does not depend on how many rows were fitted, so those are carried across as measured
    means. Where the trial covered only one size the exponent falls back to 1.0, which is
    conservative for UMAP (its true scaling is sub-linear in the layout phase).
    """
    payload = json.loads(trial_path.read_text())
    observed = payload.get("fit_timing_by_size", {})
    if not observed:
        raise SystemExit(f"{trial_path} has no fit_timing_by_size; was it a completed run?")

    sizes = sorted(int(k) for k in observed)
    times = [float(observed[str(s)]) for s in sizes]
    if len(sizes) >= 2:
        alpha = float(np.log(times[-1] / times[0]) / np.log(sizes[-1] / sizes[0]))
        scale = times[-1] / (sizes[-1] ** alpha)
    else:
        alpha, scale = 1.0, times[0] / sizes[0]
    log(f"UMAP fit scaling from {len(sizes)} measured size(s): t = {scale:.3e} * n^{alpha:.3f}")

    per_mode = payload.get("per_mode_seconds", {})
    reference_seconds = float(payload.get("reference_seconds_mean", 0.0))
    transform_seconds = float(payload.get("transform_seconds_mean", 0.0))

    target_sizes = parse_ints(args.fit_rows)
    n_graph_configs = len(parse_floats(args.min_dist)) * len(parse_ints(args.n_neighbors))
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    mode_cost = sum(float(per_mode.get(mode, 0.0)) for mode in modes)

    total = 0.0
    breakdown = []
    for size in target_sizes:
        fit_seconds = scale * (size ** alpha)
        per_cell = fit_seconds + transform_seconds + reference_seconds + mode_cost
        cells = n_graph_configs * args.seeds
        subtotal = per_cell * cells
        total += subtotal
        breakdown.append((size, fit_seconds, per_cell, cells, subtotal))

    hours = total / 3600.0
    print("\n" + "=" * 78)
    print("PROJECTED FULL SWEEP")
    print("=" * 78)
    print(f"  grid: {len(target_sizes)} sizes x {n_graph_configs} graph configs "
          f"x {args.seeds} replicates = {len(target_sizes) * n_graph_configs * args.seeds} "
          f"UMAP fits, each feeding {len(modes)} encoder(s)")
    print(f"  measured per-cell costs: transform {transform_seconds:.0f}s, "
          f"reference metrics {reference_seconds:.0f}s, "
          + ", ".join(f"{m} {float(per_mode.get(m, 0.0)):.0f}s" for m in modes))
    print()
    print(f"  {'size':>10}{'umap fit':>12}{'per cell':>12}{'cells':>8}{'hours':>10}")
    for size, fit_seconds, per_cell, cells, subtotal in breakdown:
        print(f"  {size / 1e6:>9.0f}M{fit_seconds:>11.0f}s{per_cell:>11.0f}s"
              f"{cells:>8}{subtotal / 3600:>10.1f}")
    print(f"  {'TOTAL':>10}{'':>12}{'':>12}{'':>8}{hours:>10.1f}")
    print()
    budget = args.max_hours
    if hours <= budget:
        print(f"  VERDICT: {hours:.1f} h is within the {budget:.0f} h budget -- run it.")
        print("=" * 78 + "\n")
        return 0
    print(f"  VERDICT: {hours:.1f} h EXCEEDS the {budget:.0f} h budget.")
    print(f"  Cheapest cuts, in order of what they buy:")
    largest = max(breakdown, key=lambda item: item[4])
    print(f"    - drop the {largest[0] / 1e6:.0f}M size          -> saves "
          f"{largest[4] / 3600:.1f} h")
    print(f"    - --seeds {args.seeds} -> 2                 -> saves "
          f"{hours * (1 - 2 / args.seeds):.1f} h (weakens family E to one pair)")
    if "regress" in modes and float(per_mode.get("regress", 0)) > 0:
        saved = float(per_mode["regress"]) * sum(b[3] for b in breakdown) / 3600
        print(f"    - drop the regress control       -> saves {saved:.1f} h")
    print("=" * 78 + "\n")
    return 1


# ── entry point ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep parametric UMAP over fit size x graph parameters x objective")
    parser.add_argument("--embed-dir", help="stage 1 output holding latent.npy")
    parser.add_argument("--output-dir")
    parser.add_argument("--plan-from", type=Path,
                        help="read a completed run's JSON and print the projected cost of "
                             "the grid described by the other flags, then exit")
    parser.add_argument("--max-hours", type=float, default=48.0,
                        help="budget --plan-from checks the projection against")

    # the grid
    parser.add_argument("--fit-rows", default="2000000,5000000,10000000,25000000",
                        help="nested within a replicate: the 5M set contains the 2M set")
    parser.add_argument("--min-dist", default="0.0,0.1,0.25")
    parser.add_argument("--n-neighbors", default="15,30,50",
                        help="size-coupled, which is why it is swept here against fit size "
                             "rather than pinned as in sweep 1")
    parser.add_argument("--modes", default="regress,umap,hybrid",
                        help="training objectives; all share one UMAP fit per cell")
    parser.add_argument("--seeds", type=int, default=3,
                        help="replicates; each redraws the fit rows and every seed. 2 is the "
                             "minimum for a reproducibility number, 3 shows outliers")
    parser.add_argument("--n-epochs", type=int, default=200)

    # encoder / training
    parser.add_argument("--hidden", default="256,256,128")
    parser.add_argument("--regress-steps", type=int, default=15_000)
    parser.add_argument("--umap-steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--umap-learning-rate", type=float, default=2e-4)
    parser.add_argument("--negative-sample-rate", type=int, default=None,
                        help="default: read from the fitted cuML model")
    parser.add_argument("--max-edges", type=int, default=150_000_000,
                        help="uniform subsample above this; keeps a 25M/nn50 graph resident "
                             "alongside torch")
    parser.add_argument("--checkpoint-every", type=int, default=5_000,
                        help="trace kNN-overlap during training; 0 disables")

    # evaluation tiers, all nested and drawn once
    parser.add_argument("--probe-rows", type=int, default=500_000)
    parser.add_argument("--cluster-rows", type=int, default=500_000)
    parser.add_argument("--overlap-rows", type=int, default=200_000,
                        help="matches gate 1's probe so kNN-overlap is directly comparable "
                             "to the 0.4141 determinism floor")
    parser.add_argument("--knn-rows", type=int, default=30_000)
    parser.add_argument("--in-sample-rows", type=int, default=30_000,
                        help="training rows re-scored for the generalisation gap")
    parser.add_argument("--curve-rows", type=int, default=20_000,
                        help="tiny slice used for the mid-training curve only")
    parser.add_argument("--metric-k", type=int, default=15)
    parser.add_argument("--metric-k-max", type=int, default=3_000)

    parser.add_argument("--min-cluster-size", type=int, default=250,
                        help="ratio-matched to sweep 1's 500 at 1M cluster rows")
    parser.add_argument("--min-samples", type=int, default=25)
    parser.add_argument("--transform-batch-size", type=int, default=5_000_000)
    parser.add_argument("--infer-batch-size", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-budget-gb", type=float, default=None)
    parser.add_argument("--encoder-dir", default=None,
                        help="if given, write each trained encoder here as "
                             "<rows>__<config>__<mode>__seed<n>.pt. Lets apply_parametric_full.py "
                             "reuse the exact network instead of refitting UMAP and retraining.")
    parser.add_argument("--rmm-share", type=float, default=None,
                        help="fraction of the GPU budget given to the RMM pool (cuML); the "
                             "rest goes to torch. Default 0.5 (stage 'apply'). BOTH SIDES "
                             "CAN STARVE, and which one binds flips with fit size -- see the "
                             "note above main(). Measured on a 47 GB card at --gpu-budget-gb "
                             "45: 0.7 for fits up to 10M, 0.85 at 25M. 0.9 OOMs torch.")
    return parser.parse_args()


def main() -> int:
    started_all = perf_counter()
    args = parse_args()

    if args.plan_from is not None:
        return plan_from_trial(args.plan_from, args)
    if not args.embed_dir or not args.output_dir:
        raise SystemExit("--embed-dir and --output-dir are required unless --plan-from is given")

    args.modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = set(args.modes) - {"regress", "umap", "hybrid"}
    if unknown:
        raise SystemExit(f"unknown mode(s): {', '.join(sorted(unknown))}")
    args.hidden = tuple(int(p) for p in args.hidden.split(",") if p.strip())

    embed_dir = Path(args.embed_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    latent_path = embed_dir / "latent.npy"
    if not latent_path.exists():
        raise SystemExit(f"latent.npy not found in {embed_dir}")

    use_gpu = core.gpu_available()
    if use_gpu and args.gpu_budget_gb is not None:
        # One budget, two allocators in this one process, and BOTH can starve. Which one
        # binds flips with fit size, so there is no single correct share:
        #
        #   cuML (RMM pool) holds the kNN graph and the fuzzy simplicial set during the
        #     fit; its peak scales with n_rows x n_neighbors. A 10M x nn30 fit needed
        #     ~25 GB and died against a 20 GB pool.
        #   torch holds the edge tensors for the encoder's UMAP objective -- head, tail,
        #     and the float64 weight cumsum -- plus X and Y. That is ~5.6 GB at the 150M
        #     --max-edges cap and is roughly FLAT in fit size, because the cap binds.
        #
        # So at 10M the squeeze is on torch and at 25M it is on cuML. Setting the share to
        # 0.9 to fix the first OOM left torch 4.5 GB and killed two runs at the cumsum,
        # with 16 GB still free on the card. Match the share to the largest fit in the run.
        core.apply_gpu_budget("apply", budget_gb=args.gpu_budget_gb,
                              rmm_share=args.rmm_share)

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"torch device: {device}, cuml {'available' if use_gpu else 'unavailable'}")

    latent = np.load(latent_path, mmap_mode="r")
    total_rows, latent_dim = latent.shape
    log(f"{latent_path.name}: {total_rows:,} rows x {latent_dim} dims")

    sizes = parse_ints(args.fit_rows)
    graph_configs = [
        core.UmapConfig(n_neighbors=nn, min_dist=md, n_components=2, n_epochs=args.n_epochs)
        for nn in parse_ints(args.n_neighbors)
        for md in parse_floats(args.min_dist)
    ]
    total_cells = len(sizes) * len(graph_configs) * args.seeds
    log(f"{len(sizes)} sizes x {len(graph_configs)} graph configs x {args.seeds} replicates "
        f"= {total_cells} UMAP fits, each feeding {len(args.modes)} encoder(s)")

    probe_index, pool = draw_probe(total_rows, args.probe_rows, args.seed)
    if max(sizes) > pool.size:
        raise SystemExit(
            f"largest --fit-rows {max(sizes):,} exceeds the {pool.size:,} rows left after "
            f"the probe")

    log("materialising the probe latents ...")
    probe_latent = np.ascontiguousarray(latent[probe_index])

    # Tiers are drawn once and reused everywhere, so every cell is scored on identical rows.
    subset_rng = np.random.default_rng(args.seed + 1)
    cluster_positions = np.sort(subset_rng.choice(
        len(probe_index), size=min(args.cluster_rows, len(probe_index)), replace=False))
    overlap_positions = np.sort(subset_rng.choice(
        len(probe_index), size=min(args.overlap_rows, len(probe_index)), replace=False))
    knn_positions = np.sort(subset_rng.choice(
        overlap_positions, size=min(args.knn_rows, len(overlap_positions)), replace=False))
    curve_positions = np.sort(subset_rng.choice(
        knn_positions, size=min(args.curve_rows, len(knn_positions)), replace=False))
    # in_sample is deliberately absent here: it indexes the *fit* rows, not the probe, so it
    # is drawn per replicate once the fit sets exist. See in_sample_ids_by_replicate below.
    tiers = {
        "cluster": cluster_positions,
        "overlap": overlap_positions,
        "knn": knn_positions,
        "curve": curve_positions,
    }
    in_sample_rows = min(args.in_sample_rows, min(sizes))
    log(f"tiers: probe {len(probe_index):,} / cluster {len(cluster_positions):,} / "
        f"overlap {len(overlap_positions):,} / knn {len(knn_positions):,} / "
        f"curve {len(curve_positions):,} / in-sample {in_sample_rows:,}")

    cells: dict[str, list[dict]] = {}
    reference_by_cell: dict[str, list[dict]] = {}
    fit_seconds_by_size: dict[int, list[float]] = {size: [] for size in sizes}
    transform_seconds_all: list[float] = []
    reference_seconds_all: list[float] = []
    mode_seconds: dict[str, list[float]] = {mode: [] for mode in args.modes}

    def aggregate() -> dict:
        summaries = {}
        for key, replicate_cells in cells.items():
            summary = summarise_mode(replicate_cells, use_gpu)
            references = [r for r in reference_by_cell[key] if r]
            if references:
                summary["reference_cuml"] = {
                    "rnx_auc": mean_of([r["map_quality_cuml"]["rnx_auc"] for r in references]),
                    "trustworthiness": mean_of(
                        [r["map_quality_cuml"]["trustworthiness"] for r in references]),
                    "clamped_fraction": mean_of(
                        [r["map_quality_cuml"]["clamped_fraction"] for r in references]),
                    "n_clusters": mean_of([r["n_clusters"] for r in references]),
                    "noise_fraction": mean_of([r["noise_fraction"] for r in references]),
                }
            size_part, umap_part, mode_part = key.split("|")
            summary["fit_rows"] = int(size_part)
            summary["umap_config"] = umap_part
            summary["mode"] = mode_part
            summaries[key] = summary
        return summaries

    def build_payload(summaries: dict, complete: bool) -> dict:
        return {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "sweep": "parametric_umap_size_x_parameters",
            "complete": complete,
            "embed_dir": str(embed_dir),
            "total_rows": int(total_rows),
            "latent_dim": int(latent_dim),
            "sizes": sizes,
            "graph_configs": {c.label(): c.as_dict() for c in graph_configs},
            "modes": args.modes,
            "replicates": args.seeds,
            "encoder": {
                "hidden": list(args.hidden),
                "regress_steps": args.regress_steps,
                "umap_steps": args.umap_steps,
                "batch_size": args.batch_size,
                "max_edges": args.max_edges,
            },
            "tiers": {**{name: int(len(value)) for name, value in tiers.items()},
                      "in_sample": int(in_sample_rows)},
            # in_sample rows are resolved per fit set by row id, so the same physical rows
            # are scored at every size. Runs before this flag was added used fixed positions
            # into the sorted fit set, which narrowed the slice as the fit grew.
            "in_sample_selection": "row_ids",
            "probe_rows": int(len(probe_index)),
            "metric_k": args.metric_k,
            "metric_k_max": args.metric_k_max,
            "backend": "cuml" if use_gpu else "cpu",
            "reference_lines": {
                "determinism_floor_knn_overlap": DETERMINISM_FLOOR,
                "latent_neighbour_preservation": LATENT_PRESERVATION,
                "aumap_knn_overlap": AUMAP_BASELINE,
                "regress_md0_failure_knn_overlap": REGRESS_MD0_FAILURE,
                "nonparametric_seed_ari": NONPARAMETRIC_SEED_ARI,
            },
            # consumed by --plan-from
            "fit_timing_by_size": {
                str(size): round(float(np.mean(values)), 1)
                for size, values in fit_seconds_by_size.items() if values
            },
            "transform_seconds_mean": round(float(np.mean(transform_seconds_all)), 1)
            if transform_seconds_all else 0.0,
            "reference_seconds_mean": round(float(np.mean(reference_seconds_all)), 1)
            if reference_seconds_all else 0.0,
            "per_mode_seconds": {
                mode: round(float(np.mean(values)), 1) if values else 0.0
                for mode, values in mode_seconds.items()
            },
            "summaries": summaries,
            "total_seconds": round(perf_counter() - started_all, 1),
        }

    partial_path = output_dir / "parametric_sweep_partial.json"

    # Every replicate's fit sets, drawn once. Each draw permutes the whole ~157 M-row pool,
    # so doing it inside the loop would repeat 1.3 GB of shuffling for every cell.
    log(f"drawing fit sets for {args.seeds} replicate(s) ...")
    fit_sets_by_replicate = [
        draw_nested_fit_sets(pool, sizes, args.seed, replicate)
        for replicate in range(args.seeds)
    ]

    # Family D's in-sample rows, as row ids drawn from each replicate's *smallest* fit set.
    # Sizes are nested within a replicate, so those ids are present at every size and
    # family D therefore scores the same physical rows as the fit grows -- which is what
    # makes its size axis a measurement of generalisation rather than of which slice of the
    # cohort happened to be sampled. Per replicate because the fit sets are redrawn.
    in_sample_rng = np.random.default_rng(args.seed + 2)
    in_sample_ids_by_replicate = [
        np.sort(in_sample_rng.choice(
            fit_sets_by_replicate[replicate][min(sizes)], size=in_sample_rows, replace=False))
        for replicate in range(args.seeds)
    ]

    # Replicate is the INNERMOST loop, deliberately. Family E -- agreement between
    # independently trained encoders -- is the only thing here that cannot be computed from
    # a single replicate, and it is the number the sweep exists to produce. With replicate
    # outermost, a crash anywhere in the first pass leaves every cell at one replicate and
    # family E empty no matter how much finished; that happened three runs in a row. This
    # order finishes all replicates of one cell before starting the next, so an interrupted
    # run yields fewer cells but every completed cell carries its reproducibility number.
    done = 0
    for size in sizes:
        for config in graph_configs:
            log(f"=== {format_rows(size)} rows, {config.label()} "
                f"({args.seeds} replicate(s)) ===")
            for replicate in range(args.seeds):
                fit_index = fit_sets_by_replicate[replicate][size]
                seeded = core.UmapConfig(
                    **{**config.as_dict(), "seed": args.seed + replicate})
                log(f"  replicate {replicate + 1}/{args.seeds} ...")
                cell = run_cell(
                    size, seeded, latent, fit_index, probe_latent,
                    {**tiers, "in_sample": in_sample_ids_by_replicate[replicate]},
                    args, use_gpu, device)

                fit_seconds_by_size[size].append(cell["timing"].get("umap_fit_seconds", 0.0))
                transform_seconds_all.append(cell["timing"].get("probe_transform_seconds", 0.0))
                reference_seconds_all.append(cell["timing"].get("reference_metric_seconds", 0.0))

                for mode in args.modes:
                    key = cell_key(size, config.label(), mode)
                    result = cell["modes"].get(mode)
                    if result is None:
                        result = {"failed": cell.get("failed", "cell failed before training")}
                    else:
                        spent = result.get("timing", {})
                        mode_seconds[mode].append(
                            spent.get("encoder_train_seconds", 0.0)
                            + spent.get("probe_inference_seconds", 0.0)
                            + spent.get("metric_seconds", 0.0))
                    cells.setdefault(key, []).append(result)
                    reference_by_cell.setdefault(key, []).append(cell.get("reference", {}))

                done += 1
                elapsed = perf_counter() - started_all
                log(f"    cell {done}/{total_cells}, elapsed {elapsed / 60:.0f} min, "
                    f"eta {(elapsed / done) * (total_cells - done) / 60:.0f} min")

                # Checkpoint after every cell so a crash at hour 18 loses at most one cell.
                try:
                    partial_path.write_text(
                        json.dumps(build_payload(aggregate(), complete=False), indent=2))
                except Exception as error:
                    log(f"  WARNING: could not write checkpoint: {error}")

    log("=== aggregating across replicates ===")
    payload = build_payload(aggregate(), complete=True)
    destination = output_dir / "parametric_sweep.json"
    destination.write_text(json.dumps(payload, indent=2))
    print_report(payload)
    log(f"wrote {destination}")
    if partial_path.exists():
        partial_path.unlink()
    return 0


def print_report(payload: dict) -> None:
    summaries = payload["summaries"]
    keys = list(summaries)

    def cell(value, width=10, decimals=4):
        if value is None:
            return f"{'--':>{width}}"
        if isinstance(value, float):
            text = f"{value:.{decimals}f}" if abs(value) < 1000 else f"{value:.4g}"
        else:
            text = str(value)
        return f"{text:>{width}}"

    header = (f"  {'size|config|mode':<34}{'knn15':>9}{'knn200':>9}{'rnx_net':>9}"
              f"{'rnx_cuml':>10}{'ratio':>8}{'ARIvs':>8}{'ARIrep':>8}{'gap':>8}{'train_s':>9}")
    rule = "=" * len(header)
    print("\n" + rule)
    print(f"Parametric UMAP sweep -- {len(payload['sizes'])} sizes x "
          f"{len(payload['graph_configs'])} graph configs x {len(payload['modes'])} objectives, "
          f"{payload['replicates']} replicates")
    print(rule)
    print(f"  reference lines: determinism floor knn15 = {DETERMINISM_FLOOR:.4f}, "
          f"aUMAP = {AUMAP_BASELINE:.3f}, known regress failure = {REGRESS_MD0_FAILURE:.3f}")
    print(f"  non-parametric seed ARI for comparison: "
          f"{NONPARAMETRIC_SEED_ARI['min']:.3f} to {NONPARAMETRIC_SEED_ARI['max']:.3f}")
    print()
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key in keys:
        summary = summaries[key]
        size_part, umap_part, mode_part = key.split("|")
        shown = f"{format_rows(int(size_part))}|{umap_part}|{mode_part}"
        if summary.get("failed_replicates") and not summary.get("distillation"):
            print(f"  {shown:<34}FAILED -- {summary.get('failure', '')}")
            continue
        distillation = summary.get("distillation", {})
        reproducibility = summary.get("reproducibility") or {}
        reference = summary.get("reference_cuml", {})
        print(f"  {shown:<34}"
              + cell(distillation.get("knn_overlap_k15"), 9)
              + cell(distillation.get("knn_overlap_k200"), 9)
              + cell(summary.get("map_quality_net", {}).get("rnx_auc"), 9)
              + cell(reference.get("rnx_auc"), 10)
              + cell(summary.get("rnx_ratio"), 8, 3)
              + cell(summary.get("clustering", {}).get("vs_cuml_adjusted_rand"), 8, 3)
              + cell(reproducibility.get("adjusted_rand"), 8, 3)
              + cell(summary.get("generalisation", {}).get("knn_overlap_gap"), 8, 3)
              + cell(summary.get("timing", {}).get("encoder_train_seconds"), 9, 0))

    # ── verdict ────────────────────────────────────────────────────────────────
    print("\n  -- verdict --")
    usable = [(k, s) for k, s in summaries.items() if s.get("distillation", {}).get("knn_overlap_k15")]
    if not usable:
        print("  no cell produced a usable embedding")
        print(rule + "\n")
        return

    best_distil = max(usable, key=lambda kv: kv[1]["distillation"]["knn_overlap_k15"])
    print(f"  best distillation (knn15 vs cuML transform): {best_distil[0]}  "
          f"({best_distil[1]['distillation']['knn_overlap_k15']:.4f})")
    if best_distil[1]["distillation"]["knn_overlap_k15"] < DETERMINISM_FLOOR:
        print(f"    ... which is BELOW the {DETERMINISM_FLOOR:.4f} determinism floor. No "
              f"encoder here imitates a given fit as closely as a re-run of UMAP does. "
              f"Judge on rnx_ratio and ARIrep instead -- see below.")

    ratios = [(k, s["rnx_ratio"]) for k, s in summaries.items() if s.get("rnx_ratio")]
    if ratios:
        best_ratio = max(ratios, key=lambda kv: kv[1])
        print(f"  best map quality (rnx_net / rnx_cuml):      {best_ratio[0]}  "
              f"({best_ratio[1]:.3f})")
        if best_ratio[1] >= 0.95:
            print(f"    at {best_ratio[1]:.3f} the encoder's own map is as faithful to the "
                  f"16-D latent as cuML's, so disagreement with transform() is a different "
                  f"map rather than a worse one.")

    repro = [(k, (s.get("reproducibility") or {}).get("adjusted_rand"))
             for k, s in summaries.items()]
    repro = [(k, v) for k, v in repro if v is not None]
    if repro:
        best_repro = max(repro, key=lambda kv: kv[1])
        print(f"  best reproducibility (replicate ARI):       {best_repro[0]}  "
              f"({best_repro[1]:.4f})")
        reference = NONPARAMETRIC_SEED_ARI["max"]
        if best_repro[1] > reference:
            print(f"    beats the best non-parametric config ({reference:.3f}) -- the encoder "
                  f"is genuinely more reproducible across row draws, which is the whole "
                  f"reason for building it.")
        else:
            print(f"    does NOT beat the best non-parametric config ({reference:.3f}). The "
                  f"encoder buys inference speed but not reproducibility; say so plainly "
                  f"rather than adopting it for the wrong reason.")
    else:
        print("  no reproducibility numbers -- rerun with --seeds >= 2")

    gaps = [(k, s.get("generalisation", {}).get("knn_overlap_gap"))
            for k, s in summaries.items()]
    gaps = [(k, v) for k, v in gaps if v is not None]
    if gaps:
        worst_gap = max(gaps, key=lambda kv: kv[1])
        print(f"  largest generalisation gap (in - out):      {worst_gap[0]}  "
              f"({worst_gap[1]:+.4f})")
        if worst_gap[1] > 0.05:
            print(f"    a gap that size means the encoder fits its training rows better than "
                  f"new ones; check whether it shrinks with fit size, which is what the size "
                  f"axis is for.")

    print("\n  Ranked on knn15 (against the measured determinism floor), rnx_ratio and "
          "replicate ARI.")
    print("  Coordinate error is printed in the JSON and deliberately not ranked on: the "
          "known\n  regress failure scored 0.055 embedding std (excellent) at 0.055 "
          "kNN-overlap (useless).")
    print(rule + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
