"""Parametric UMAP in PyTorch -- learn the 16-D -> 2-D map, then infer with a forward pass.

Motivation, measured rather than assumed:

    phase A      fit 5 M rows = 115 s, transform 152.5 M rows = 3291 s (98% of the stage)
    aumap.py     kNN interpolation: 3x faster, but kNN-overlap 0.44 -- rejected

aUMAP failed for a specific reason worth keeping in mind here. At ``min_dist=0.0`` UMAP
packs points inside an island so tightly that the true 15th neighbour sits a hair away;
inverse-distance blending of reference coordinates lands close in absolute terms (median
error 0.003 embedding std) yet reorders neighbours completely. Coordinate accuracy and
neighbour accuracy came apart. **kNN-overlap remains the acceptance criterion here, not
coordinate error**, and this module reports both so the same trap is visible if it recurs.

A network can succeed where interpolation failed because it fits a smooth function over
the whole 16-D space instead of averaging whatever reference points happen to be nearby --
it can represent a sharp boundary, which a distance-weighted mean structurally cannot.

Two training objectives, because they fail differently and the comparison is cheap:

``regress``
    Supervised MSE onto cuML's own embedding, using the 5 M (latent, coordinate) pairs the
    fit already produced. Distillation of ``transform()``. Output is *aligned* with the
    cuML embedding, so raw coordinate error stays meaningful and the result drops into the
    existing pipeline unchanged. Risk: MSE averages, so points near a boundary may be
    pulled into the gap between islands -- aUMAP's failure in a different guise.

The network is a plain MLP trained on raw values -- no input standardisation, no output
rescaling. Both reference implementations leave scaling to the caller, and the UMAP loss
needs cuML's own coordinate units anyway, since ``a`` and ``b`` encode an absolute
distance scale.

``umap``
    The real UMAP cross-entropy on the fitted graph: attraction along ``graph_`` edges,
    repulsion against uniformly sampled non-edges. Optimises the objective UMAP itself
    optimises, so boundaries stay sharp. Output lives in its own frame (rotation /
    reflection / scale are unconstrained), so it is compared after Procrustes alignment
    and judged on kNN-overlap.

``hybrid`` (default)
    Regress first for a warm start in the right frame, then fine-tune under UMAP loss.
    Keeps the alignment of the first and the boundary behaviour of the second.

Nothing here calls TensorFlow. The env's TF install cannot see the Blackwell GPU
(``Cannot dlopen some GPU libraries``), which is what ruled out umap-learn's own
parametric implementation; torch already runs on this card for VAE training.

Everything needed comes off the fitted cuML model, verified present on this install:
``graph_`` (scipy COO, numpy data), ``_knn_indices``, ``negative_sample_rate``. The one
gap is ``a``/``b``, which cuML leaves as ``None`` after ``fit`` -- they are recomputed
from ``min_dist`` with umap-learn's own ``find_ab_params``, the same routine cuML uses
internally.

    python umap_hdbscan_sweep/parametric_umap.py \\
        --embed-dir <stage1-output> \\
        --output-dir umap_hdbscan_sweep/umap_tests
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
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
import torch
import torch.nn as nn

import sweep_core as core
from aumap import compare_embeddings

FULL_COHORT_HELD_ROWS = 152_501_580

# aumap.py's best result, printed alongside so the comparison needs no cross-referencing.
AUMAP_BASELINE = {"knn_overlap": 0.440, "full_cohort_minutes": 20.3, "interp_k": 5}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# ── the network ────────────────────────────────────────────────────────────────

class ParametricEncoder(nn.Module):
    """Plain MLP, 16 -> 2. Deliberately small.

    Inference is memory-bandwidth bound, not compute bound: 152.5 M x 16 float32 is 10 GB
    to read whatever the layer widths are, and the arithmetic here is a rounding error
    next to that. Width buys capacity essentially for free, so the default is generous;
    depth past ~4 layers has not helped in practice on a map this low-dimensional.

    No dropout or batch norm. Both objectives below are already heavily stochastic
    (edge sampling, negative sampling), and batch norm in particular interacts badly with
    the UMAP loss, whose scale is not fixed a priori.
    """

    def __init__(self, input_dim: int, output_dim: int = 2,
                 hidden: tuple[int, ...] = (256, 256, 128)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden:
            layers += [nn.Linear(previous, width), nn.ReLU(inplace=True)]
            previous = width
        layers.append(nn.Linear(previous, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── graph extraction ───────────────────────────────────────────────────────────

def extract_edges(graph) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(head, tail, weight) for each undirected edge of the fuzzy simplicial set.

    ``graph_`` is the symmetric fuzzy union, so ``(i, j)`` and ``(j, i)`` both carry the
    same weight; keeping ``row < col`` takes each undirected edge exactly once. The weight
    is p_ij -- UMAP's membership strength, already in [0, 1].
    """
    coo = graph.tocoo()
    # This install returns scipy COO with numpy arrays, but cuML has shipped cupy-backed
    # sparse in other versions -- .get() covers both without importing cupy.
    row, col, data = (
        array.get() if hasattr(array, "get") else array
        for array in (coo.row, coo.col, coo.data)
    )

    upper = row < col
    return (
        row[upper].astype(np.int64, copy=False),
        col[upper].astype(np.int64, copy=False),
        data[upper].astype(np.float32, copy=False),
    )


def convert_distance_to_log_probability(d2: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """``log(q)`` for ``q = 1 / (1 + a * d^2b)`` -- umap-learn's function of the same name.

    Theirs takes Euclidean ``d`` and computes ``d ** (2 * b)``; this takes the squared
    distance and raises it to ``b``, which is the same quantity without a sqrt that would
    only be squared again.
    """
    return -torch.log1p(a * d2.pow(b))


def compute_cross_entropy(
    probabilities_graph: torch.Tensor,
    log_probabilities_distance: torch.Tensor,
    repulsion_strength: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Port of ``lmcinnes/umap``'s ``compute_cross_entropy`` (``parametric_umap.py``).

    ``log_probabilities_distance`` is treated as a **logit**, not as a log-probability:
    the effective low-dimensional membership is ``sigmoid(log q) = q / (1 + q)``. That is
    a different target than the ``q`` non-parametric UMAP optimises, and it is deliberate
    upstream -- it is what makes the repulsive term bounded (see below), and it is the
    objective the released Parametric UMAP was validated against.

    The repellent term uses the identity cited in their source, Shi et al. 2022
    (arXiv:2111.08851)::

        log(1 - sigmoid(x)) = log_sigmoid(x) - x

    which is what makes the distance floor umap-learn's *non-parametric* kernel needs
    (``0.001 + dist_squared``) unnecessary here. Writing ``w = a * d^2b``, the repellent
    term reduces to ``log((2 + w) / (1 + w))``: it is **0.693 at coincident points**, not
    infinite, and its derivative ``-1 / ((2 + w)(1 + w))`` is likewise bounded. The
    ``-log(1 - q)`` form used by non-parametric UMAP diverges at exactly that point.
    """
    log_sigmoid = torch.nn.functional.logsigmoid(log_probabilities_distance)
    attraction_term = -probabilities_graph * log_sigmoid
    repellant_term = (
        -(1.0 - probabilities_graph)
        * (log_sigmoid - log_probabilities_distance)
        * repulsion_strength
    )
    return attraction_term, repellant_term, attraction_term + repellant_term


def ab_params(min_dist: float, spread: float = 1.0) -> tuple[float, float]:
    """UMAP's ``a``/``b`` for a given ``min_dist``.

    cuML computes these C++-side and leaves the Python attributes as ``None`` after
    ``fit``, so they are recomputed here with the identical curve fit umap-learn uses.
    """
    from umap.umap_ import find_ab_params

    a, b = find_ab_params(spread=spread, min_dist=min_dist)
    return float(a), float(b)


# ── the estimator ──────────────────────────────────────────────────────────────

class ParametricUmap:
    """Trained 16-D -> 2-D encoder with a ``transform`` that is a forward pass.

    No input or output normalisation: the encoder is trained directly on raw latent
    values and, in ``regress``/``hybrid``, raw cuML embedding coordinates. ``transform``
    is nothing but ``encoder(x)``.
    """

    def __init__(self, encoder: ParametricEncoder, device: torch.device,
                 mode: str = "hybrid") -> None:
        self.encoder = encoder
        self.device = device
        self.mode = mode

    def transform(self, X: np.ndarray, batch_size: int = 2_000_000) -> np.ndarray:
        """Embed rows the fit never saw.

        Batch-invariant by construction: each row's output depends only on that row, so
        unlike ``FittedUmap.transform`` there is no batch-size-dependent drift to pin down.
        """
        self.encoder.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, X.shape[0], batch_size):
                chunk = np.ascontiguousarray(X[start:start + batch_size], dtype=np.float32)
                tensor = torch.from_numpy(chunk).to(self.device, non_blocking=True)
                out = self.encoder(tensor)
                outputs.append(out.cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(outputs, axis=0) if len(outputs) > 1 else outputs[0]

    def save(self, path: Path) -> None:
        torch.save({"state_dict": self.encoder.state_dict(), "mode": self.mode}, path)


# ── training ───────────────────────────────────────────────────────────────────

def train_regression(
    encoder: ParametricEncoder,
    X: torch.Tensor,
    Y: torch.Tensor,
    steps: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    checkpoint_every: int | None = None,
    on_checkpoint=None,
) -> list[float]:
    """MSE onto cuML's embedding, in cuML's own coordinate units. Both tensors on device.

    ``on_checkpoint(step, encoder)`` fires every ``checkpoint_every`` steps when both are
    given, and is otherwise never touched -- with the defaults this function behaves exactly
    as it did before the hook existed. It lets a caller trace quality *during* training
    rather than re-running the whole thing at several step budgets, which is what turns
    "how many steps are enough" from a grid axis into a free by-product. The callback is
    expected to flip the encoder to eval mode, so training mode is restored after it returns.
    """
    optimiser = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=steps)
    n_rows = X.shape[0]
    history = []
    encoder.train()
    for step in range(steps):
        idx = torch.randint(0, n_rows, (batch_size,), device=device)
        loss = nn.functional.mse_loss(encoder(X[idx]), Y[idx])
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        schedule.step()
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            history.append(round(float(loss.item()), 6))
            log(f"    regress step {step + 1}/{steps}  mse {loss.item():.6f}")
        if on_checkpoint is not None and checkpoint_every and (step + 1) % checkpoint_every == 0:
            on_checkpoint(step + 1, encoder)
            encoder.train()
    return history


def train_umap_loss(
    encoder: ParametricEncoder,
    X: torch.Tensor,
    edge_head: torch.Tensor,
    edge_tail: torch.Tensor,
    edge_cumsum: torch.Tensor,
    a: float,
    b: float,
    steps: int,
    batch_size: int,
    learning_rate: float,
    negative_sample_rate: int,
    repulsion_strength: float,
    device: torch.device,
    eps: float = 1e-4,
    checkpoint_every: int | None = None,
    on_checkpoint=None,
) -> list[float]:
    """UMAP cross-entropy, ported from ``lmcinnes/umap``'s ``UMAPModel._umap_loss``.

    The body follows that method step for step: repeat-and-shuffle negative sampling,
    one concatenated distance vector (positives then negatives), a constant
    ``probabilities_graph`` of ones-then-zeros, ``convert_distance_to_log_probability``,
    ``compute_cross_entropy``, and a single ``mean`` over the pooled result. The
    membership target is therefore ``sigmoid(log q) = q / (1 + q)``, not ``q`` -- see
    ``compute_cross_entropy`` for why that is upstream's deliberate choice and what it
    buys numerically.

    Two things differ from upstream, both forced by scale rather than preference:

    **Edge sampling.** Upstream expands the edge list by ``epochs_per_sample`` (``np.repeat``
    of head/tail), permutes it once, and streams it through ``tf.data`` with a second
    10k-window shuffle. At ``n_epochs=200`` over tens of millions of edges that array runs
    to ~2e9 entries, so it is not built here. Instead each step draws edges inverse-CDF
    from a cumulative weight sum (``searchsorted``, on device, no ``torch.multinomial``
    category limit). Same per-edge frequency in expectation, i.i.d. with replacement
    rather than a shuffled fixed-count pass -- correct expectation, some variance in the
    realised per-edge visit count.

    **Where the input rows come from.** Upstream's ``tf.data`` pipeline gathers ``X`` rows
    for each edge; here ``X`` is already resident on the GPU, so the gather is a direct
    index. No behavioural difference.

    ``checkpoint_every`` / ``on_checkpoint`` behave exactly as in ``train_regression``: off
    by default, and when supplied they only observe the encoder mid-training.

    ``edge_cumsum`` should be **float64**. Weights are O(0.5) each, so a float32 running sum
    stops advancing past ~1.7e7 (the 24-bit mantissa) and every draw beyond that point lands
    on the same edge. That threshold is already crossed by a 5 M-row graph at
    ``n_neighbors=15`` (~3.7e7 edges), so the caller building this tensor is responsible for
    the dtype -- ``torch.rand(...) * total_weight`` then promotes the draw to match.

    Negative sampling itself now matches upstream exactly: ``embedding_from`` repeated
    ``negative_sample_rate`` times and shuffled, so negative partners come from the current
    batch's vertices rather than uniformly from all N. A negative pair that happens to be a
    real edge is left alone, as upstream also does.
    """
    optimiser = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=steps)
    total_weight = edge_cumsum[-1]
    history = []
    encoder.train()

    # Constant across steps: 1 for the positive block, 0 for the negative block. This is
    # umap-learn's `probabilities_graph` -- p_ij never appears as a value here because the
    # weighting lives in how often an edge is sampled, not in the loss.
    probabilities_graph = torch.cat([
        torch.ones(batch_size, device=device),
        torch.zeros(batch_size * negative_sample_rate, device=device),
    ])

    for step in range(steps):
        draw = torch.rand(batch_size, device=device) * total_weight
        edge_idx = torch.searchsorted(edge_cumsum, draw).clamp_max_(edge_head.shape[0] - 1)
        head, tail = edge_head[edge_idx], edge_tail[edge_idx]

        embedding_to = encoder(X[head])
        embedding_from = encoder(X[tail])

        # Negative sampling exactly as `_umap_loss` does it: repeat both sides
        # `negative_sample_rate` times, then shuffle only the `from` side, so each positive
        # head is paired against `rate` other vertices drawn from this batch.
        embedding_neg_to = embedding_to.repeat_interleave(negative_sample_rate, dim=0)
        repeat_neg = embedding_from.repeat_interleave(negative_sample_rate, dim=0)
        shuffled = torch.randperm(repeat_neg.shape[0], device=device)
        embedding_neg_from = repeat_neg[shuffled]

        # One concatenated distance vector, positives then negatives -- the layout the
        # constant `probabilities_graph` above is built to match.
        distance_embedding = torch.cat([
            ((embedding_to - embedding_from) ** 2).sum(1),
            ((embedding_neg_to - embedding_neg_from) ** 2).sum(1),
        ]).clamp_min(eps)

        log_probabilities_distance = convert_distance_to_log_probability(
            distance_embedding, a, b
        )
        attraction_term, repellant_term, ce_loss = compute_cross_entropy(
            probabilities_graph, log_probabilities_distance,
            repulsion_strength=repulsion_strength,
        )
        # A single mean over the pooled array, matching `ops.mean(ce_loss)`. This is what
        # weights each positive edge and each of its `rate` negative samples as one event
        # apiece; taking two separate group means and adding them would weight the blocks
        # equally regardless of size, which is a different objective.
        loss = ce_loss.mean()

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        # umap-learn clips gradients at 4.0. Attraction is unbounded above when a positive
        # pair is driven apart, so an unclipped step can wreck a good warm start.
        torch.nn.utils.clip_grad_value_(encoder.parameters(), 4.0)
        optimiser.step()
        schedule.step()

        if step % max(1, steps // 20) == 0 or step == steps - 1:
            history.append(round(float(loss.item()), 6))
            # Only the first block is nonzero in attraction and only the rest in repulsion,
            # so normalise each by its own count rather than the pooled length.
            log(f"    umap  step {step + 1}/{steps}  loss {loss.item():.4f} "
                f"(attract {attraction_term.sum().item() / batch_size:.4f}, "
                f"repel {repellant_term.sum().item() / (batch_size * negative_sample_rate):.4f})")
        if on_checkpoint is not None and checkpoint_every and (step + 1) % checkpoint_every == 0:
            on_checkpoint(step + 1, encoder)
            encoder.train()
    return history


# ── comparison helper ──────────────────────────────────────────────────────────

def procrustes_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Best rotation / reflection / uniform scale / translation of ``source`` onto ``target``.

    The UMAP objective is invariant to all four, so an encoder trained under ``umap`` loss
    lands in an arbitrary frame. Without this, raw coordinate error against cuML's
    embedding would measure the frame mismatch rather than the quality of the map.
    kNN-overlap needs no such correction, which is exactly why it is the primary metric.
    """
    source_centre, target_centre = source.mean(0), target.mean(0)
    source_c, target_c = source - source_centre, target - target_centre
    source_norm = np.linalg.norm(source_c)
    if source_norm == 0:
        return source.copy()
    u, singular_values, vt = np.linalg.svd(source_c.T @ target_c)
    rotation = u @ vt
    scale = singular_values.sum() / (source_norm ** 2)
    return (source_c @ rotation) * scale + target_centre


# ── validation entry point ─────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and validate parametric UMAP")
    parser.add_argument("--embed-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", default="hybrid", choices=["regress", "umap", "hybrid"])
    parser.add_argument("--fit-rows", type=int, default=5_000_000)
    parser.add_argument("--probe-rows", type=int, default=200_000)
    parser.add_argument("--hidden", default="256,256,128")
    parser.add_argument("--regress-steps", type=int, default=15_000)
    parser.add_argument("--umap-steps", type=int, default=30_000,
                        help="30k x batch 16384 is ~500 M edge samples. umap-learn's own "
                             "default schedule (n_epochs=200 over the expanded edge list) "
                             "is nearer 2 B, so raise this if quality looks step-limited "
                             "-- the loss history in the JSON shows whether it plateaued.")
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--umap-learning-rate", type=float, default=2e-4,
                        help="lower than --learning-rate: in hybrid mode this fine-tunes "
                             "an already-good map and a large step undoes the warm start")
    parser.add_argument("--negative-sample-rate", type=int, default=None,
                        help="default: read from the fitted cuML model (5), so the network "
                             "sees the same repulsion balance the target embedding was "
                             "optimised under. The Sainburg et al. paper code used 2.")
    parser.add_argument("--repulsion-strength", type=float, default=1.0)
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.0)
    parser.add_argument("--infer-batch-size", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-budget-gb", type=float, default=None)
    parser.add_argument("--save-encoder", action="store_true")
    return parser.parse_args()


def main() -> int:
    started = perf_counter()
    args = parse_args()
    hidden = tuple(int(part) for part in args.hidden.split(",") if part.strip())

    embed_dir = Path(args.embed_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    latent_path = embed_dir / "latent.npy"
    if not latent_path.exists():
        raise SystemExit(f"latent.npy not found in {embed_dir}")

    # Both allocators are live in this script -- cuML fits the UMAP, torch trains the net --
    # so the budget has to be split rather than handed to one of them.
    if core.gpu_available() and args.gpu_budget_gb is not None:
        core.apply_gpu_budget("apply", budget_gb=args.gpu_budget_gb)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"torch device: {device}")

    latent = np.load(latent_path, mmap_mode="r")
    total_rows, latent_dim = latent.shape
    log(f"{latent_path.name}: {total_rows:,} rows x {latent_dim} dims")

    rng = np.random.default_rng(args.seed)
    fit_idx = np.sort(rng.choice(total_rows, size=args.fit_rows, replace=False))
    outside = np.setdiff1d(np.arange(total_rows), fit_idx, assume_unique=True)
    probe_idx = np.sort(
        np.random.default_rng(args.seed + 1).choice(
            outside, size=min(args.probe_rows, len(outside)), replace=False
        )
    )
    log(f"fit set {len(fit_idx):,} rows, probe set {len(probe_idx):,} held-out rows")

    config = core.UmapConfig(
        n_neighbors=args.umap_n_neighbors, min_dist=args.umap_min_dist,
        n_components=2, seed=args.seed, n_epochs=200,
    )
    log(f"UMAP config: {config.as_dict()}")

    fit_latent = np.ascontiguousarray(latent[fit_idx])
    log(f"fitting UMAP on {len(fit_idx):,} rows ...")
    t0 = perf_counter()
    fitted = core.fit_umap(fit_latent, config)
    umap_fit_seconds = perf_counter() - t0
    log(f"  fit in {umap_fit_seconds:.1f}s (backend={fitted.backend})")

    probe_latent = np.ascontiguousarray(latent[probe_idx])
    log(f"true transform() of {len(probe_idx):,} probe rows (ground truth) ...")
    t0 = perf_counter()
    true_coords = fitted.transform(probe_latent)
    true_transform_seconds = perf_counter() - t0
    true_full_minutes = true_transform_seconds * FULL_COHORT_HELD_ROWS / len(probe_idx) / 60
    log(f"  transform in {true_transform_seconds:.1f}s (-> {true_full_minutes:.0f} min full cohort)")

    # Raw latents in, raw cuML coordinates out -- no normalisation on either side. Both
    # umap-learn's parametric module and the Sainburg et al. paper code leave scaling to
    # the caller, and the UMAP loss needs cuML's own coordinate units regardless because
    # a and b encode an absolute distance scale.
    X = torch.from_numpy(fit_latent).to(device)
    Y = torch.from_numpy(fitted.embedding).to(device)

    encoder = ParametricEncoder(latent_dim, output_dim=2, hidden=hidden).to(device)
    n_params = sum(p.numel() for p in encoder.parameters())
    log(f"encoder {latent_dim} -> {' -> '.join(map(str, hidden))} -> 2 "
        f"({n_params:,} parameters)")

    train_started = perf_counter()
    regress_history: list[float] = []
    umap_history: list[float] = []

    if args.mode in ("regress", "hybrid"):
        log(f"training: MSE regression onto cuML embedding, {args.regress_steps:,} steps")
        regress_history = train_regression(
            encoder, X, Y, steps=args.regress_steps, batch_size=args.batch_size,
            learning_rate=args.learning_rate, device=device,
        )

    if args.mode in ("umap", "hybrid"):
        a, b = ab_params(args.umap_min_dist)
        # Take the repulsion rate from the model that produced the target embedding rather
        # than a literal, so the network is trained under the same attraction/repulsion
        # balance cuML used. cuML and umap-learn default to 5; the paper's code used 2.
        negative_sample_rate = args.negative_sample_rate
        if negative_sample_rate is None:
            negative_sample_rate = int(getattr(fitted.model, "negative_sample_rate", 5))
        log(f"UMAP loss parameters: a={a:.5f} b={b:.5f} "
            f"(min_dist={args.umap_min_dist}), negative_sample_rate={negative_sample_rate}")
        head_np, tail_np, weight_np = extract_edges(fitted.model.graph_)
        log(f"graph: {len(head_np):,} undirected edges "
            f"(weight min {weight_np.min():.4f}, max {weight_np.max():.4f})")
        edge_head = torch.from_numpy(head_np).to(device)
        edge_tail = torch.from_numpy(tail_np).to(device)
        # float64: a float32 running sum saturates at ~1.7e7, which a 5 M-row nn=15 graph
        # (~3.7e7 edges) already exceeds -- every draw past that point would resolve to the
        # same edge and most of the graph would never be sampled.
        edge_cumsum = torch.cumsum(torch.from_numpy(weight_np).to(device).double(), 0)

        learning_rate = args.umap_learning_rate if args.mode == "hybrid" else args.learning_rate
        log(f"training: UMAP cross-entropy, {args.umap_steps:,} steps, lr={learning_rate}")
        umap_history = train_umap_loss(
            encoder, X, edge_head, edge_tail, edge_cumsum, a=a, b=b,
            steps=args.umap_steps, batch_size=args.batch_size,
            learning_rate=learning_rate,
            negative_sample_rate=negative_sample_rate,
            repulsion_strength=args.repulsion_strength, device=device,
        )

    train_seconds = perf_counter() - train_started
    log(f"training complete in {train_seconds:.1f}s")

    model = ParametricUmap(encoder=encoder, device=device, mode=args.mode)

    log(f"parametric transform of {len(probe_idx):,} probe rows ...")
    t0 = perf_counter()
    approx_coords = model.transform(probe_latent, batch_size=args.infer_batch_size)
    infer_seconds = perf_counter() - t0
    infer_full_seconds = infer_seconds * FULL_COHORT_HELD_ROWS / len(probe_idx)
    log(f"  inference in {infer_seconds:.2f}s "
        f"(-> {infer_full_seconds / 60:.1f} min full cohort)")

    # kNN-overlap is frame-invariant; coordinate error is not, so align first. In regress
    # mode the alignment should be near-identity -- if it is not, the regression drifted.
    aligned = procrustes_align(approx_coords, true_coords)
    quality_aligned = compare_embeddings(true_coords, aligned, k=15)
    quality_raw = compare_embeddings(true_coords, approx_coords, k=15)

    total_current = umap_fit_seconds + true_transform_seconds * FULL_COHORT_HELD_ROWS / len(probe_idx)
    total_parametric = umap_fit_seconds + train_seconds + infer_full_seconds

    results = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "total_rows": int(total_rows),
        "latent_dim": int(latent_dim),
        "umap_config": config.as_dict(),
        "encoder": {
            "hidden": list(hidden),
            "parameters": int(n_params),
            "regress_steps": args.regress_steps if args.mode in ("regress", "hybrid") else 0,
            "umap_steps": args.umap_steps if args.mode in ("umap", "hybrid") else 0,
            "batch_size": args.batch_size,
            "negative_sample_rate": (
                negative_sample_rate if args.mode in ("umap", "hybrid") else None
            ),
        },
        "fit_rows": int(len(fit_idx)),
        "probe_rows": int(len(probe_idx)),
        "timing": {
            "umap_fit_seconds": round(umap_fit_seconds, 2),
            "encoder_train_seconds": round(train_seconds, 2),
            "true_transform_probe_seconds": round(true_transform_seconds, 2),
            "true_transform_full_cohort_minutes": round(true_full_minutes, 1),
            "parametric_probe_seconds": round(infer_seconds, 3),
            "parametric_full_cohort_minutes": round(infer_full_seconds / 60, 2),
            "inference_speedup": round(
                (true_transform_seconds / infer_seconds) if infer_seconds > 0 else float("nan"), 1
            ),
            "end_to_end_current_minutes": round(total_current / 60, 1),
            "end_to_end_parametric_minutes": round(total_parametric / 60, 1),
        },
        "quality_procrustes_aligned": quality_aligned,
        "quality_raw": quality_raw,
        "loss_history": {"regression_mse": regress_history, "umap_ce": umap_history},
        "aumap_baseline": AUMAP_BASELINE,
        "total_seconds": round(perf_counter() - started, 1),
    }
    out_path = output_dir / f"parametric_umap_{args.mode}.json"
    out_path.write_text(json.dumps(results, indent=2))
    log(f"wrote {out_path}")

    if args.save_encoder:
        encoder_path = output_dir / f"parametric_encoder_{args.mode}.pt"
        model.save(encoder_path)
        log(f"wrote {encoder_path}")

    overlap = quality_aligned["knn_overlap"]
    print(f"\n-- parametric UMAP ({args.mode}) -------------------------")
    print(f"probe rows: {len(probe_idx):,} held out of a {len(fit_idx):,}-row fit")
    print()
    print(f"UMAP fit        : {umap_fit_seconds:7.1f}s  (unchanged, still required)")
    print(f"encoder train   : {train_seconds:7.1f}s  (one-off, per UMAP fit)")
    print(f"true transform  : {true_transform_seconds:7.2f}s probe -> "
          f"{true_full_minutes:.0f} min full cohort")
    print(f"parametric infer: {infer_seconds:7.2f}s probe -> "
          f"{infer_full_seconds / 60:.1f} min full cohort "
          f"({results['timing']['inference_speedup']:.0f}x)")
    print()
    print(f"end to end   current {results['timing']['end_to_end_current_minutes']:.1f} min"
          f"   ->   parametric {results['timing']['end_to_end_parametric_minutes']:.1f} min")
    print()
    print("quality vs true transform() (Procrustes-aligned):")
    print(f"  kNN-overlap   {overlap:.3f}      <- decides it")
    print(f"  mean error    {quality_aligned['mean_error_in_std']:.3f} embedding std")
    print(f"  p95 error     {quality_aligned['p95_error_in_std']:.3f} embedding std")
    print(f"  aUMAP was     {AUMAP_BASELINE['knn_overlap']:.3f} overlap "
          f"at {AUMAP_BASELINE['full_cohort_minutes']:.1f} min")
    print("--------------------------------------------------------")
    if overlap < AUMAP_BASELINE["knn_overlap"]:
        print("\nWorse than aUMAP. Do not tune this further before checking the Gate 1")
        print("determinism floor -- if two identical cuML fits already disagree at this")
        print("level, no approximation can be shown to beat it and the comparison is moot.")
    elif overlap < 0.7:
        print("\nBetter than aUMAP but still low in absolute terms. Worth trying more")
        print("--umap-steps and --mode umap before judging; also measure the Gate 1 floor,")
        print("which is the only thing that says what 'good enough' means here.")
    else:
        print("\nStrong. Confirm it holds where it matters by clustering both embeddings")
        print("and comparing labels (ARI) -- neighbourhood overlap is a proxy for that,")
        print("not a substitute.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
