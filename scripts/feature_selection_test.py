"""Test different feature-selection methods for the TabularVAE.

Goal
----
Find the *smallest* subset of input features that still produces essentially the
same latent space as training on all features, so that VAE training + inference
(and everything downstream: UMAP -> HDBSCAN clustering -> UV signatures) runs on
fewer columns and less compute. This is README todo item 10:
"drop unimportant columns after training run and retrain -> check difference with
procrustes, etc."

Because the VAE is unsupervised and feeds *unsupervised* clustering, there is no
label to select against. Every method here is therefore either model-free
(distribution of the raw feature) or self-supervised (measured against the VAE's
own latent space). Nothing peeks at a target.

------------------------------------------------------------------------------
HOW FEATURES ARE SELECTED  (five rankings, each maps feature -> keep-score)
------------------------------------------------------------------------------
1. variance / low-information filter  [model-free, pre-training]
     score = normalized Shannon entropy of the feature's distribution
             (category frequencies for categoricals; 20 quantile bins for
             numerics), in [0, 1]. Near-0 => near-constant => carries no signal.

2. redundancy / collinearity filter   [model-free, pre-training]
     score = entropy * (1 - max |Spearman r| to any higher-entropy numeric
             feature). An mRMR-flavoured relevance*(1-redundancy) score: an
             informative feature that merely duplicates a stronger one (e.g. one
             of l1..l7 / q2..q6 if they co-vary) is pushed down. Categoricals
             have no well-defined numeric correlation here, so they keep their
             entropy score.

3. encoder first-layer weight norm     [model-based, cheap]
     score = L2 norm of the trained encoder's first Linear weights on that
             feature's input columns (an embedding block for categoricals, one
             column for numerics), normalized by sqrt(block width). How much the
             encoder "listens" to the feature.

4. latent permutation importance       [model-based, PRIMARY]
     Shuffle one feature across rows, re-encode the fixed reference set with the
     unchanged baseline model, and measure how far the latent geometry moved:
             score = 0.5 * procrustes_distance + 0.5 * (1 - linear_CKA)
     Big move => the latent space genuinely depends on that feature. This is the
     ranking that most directly matches "does removing it change the space".

5. mutual information to latent         [model-based, optional --mutual-info]
     score = sum over latent dims of MI(latent ; feature). How much information
     the latent actually retained about the feature. Uses mutual_info_classif
     for categoricals, mutual_info_regression for numerics.

Structural / matrix-factorization rankings [model-free, all scikit-learn].
These operate on a per-feature surrogate matrix (one standardized column per
original feature: numerics standardized; each categorical summarized by the top
principal coordinate of its one-hot, so every feature maps 1:1 to a column). They
describe the linear/statistical structure of the features independent of the VAE,
which complements the VAE-based permutation ranking.

6. PCA loadings          [sklearn.decomposition.PCA]
     Keep the top PCs up to --pca-variance cumulative explained variance;
     score = sum_c explained_variance_ratio_c * |loading_{c,feature}|. Features
     that drive the high-variance directions score high.

7. ICA contribution      [sklearn.decomposition.FastICA]
     score = sum over independent components of |mixing_{feature,c}|. Features
     that participate in the statistically-independent sources score high.

8. NMF basis weight      [sklearn.decomposition.NMF]
     On a non-negative (min-max) surrogate; score = sum_c prevalence_c *
     H_{c,feature}. Parts-based factorization: features with large additive
     basis weight score high.

9. feature clustering    [sklearn.cluster.AgglomerativeClustering]
     "Cluster similar features, keep representatives." Cluster features on
     1 - |correlation| into --n-clusters groups; the highest-entropy feature in
     each group is the representative (full score), redundant members are
     down-weighted. (Variance Threshold and Correlation Filtering from the same
     table are already methods 1 and 2 above.)

------------------------------------------------------------------------------
HOW SUBSETS ARE EVALUATED  (the metrics, reusing uv_vae.evaluation)
------------------------------------------------------------------------------
For a candidate subset we retrain a VAE *on those features only* (with the same
early stopping as the real pipeline), encode the fixed reference set, and compare
that latent to the all-feature baseline latent with the exact metrics used
elsewhere in this project (`compare_latent_spaces`):
    procrustes_distance   lower  = closer to baseline geometry
    cka_similarity        higher = closer   (headline fidelity number)
    jaccard_knn10 / 30    higher = same local neighbourhoods
    trustworthiness       higher = local structure preserved
    continuity            higher = no false neighbours introduced
plus active_latent_dims_pct (Burda active units / latent_dim), final val loss,
train seconds, epochs actually run (early-stopping savings), and the input
dimensionality. Optionally (--clustering) we also cluster both latents with
HDBSCAN (optionally UMAP first) and report ARI / NMI agreement, closing the loop
on the real downstream task.

A random-subset baseline of the same size is evaluated too, so "the method beats
picking columns at random" is something you can see rather than assume.

Early stopping is on for every run (baseline and every subset): --epochs is a
ceiling, --patience decides when to stop, so a bad/tiny subset that plateaus
early does not waste the full budget.

Example
-------
    python scripts/feature_selection_test.py \
        --parquet-path /path/wt0-12-ppm0050.featuremap.parquet \
        --sample-rows 200000 --reference-rows 4000 \
        --epochs 40 --patience 6 \
        --eval-keep 16 --keep-sizes 8,12,16,20,24 \
        --clustering hdbscan \
        --output-dir feature_selection_results

Defaults are tuned for fast iteration (small sample, aggressive early stopping).
Scale --sample-rows up toward the 2.5M-5M sweet spot before making a final call.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from scipy.spatial import procrustes
from torch.utils.data import DataLoader, TensorDataset

from uv_vae.data import (
    connect_duckdb,
    get_non_null_counts,
    get_row_count,
    quote_ident,
    sample_frame,
)
from uv_vae.early_stopping import (
    DEFAULT_ACTIVE_UNIT_THRESHOLD,
    EarlyStoppingConfig,
    EarlyStoppingMonitor,
    compute_latent_diagnostics,
)
from uv_vae.evaluation import compare_clusterings, compare_latent_spaces
from uv_vae.features import FeatureSpec, load_feature_specs
from uv_vae.preprocess import (
    encode_categorical_column,
    encode_numeric_column,
    infer_embedding_dim,
    prepare_tensors,
    transform_frame,
)
from uv_vae.training import (
    DEFAULT_TRAINING_ROW_FILTER,
    build_model,
    run_epoch,
    seed_everything,
)

DEFAULT_FEATURE_SPEC_PATH = "ml_features.json"
DEFAULT_SAMPLE_ROWS = 200_000
DEFAULT_REFERENCE_ROWS = 4_000
DEFAULT_KEEP_SIZES = "8,12,16,20,24"
PRIMARY_METHOD = "permutation"
ALL_METHODS = (
    "variance",
    "redundancy",
    "feature_clustering",
    "pca",
    "ica",
    "nmf",
    "encoder_weights",
    "permutation",
    "mutual_info",
)
STRUCTURAL_METHODS = frozenset({"pca", "ica", "nmf", "feature_clustering"})


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# In-memory training (mirrors uv_vae.early_stopping.train_with_early_stopping,
# but on an already-loaded frame + an arbitrary feature subset, no disk writes).
# ---------------------------------------------------------------------------
@dataclass
class TrainedVAE:
    model: torch.nn.Module
    categorical_specs: list[FeatureSpec]
    numeric_specs: list[FeatureSpec]
    numeric_means: dict[str, float]
    numeric_stds: dict[str, float]
    latent_dim: int
    device: torch.device
    history: list[dict]
    epochs_run: int
    train_seconds: float
    final_active_units: int
    input_dim: int

    @property
    def feature_names(self) -> list[str]:
        return [s.name for s in self.categorical_specs] + [s.name for s in self.numeric_specs]

    @property
    def active_pct(self) -> float:
        return self.final_active_units / self.latent_dim if self.latent_dim else 0.0

    @property
    def final_val_loss(self) -> float:
        return float(self.history[-1]["val_total_loss"]) if self.history else float("nan")


def _input_dim(prepared) -> int:
    cat_dim = sum(infer_embedding_dim(spec.cardinality) for spec in prepared.categorical_specs)
    return cat_dim + len(prepared.numeric_specs)


def train_vae_in_memory(
    frame: pl.DataFrame,
    specs: list[FeatureSpec],
    non_null_counts: dict[str, int],
    *,
    hidden_dims: list[int],
    latent_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    kl_weight: float,
    train_fraction: float,
    seed: int,
    early_stopping: EarlyStoppingConfig,
) -> TrainedVAE:
    """Train one VAE on `frame` restricted to `specs`, with early stopping."""
    seed_everything(seed)
    prepared = prepare_tensors(
        frame=frame,
        specs=specs,
        non_null_counts=non_null_counts,
        total_rows=frame.height,
        train_fraction=train_fraction,
        seed=seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        TensorDataset(prepared.train_cat, prepared.train_num, prepared.train_mask),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        TensorDataset(prepared.val_cat, prepared.val_num, prepared.val_mask),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = build_model(prepared, hidden_dims=hidden_dims, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    monitor = EarlyStoppingMonitor(early_stopping)
    history: list[dict] = []
    best_state: dict[str, torch.Tensor] | None = None
    final_active_units = 0
    start = perf_counter()

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, kl_weight, optimizer)
        val_metrics = run_epoch(model, val_loader, device, kl_weight)
        diagnostics = compute_latent_diagnostics(
            model, val_loader, device, early_stopping.active_unit_threshold
        )
        val_loss = val_metrics["total_loss"]
        final_active_units = int(diagnostics["active_units"])
        history.append(
            {
                "epoch": epoch,
                "train_total_loss": train_metrics["total_loss"],
                "val_total_loss": val_metrics["total_loss"],
                "val_numeric_loss": val_metrics["numeric_loss"],
                "val_categorical_loss": val_metrics["categorical_loss"],
                "val_kl_loss": val_metrics["kl_loss"],
                "active_units": final_active_units,
            }
        )
        if monitor.is_new_best(val_loss):
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if monitor.update(epoch=epoch, val_loss=val_loss, active_units=final_active_units):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    return TrainedVAE(
        model=model,
        categorical_specs=prepared.categorical_specs,
        numeric_specs=prepared.numeric_specs,
        numeric_means=prepared.numeric_means,
        numeric_stds=prepared.numeric_stds,
        latent_dim=latent_dim,
        device=device,
        history=history,
        epochs_run=len(history),
        train_seconds=perf_counter() - start,
        final_active_units=final_active_units,
        input_dim=_input_dim(prepared),
    )


def encode_frame(trained: TrainedVAE, frame: pl.DataFrame, batch_size: int = 4096) -> np.ndarray:
    """Encode `frame` to latent posterior means with a trained model's own specs."""
    categorical_inputs, numeric_inputs = transform_frame(
        frame=frame,
        categorical_specs=trained.categorical_specs,
        numeric_specs=trained.numeric_specs,
        numeric_means=trained.numeric_means,
        numeric_stds=trained.numeric_stds,
    )
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, frame.height, batch_size):
            stop = start + batch_size
            batch_cat = categorical_inputs[start:stop].to(trained.device, non_blocking=True)
            batch_num = numeric_inputs[start:stop].to(trained.device, non_blocking=True)
            mu, _ = trained.model.encode(batch_cat, batch_num)
            chunks.append(mu.detach().cpu().numpy().astype(np.float32, copy=False))
    if not chunks:
        return np.zeros((0, trained.latent_dim), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


# ---------------------------------------------------------------------------
# Small metric helpers
# ---------------------------------------------------------------------------
def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA in feature space (1 = identical geometry, 0 = unrelated)."""
    x = np.asarray(x, dtype=np.float64) - np.asarray(x, dtype=np.float64).mean(axis=0)
    y = np.asarray(y, dtype=np.float64) - np.asarray(y, dtype=np.float64).mean(axis=0)
    numerator = float(np.sum((x.T @ y) ** 2))
    denom = float(np.sqrt(np.sum((x.T @ x) ** 2) * np.sum((y.T @ y) ** 2)))
    return numerator / denom if denom > 1e-12 else 0.0


def latent_drift(reference: np.ndarray, permuted: np.ndarray) -> float:
    """Cheap latent-geometry drift used for permutation importance ranking."""
    _, _, disparity = procrustes(reference, permuted)
    return 0.5 * float(disparity) + 0.5 * (1.0 - linear_cka(reference, permuted))


def normalized_entropy_numeric(values: np.ndarray, bins: int = 20) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    quantiles = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, bins + 1)))
    if quantiles.size < 2:
        return 0.0
    counts, _ = np.histogram(finite, bins=quantiles)
    return _entropy_ratio(counts)


def normalized_entropy_categorical(codes: np.ndarray, cardinality: int) -> float:
    if codes.size == 0:
        return 0.0
    _, counts = np.unique(codes, return_counts=True)
    if counts.size < 2:  # single observed category -> no information
        return 0.0
    probs = counts / counts.sum()  # all strictly > 0, so log is safe
    entropy = float(-np.sum(probs * np.log(probs)))
    denom = np.log(max(cardinality, 2))
    return entropy / denom if denom > 0 else 0.0


def _entropy_ratio(counts: np.ndarray) -> float:
    counts = counts[counts > 0].astype(np.float64)
    if counts.size < 2:
        return 0.0
    probs = counts / counts.sum()
    entropy = float(-np.sum(probs * np.log(probs)))
    return entropy / np.log(counts.size)


# ---------------------------------------------------------------------------
# Feature-importance rankings (each returns {feature_name: keep_score})
# ---------------------------------------------------------------------------
def rank_by_variance(frame: pl.DataFrame, specs: list[FeatureSpec]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for spec in specs:
        if spec.name not in frame.columns:
            continue
        if spec.is_categorical:
            codes = encode_categorical_column(frame, spec)
            scores[spec.name] = normalized_entropy_categorical(codes, spec.cardinality)
        else:
            values, _ = encode_numeric_column(frame, spec)
            scores[spec.name] = normalized_entropy_numeric(values.astype(np.float64))
    return scores


def rank_by_redundancy(
    frame: pl.DataFrame,
    specs: list[FeatureSpec],
    entropy_scores: dict[str, float],
    corr_threshold: float = 0.95,
) -> dict[str, float]:
    """entropy * (1 - max |Spearman r| to a higher-entropy numeric feature)."""
    numeric_specs = [s for s in specs if s.is_numeric and s.name in frame.columns]
    scores: dict[str, float] = dict(entropy_scores)

    if len(numeric_specs) >= 2:
        columns = []
        names = []
        for spec in numeric_specs:
            values, _ = encode_numeric_column(frame, spec)
            values = values.astype(np.float64)
            mean = np.nanmean(values) if np.isfinite(values).any() else 0.0
            columns.append(np.where(np.isfinite(values), values, mean))
            names.append(spec.name)
        matrix = np.vstack([_rankdata(col) for col in columns])  # Spearman == Pearson on ranks
        corr = np.corrcoef(matrix)
        corr = np.nan_to_num(corr, nan=0.0)

        order = sorted(range(len(names)), key=lambda i: entropy_scores.get(names[i], 0.0), reverse=True)
        for rank_pos, idx in enumerate(order):
            stronger = order[:rank_pos]
            max_corr = max((abs(corr[idx, j]) for j in stronger), default=0.0)
            # Only penalise a feature once it is collinear (past the threshold)
            # with an already-kept, higher-entropy feature.
            penalty = (1.0 - max_corr) if max_corr > corr_threshold else 1.0
            scores[names[idx]] = entropy_scores.get(names[idx], 0.0) * penalty
    return scores


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = values.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def rank_by_encoder_weights(trained: TrainedVAE) -> dict[str, float]:
    model = trained.model
    first_linear = model.encoder[0]
    weight = first_linear.weight.detach().cpu().numpy()  # (hidden0, input_dim)
    col_norm = np.linalg.norm(weight, axis=0)  # per input dimension

    scores: dict[str, float] = {}
    idx = 0
    for name in model.categorical_names:
        dim = model.config.embedding_dims[name]
        block = col_norm[idx : idx + dim]
        scores[name] = float(np.sqrt(np.sum(block**2)) / np.sqrt(dim))
        idx += dim
    for offset, spec in enumerate(trained.numeric_specs):
        scores[spec.name] = float(col_norm[idx + offset])
    return scores


def rank_by_latent_permutation(
    baseline: TrainedVAE,
    reference_frame: pl.DataFrame,
    reference_latent: np.ndarray,
    seed: int,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for i, name in enumerate(baseline.feature_names):
        permuted_frame = reference_frame.with_columns(pl.col(name).shuffle(seed=seed + i))
        permuted_latent = encode_frame(baseline, permuted_frame)
        scores[name] = latent_drift(reference_latent, permuted_latent)
    return scores


def rank_by_mutual_info(
    baseline: TrainedVAE,
    reference_frame: pl.DataFrame,
    reference_latent: np.ndarray,
    seed: int,
    max_rows: int = 4000,
) -> dict[str, float]:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    n = reference_latent.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)[: min(max_rows, n)]
    latent = reference_latent[idx]

    scores: dict[str, float] = {}
    for spec in baseline.categorical_specs:
        codes = encode_categorical_column(reference_frame, spec)[idx]
        if np.unique(codes).size < 2:
            scores[spec.name] = 0.0
            continue
        mi = mutual_info_classif(latent, codes, discrete_features=False, random_state=seed)
        scores[spec.name] = float(np.sum(mi))
    for spec in baseline.numeric_specs:
        values, _ = encode_numeric_column(reference_frame, spec)
        values = values[idx].astype(np.float64)
        mean = np.nanmean(values) if np.isfinite(values).any() else 0.0
        target = np.where(np.isfinite(values), values, mean)
        if np.std(target) < 1e-9:
            scores[spec.name] = 0.0
            continue
        mi = mutual_info_regression(latent, target, random_state=seed)
        scores[spec.name] = float(np.sum(mi))
    return scores


# ---------------------------------------------------------------------------
# Structural / matrix-factorization rankings (all scikit-learn).
# They share a per-feature surrogate matrix: one standardized column per original
# feature so every method maps 1:1 to features. Categoricals are summarized by the
# top principal coordinate of their one-hot encoding (a 1-D linear summary).
# ---------------------------------------------------------------------------
def build_feature_surrogate_matrix(
    frame: pl.DataFrame, specs: list[FeatureSpec]
) -> tuple[list[str], np.ndarray]:
    from sklearn.decomposition import PCA

    names: list[str] = []
    columns: list[np.ndarray] = []
    for spec in specs:
        if spec.name not in frame.columns:
            continue
        if spec.is_numeric:
            values, _ = encode_numeric_column(frame, spec)
            column = values.astype(np.float64)
            mean = float(np.nanmean(column)) if np.isfinite(column).any() else 0.0
            column = np.where(np.isfinite(column), column, mean)
        else:
            codes = encode_categorical_column(frame, spec)
            if np.unique(codes).size < 2:
                column = np.zeros(codes.size, dtype=np.float64)
            else:
                one_hot = np.zeros((codes.size, spec.cardinality), dtype=np.float64)
                one_hot[np.arange(codes.size), codes] = 1.0
                column = PCA(n_components=1, random_state=0).fit_transform(one_hot).ravel()
        std = float(np.std(column))
        column = (column - column.mean()) / std if std > 1e-9 else np.zeros_like(column)
        names.append(spec.name)
        columns.append(column)

    matrix = np.column_stack(columns) if columns else np.zeros((frame.height, 0), dtype=np.float64)
    return names, matrix


def rank_by_pca(names: list[str], matrix: np.ndarray, variance: float) -> dict[str, float]:
    from sklearn.decomposition import PCA

    pca = PCA(random_state=0).fit(matrix)
    evr = pca.explained_variance_ratio_
    # Smallest #components whose cumulative variance reaches `variance`; the 1e-9
    # tolerance keeps exact ties (e.g. cumsum 0.8999999 vs target 0.9) at the
    # intended component count instead of grabbing one more.
    k = int(np.searchsorted(np.cumsum(evr), variance - 1e-9) + 1)
    k = max(1, min(k, len(evr)))
    loadings = np.abs(pca.components_[:k])  # (k, n_features)
    scores = (evr[:k, None] * loadings).sum(axis=0)
    return {name: float(scores[i]) for i, name in enumerate(names)}


def rank_by_ica(names: list[str], matrix: np.ndarray, n_components: int, seed: int) -> dict[str, float]:
    from sklearn.decomposition import FastICA

    k = max(1, min(n_components, matrix.shape[1]))
    try:
        ica = FastICA(n_components=k, random_state=seed, max_iter=1000, whiten="unit-variance")
        ica.fit(matrix)
        mixing = np.abs(ica.mixing_)  # (n_features, k)
    except Exception as exc:  # non-convergence / older sklearn API
        log(f"ICA failed ({exc}); falling back to |whitened correlation| magnitude")
        mixing = np.abs(np.corrcoef(matrix.T))
    scores = mixing.sum(axis=1)
    return {name: float(scores[i]) for i, name in enumerate(names)}


def rank_by_nmf(names: list[str], matrix: np.ndarray, n_components: int, seed: int) -> dict[str, float]:
    from sklearn.decomposition import NMF

    lower = matrix.min(axis=0)
    span = matrix.max(axis=0) - lower
    span = np.where(span > 1e-9, span, 1.0)
    non_negative = (matrix - lower) / span  # min-max to [0, 1]

    k = max(1, min(n_components, matrix.shape[1]))
    model = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=1000)
    weights = model.fit_transform(non_negative)  # (n, k)
    components = model.components_  # (k, n_features)
    prevalence = weights.mean(axis=0)  # how much each basis is used
    scores = (prevalence[:, None] * components).sum(axis=0)
    return {name: float(scores[i]) for i, name in enumerate(names)}


def rank_by_feature_clustering(
    names: list[str], matrix: np.ndarray, entropy_scores: dict[str, float], n_clusters: int
) -> dict[str, float]:
    from sklearn.cluster import AgglomerativeClustering

    n_features = matrix.shape[1]
    k = max(1, min(n_clusters, n_features))
    correlation = np.nan_to_num(np.corrcoef(matrix.T), nan=0.0)
    distance = 1.0 - np.abs(correlation)
    np.fill_diagonal(distance, 0.0)
    try:
        labels = AgglomerativeClustering(
            n_clusters=k, metric="precomputed", linkage="average"
        ).fit_predict(distance)
    except TypeError:  # sklearn < 1.2 used `affinity`
        labels = AgglomerativeClustering(
            n_clusters=k, affinity="precomputed", linkage="average"
        ).fit_predict(distance)

    scores: dict[str, float] = {}
    for cluster_id in set(labels):
        members = [names[i] for i in range(n_features) if labels[i] == cluster_id]
        representative = max(members, key=lambda nm: entropy_scores.get(nm, 0.0))
        for name in members:
            base = entropy_scores.get(name, 0.0)
            scores[name] = base if name == representative else base / len(members)
    return scores


def compute_structural_rankings(
    frame: pl.DataFrame,
    specs: list[FeatureSpec],
    entropy_scores: dict[str, float],
    needed: set[str],
    *,
    n_components: int,
    pca_variance: float,
    n_clusters: int,
    seed: int,
    max_rows: int,
) -> dict[str, dict[str, float]]:
    subframe = frame
    if frame.height > max_rows:
        subframe = frame.sample(n=max_rows, seed=seed)
    names, matrix = build_feature_surrogate_matrix(subframe, specs)

    rankings: dict[str, dict[str, float]] = {}
    if "pca" in needed:
        rankings["pca"] = rank_by_pca(names, matrix, pca_variance)
    if "ica" in needed:
        rankings["ica"] = rank_by_ica(names, matrix, n_components, seed)
    if "nmf" in needed:
        rankings["nmf"] = rank_by_nmf(names, matrix, n_components, seed)
    if "feature_clustering" in needed:
        rankings["feature_clustering"] = rank_by_feature_clustering(
            names, matrix, entropy_scores, n_clusters
        )
    return rankings


# ---------------------------------------------------------------------------
# Selection + evaluation
# ---------------------------------------------------------------------------
def ranked_names(scores: dict[str, float]) -> list[str]:
    """Feature names best-first; non-finite/unscored features sort last, ties by name."""
    def key(name: str) -> tuple[float, str]:
        score = scores[name]
        return (-score if np.isfinite(score) else float("inf"), name)

    return sorted(scores, key=key)


def top_k(scores: dict[str, float], k: int) -> list[str]:
    return ranked_names(scores)[:k]


@dataclass
class SubsetEvaluation:
    label: str
    method: str
    n_features: int
    input_dim: int
    train_seconds: float
    epochs_run: int
    active_pct: float
    val_total_loss: float
    procrustes_distance: float
    cka_similarity: float
    jaccard_knn10: float
    jaccard_knn30: float
    trustworthiness: float
    continuity: float
    ari: float = float("nan")
    nmi: float = float("nan")
    features: list[str] = field(default_factory=list)


def cluster_latent(
    latent: np.ndarray,
    method: str,
    *,
    min_cluster_size: int,
    min_samples: int,
    umap_neighbors: int,
    umap_min_dist: float,
    seed: int,
) -> np.ndarray | None:
    if method == "none":
        return None
    try:
        import hdbscan
    except ImportError:
        log("hdbscan not installed - skipping ARI/NMI clustering agreement")
        return None

    features = latent
    if method == "umap-hdbscan":
        try:
            from umap import UMAP
        except ImportError:
            log("umap-learn not installed - clustering directly on latent instead")
        else:
            features = UMAP(
                n_neighbors=umap_neighbors,
                min_dist=umap_min_dist,
                metric="euclidean",
                random_state=seed,
            ).fit_transform(latent)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        core_dist_n_jobs=1,
    )
    return clusterer.fit_predict(features.astype(np.float32, copy=False))


def evaluate_subset(
    *,
    label: str,
    method: str,
    feature_names: list[str],
    all_specs: list[FeatureSpec],
    train_frame: pl.DataFrame,
    reference_frame: pl.DataFrame,
    baseline_latent: np.ndarray,
    baseline_labels: np.ndarray | None,
    non_null_counts: dict[str, int],
    train_kwargs: dict,
    early_stopping: EarlyStoppingConfig,
    clustering: str,
    cluster_kwargs: dict,
    seed: int,
) -> SubsetEvaluation:
    keep = set(feature_names)
    subset_specs = [spec for spec in all_specs if spec.name in keep]
    subset_counts = {name: non_null_counts.get(name, 0) for name in keep}

    trained = train_vae_in_memory(
        train_frame,
        subset_specs,
        subset_counts,
        early_stopping=early_stopping,
        **train_kwargs,
    )
    subset_latent = encode_frame(trained, reference_frame)
    comparison = compare_latent_spaces(baseline_latent, subset_latent)

    ari = nmi = float("nan")
    if clustering != "none" and baseline_labels is not None:
        subset_labels = cluster_latent(subset_latent, clustering, seed=seed, **cluster_kwargs)
        if subset_labels is not None:
            agreement = compare_clusterings(baseline_labels, subset_labels)
            ari, nmi = agreement["ari"], agreement["nmi"]

    return SubsetEvaluation(
        label=label,
        method=method,
        n_features=len(trained.feature_names),
        input_dim=trained.input_dim,
        train_seconds=trained.train_seconds,
        epochs_run=trained.epochs_run,
        active_pct=trained.active_pct,
        val_total_loss=trained.final_val_loss,
        procrustes_distance=comparison["procrustes_distance"],
        cka_similarity=comparison["cka_similarity"],
        jaccard_knn10=comparison["jaccard_knn10"],
        jaccard_knn30=comparison["jaccard_knn30"],
        trustworthiness=comparison["trustworthiness"],
        continuity=comparison["continuity"],
        ari=ari,
        nmi=nmi,
        features=sorted(trained.feature_names),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_ranking_table(rankings: dict[str, dict[str, float]], all_names: list[str]) -> None:
    methods = [m for m in ALL_METHODS if m in rankings]
    header = f"{'feature':<14}" + "".join(f"{m[:12]:>14}" for m in methods)
    print("\n=== Feature-importance rankings (rank #, 1 = most important) ===")
    print(header)
    print("-" * len(header))
    order_by_method = {m: {name: pos + 1 for pos, name in enumerate(ranked_names(rankings[m]))} for m in methods}
    # Order rows by average rank across methods for readability.
    def avg_rank(name: str) -> float:
        return float(np.mean([order_by_method[m].get(name, len(all_names)) for m in methods]))

    for name in sorted(all_names, key=avg_rank):
        cells = "".join(f"{order_by_method[m].get(name, 0):>14}" for m in methods)
        print(f"{name:<14}{cells}")


def print_evaluation_table(evaluations: list[SubsetEvaluation], baseline: SubsetEvaluation) -> None:
    print("\n=== Subset evaluation vs all-feature baseline ===")
    header = (
        f"{'subset':<22}{'method':<14}{'#feat':>6}{'in_dim':>7}{'sec':>7}{'ep':>4}"
        f"{'CKA':>7}{'procr':>7}{'trust':>7}{'jac30':>7}{'act%':>6}{'ARI':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in [baseline, *evaluations]:
        print(
            f"{row.label:<22}{row.method:<14}{row.n_features:>6}{row.input_dim:>7}"
            f"{row.train_seconds:>7.1f}{row.epochs_run:>4}{row.cka_similarity:>7.3f}"
            f"{row.procrustes_distance:>7.3f}{row.trustworthiness:>7.3f}{row.jaccard_knn30:>7.3f}"
            f"{row.active_pct:>6.2f}{row.ari:>7.3f}"
        )


def plot_results(
    evaluations: list[SubsetEvaluation],
    baseline: SubsetEvaluation,
    primary_method: str,
    output_path: Path,
) -> None:
    # Elbow curve = only the primary-method size sweep (labelled "{method}@{k}"),
    # not the head-to-head entry that happens to share the method name.
    primary = sorted(
        [e for e in evaluations if e.label.startswith(f"{primary_method}@")],
        key=lambda e: e.n_features,
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("VAE feature selection: fidelity & compute vs subset size", fontweight="bold")

    if primary:
        sizes = [e.n_features for e in primary]
        axes[0, 0].plot(sizes, [e.cka_similarity for e in primary], marker="o", label="CKA (↑)")
        axes[0, 0].plot(sizes, [e.trustworthiness for e in primary], marker="s", label="trustworthiness (↑)")
        axes[0, 0].plot(sizes, [e.jaccard_knn30 for e in primary], marker="^", label="jaccard knn30 (↑)")
        axes[0, 0].axhline(1.0, color="grey", ls="--", lw=1)
        axes[0, 0].set_title(f"Latent fidelity vs baseline ({primary_method})")
        axes[0, 0].set_xlabel("# features kept")
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].plot(sizes, [e.procrustes_distance for e in primary], marker="o", color="tab:red")
        axes[0, 1].set_title("Procrustes distance to baseline (↓ better)")
        axes[0, 1].set_xlabel("# features kept")
        axes[0, 1].grid(alpha=0.3)

        ax_time = axes[1, 0]
        ax_time.plot(sizes, [e.train_seconds for e in primary], marker="o", color="tab:blue", label="train sec")
        ax_time.axhline(baseline.train_seconds, color="tab:blue", ls="--", lw=1, label="baseline sec")
        ax_time.set_title("Training time vs subset size")
        ax_time.set_xlabel("# features kept")
        ax_time.set_ylabel("seconds")
        ax_time.legend(fontsize=8)
        ax_time.grid(alpha=0.3)

    # Method comparison at the shared eval size: CKA per method (plus random).
    method_rows = [e for e in evaluations if e.label.startswith("method:") or e.method == "random"]
    if method_rows:
        names = [e.method for e in method_rows]
        ckas = [e.cka_similarity for e in method_rows]
        colors = ["tab:gray" if e.method == "random" else "tab:green" for e in method_rows]
        axes[1, 1].barh(names, ckas, color=colors)
        axes[1, 1].set_title("CKA by selection method (same #features)")
        axes[1, 1].set_xlabel("CKA similarity to baseline")
        axes[1, 1].grid(alpha=0.3, axis="x")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved plot -> {output_path}")


def recommend(evaluations: list[SubsetEvaluation], min_cka: float, min_trust: float) -> SubsetEvaluation | None:
    passing = [
        e for e in evaluations
        if e.cka_similarity >= min_cka and e.trustworthiness >= min_trust
    ]
    if not passing:
        return None
    return min(passing, key=lambda e: (e.n_features, -e.cka_similarity))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test feature-selection methods for the TabularVAE")
    parser.add_argument("--parquet-path", required=True, help="Source feature-map parquet")
    parser.add_argument("--feature-spec-path", default=DEFAULT_FEATURE_SPEC_PATH)
    parser.add_argument("--output-dir", default="feature_selection_results")
    parser.add_argument("--row-filter", default=DEFAULT_TRAINING_ROW_FILTER)

    parser.add_argument("--sample-rows", type=int, default=DEFAULT_SAMPLE_ROWS,
                        help="Training rows (kept small for fast iteration). "
                             "Set to 0 (or use --all-rows) to train on every filtered row.")
    parser.add_argument("--all-rows", action="store_true",
                        help="Use ALL filtered rows for training (ignores --sample-rows). "
                             "Reads the full parquet/glob with no reservoir subsampling.")
    parser.add_argument("--reference-rows", type=int, default=DEFAULT_REFERENCE_ROWS,
                        help="Fixed rows encoded by every model for the fidelity metrics.")
    parser.add_argument("--reference-parquet-path", default=None,
                        help="Optional separate parquet for the reference set (else drawn from --parquet-path).")

    # VAE hyperparameters (match the rest of the pipeline).
    parser.add_argument("--epochs", type=int, default=40, help="Epoch ceiling; early stopping decides the real stop.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dims", default="256,128")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--kl-weight", type=float, default=0.05)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)

    # Early stopping.
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--active-unit-threshold", type=float, default=DEFAULT_ACTIVE_UNIT_THRESHOLD)

    # Selection.
    parser.add_argument("--methods", default="permutation,pca,ica,nmf,feature_clustering",
                        help="Comma-separated ranking methods to compare head-to-head. "
                             f"Available: {', '.join(ALL_METHODS)}.")
    parser.add_argument("--mutual-info", action="store_true", help="Also compute the (slower) MI-to-latent ranking.")
    parser.add_argument("--primary-method", default=PRIMARY_METHOD,
                        help="Method swept across --keep-sizes for the elbow curve.")
    # Structural-method (PCA / ICA / NMF / feature clustering) knobs.
    parser.add_argument("--pca-variance", type=float, default=0.9,
                        help="Cumulative explained variance retained for the PCA-loading ranking.")
    parser.add_argument("--n-components", type=int, default=12,
                        help="Number of components for the ICA and NMF rankings (clamped to #features).")
    parser.add_argument("--n-clusters", type=int, default=0,
                        help="Feature-clustering groups (0 => use --eval-keep).")
    parser.add_argument("--structural-rows", type=int, default=20000,
                        help="Rows subsampled from the training set for the structural rankings.")
    parser.add_argument("--keep-sizes", default=DEFAULT_KEEP_SIZES,
                        help="Comma-separated subset sizes for the primary-method elbow sweep.")
    parser.add_argument("--eval-keep", type=int, default=16,
                        help="Shared #features at which every method is compared head-to-head.")
    parser.add_argument("--force-keep", default="",
                        help="Comma-separated features to always keep (e.g. biologically essential REF,ALT,X_PREV1,X_NEXT1).")

    # Downstream clustering agreement (optional).
    parser.add_argument("--clustering", default="none", choices=["none", "hdbscan", "umap-hdbscan"])
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=50)
    parser.add_argument("--hdbscan-min-samples", type=int, default=10)
    parser.add_argument("--umap-n-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.05)

    # Recommendation thresholds.
    parser.add_argument("--min-cka", type=float, default=0.9)
    parser.add_argument("--min-trust", type=float, default=0.9)
    return parser.parse_args()


def read_all_rows(
    conn,
    parquet_path: str,
    feature_names: list[str],
    where: str | None,
) -> pl.DataFrame:
    """Full read of every filtered row (no reservoir subsampling). Globs across a multi-file union."""
    select_list = ", ".join(quote_ident(name) for name in feature_names)
    sql = f"SELECT {select_list} FROM read_parquet(?)"
    if where:
        sql += f" WHERE {where}"
    return conn.execute(sql, [str(parquet_path)]).pl()


def load_frames(args: argparse.Namespace, feature_names: list[str]) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, int]]:
    with connect_duckdb(threads=args.threads) as conn:
        eligible = get_row_count(conn, args.parquet_path, where=args.row_filter)
        non_null_counts = get_non_null_counts(conn, args.parquet_path, feature_names, where=args.row_filter)
    if eligible <= 0:
        raise RuntimeError(f"No rows matched the filter: {args.row_filter}")
    use_all_rows = args.all_rows or args.sample_rows <= 0
    sample_rows = eligible if use_all_rows else min(args.sample_rows, eligible)
    how = "ALL filtered rows" if use_all_rows else f"reservoir sample of {sample_rows:,}"
    log(f"Eligible rows={eligible:,}; training on {how}; reference={args.reference_rows:,}")

    with connect_duckdb(threads=1) as conn:  # threads=1 -> deterministic REPEATABLE sampling
        if use_all_rows:
            train_frame = read_all_rows(conn, args.parquet_path, feature_names, where=args.row_filter)
        else:
            train_frame = sample_frame(
                conn=conn,
                parquet_path=args.parquet_path,
                feature_names=feature_names,
                sample_rows=sample_rows,
                seed=args.seed,
                where=args.row_filter,
            )
    reference_source = args.reference_parquet_path or args.parquet_path
    with connect_duckdb(threads=1) as conn:
        reference_frame = sample_frame(
            conn=conn,
            parquet_path=reference_source,
            feature_names=feature_names,
            sample_rows=args.reference_rows,
            seed=args.seed + 1234,  # different draw so the reference set is not the training rows
            where=args.row_filter,
        )
    return train_frame, reference_frame, non_null_counts


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = load_feature_specs(args.feature_spec_path)
    feature_names = [spec.name for spec in specs]
    hidden_dims = [int(d) for d in args.hidden_dims.split(",") if d.strip()]
    force_keep = [name.strip() for name in args.force_keep.split(",") if name.strip()]
    early_stopping = EarlyStoppingConfig(
        patience=args.patience, min_delta=args.min_delta, active_unit_threshold=args.active_unit_threshold
    )
    train_kwargs = dict(
        hidden_dims=hidden_dims,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        kl_weight=args.kl_weight,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    cluster_kwargs = dict(
        min_cluster_size=args.hdbscan_min_cluster_size,
        min_samples=args.hdbscan_min_samples,
        umap_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
    )

    train_frame, reference_frame, non_null_counts = load_frames(args, feature_names)

    # --- 1. Baseline VAE on ALL features -----------------------------------
    log("Training baseline VAE on all features")
    baseline = train_vae_in_memory(
        train_frame, specs, non_null_counts, early_stopping=early_stopping, **train_kwargs
    )
    baseline_latent = encode_frame(baseline, reference_frame)
    log(
        f"Baseline: {len(baseline.feature_names)} feats, input_dim={baseline.input_dim}, "
        f"{baseline.epochs_run} epochs, {baseline.train_seconds:.1f}s, "
        f"active_units={baseline.final_active_units}/{baseline.latent_dim}"
    )
    baseline_labels = None
    if args.clustering != "none":
        baseline_labels = cluster_latent(baseline_latent, args.clustering, seed=args.seed, **cluster_kwargs)

    # --- 2. Feature-importance rankings ------------------------------------
    requested_methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if args.mutual_info and "mutual_info" not in requested_methods:
        requested_methods.append("mutual_info")
    unknown = [m for m in requested_methods if m not in ALL_METHODS]
    if unknown:
        raise SystemExit(f"Unknown --methods: {unknown}. Available: {', '.join(ALL_METHODS)}")
    needed = set(requested_methods) | {args.primary_method}
    n_clusters = args.n_clusters or args.eval_keep

    log(f"Computing feature-importance rankings for: {', '.join(sorted(needed))}")
    # Entropy is cheap and is a dependency of redundancy + feature clustering.
    entropy_scores = rank_by_variance(train_frame, specs)
    rankings: dict[str, dict[str, float]] = {}
    if "variance" in needed:
        rankings["variance"] = entropy_scores
    if "redundancy" in needed:
        rankings["redundancy"] = rank_by_redundancy(train_frame, specs, entropy_scores)
    if "encoder_weights" in needed:
        rankings["encoder_weights"] = rank_by_encoder_weights(baseline)
    if "permutation" in needed:
        log("  permutation importance re-encodes the reference set once per feature")
        rankings["permutation"] = rank_by_latent_permutation(baseline, reference_frame, baseline_latent, args.seed)
    if "mutual_info" in needed:
        rankings["mutual_info"] = rank_by_mutual_info(baseline, reference_frame, baseline_latent, args.seed)
    structural = STRUCTURAL_METHODS & needed
    if structural:
        rankings.update(
            compute_structural_rankings(
                train_frame, specs, entropy_scores, structural,
                n_components=args.n_components, pca_variance=args.pca_variance,
                n_clusters=n_clusters, seed=args.seed, max_rows=args.structural_rows,
            )
        )

    requested_methods = [m for m in requested_methods if m in rankings]
    print_ranking_table(rankings, baseline.feature_names)

    # --- 3. Build candidate subsets ----------------------------------------
    keep_sizes = sorted({int(s) for s in args.keep_sizes.split(",") if s.strip()})
    total = len(baseline.feature_names)
    keep_sizes = [k for k in keep_sizes if 0 < k < total]

    def with_forced(names: list[str], k: int) -> list[str]:
        merged = list(dict.fromkeys(force_keep + names))
        return merged[:max(k, len(force_keep))]

    primary_scores = rankings[args.primary_method]  # guaranteed present (added to `needed`)

    # (a) primary method across sizes -> elbow curve
    subset_specs: list[tuple[str, str, list[str]]] = []
    for k in keep_sizes:
        subset_specs.append((f"{args.primary_method}@{k}", args.primary_method, with_forced(top_k(primary_scores, k), k)))
    # (b) every requested method at the shared eval size -> head-to-head
    for method in requested_methods:
        names = with_forced(top_k(rankings[method], args.eval_keep), args.eval_keep)
        subset_specs.append((f"method:{method}@{args.eval_keep}", method, names))
    # (c) random subset of the same size -> sanity baseline
    rng = np.random.default_rng(args.seed)
    shuffled = [str(name) for name in rng.permutation(baseline.feature_names)]
    random_names = with_forced(shuffled[: args.eval_keep], args.eval_keep)
    subset_specs.append((f"random@{args.eval_keep}", "random", random_names))

    # --- 4. Evaluate subsets (cache identical feature sets) ----------------
    evaluations: list[SubsetEvaluation] = []
    cache: dict[frozenset[str], SubsetEvaluation] = {}
    for label, method, names in subset_specs:
        key = frozenset(names)
        if key in cache:
            cached = cache[key]
            log(f"{label}: reusing identical subset already evaluated as '{cached.label}'")
            evaluations.append(
                SubsetEvaluation(**{**asdict(cached), "label": label, "method": method})
            )
            continue
        log(f"Evaluating {label} ({len(names)} features)")
        result = evaluate_subset(
            label=label,
            method=method,
            feature_names=names,
            all_specs=specs,
            train_frame=train_frame,
            reference_frame=reference_frame,
            baseline_latent=baseline_latent,
            baseline_labels=baseline_labels,
            non_null_counts=non_null_counts,
            train_kwargs=train_kwargs,
            early_stopping=early_stopping,
            clustering=args.clustering,
            cluster_kwargs=cluster_kwargs,
            seed=args.seed,
        )
        cache[key] = result
        evaluations.append(result)
        log(
            f"  CKA={result.cka_similarity:.3f} procrustes={result.procrustes_distance:.3f} "
            f"trust={result.trustworthiness:.3f} time={result.train_seconds:.1f}s "
            f"speedup={baseline.train_seconds / max(result.train_seconds, 1e-6):.2f}x"
        )

    baseline_row = SubsetEvaluation(
        label="baseline(all)", method="all", n_features=len(baseline.feature_names),
        input_dim=baseline.input_dim, train_seconds=baseline.train_seconds, epochs_run=baseline.epochs_run,
        active_pct=baseline.active_pct, val_total_loss=baseline.final_val_loss,
        procrustes_distance=0.0, cka_similarity=1.0, jaccard_knn10=1.0, jaccard_knn30=1.0,
        trustworthiness=1.0, continuity=1.0, ari=1.0, nmi=1.0, features=sorted(baseline.feature_names),
    )

    # --- 5. Report ---------------------------------------------------------
    print_evaluation_table(evaluations, baseline_row)
    plot_results(evaluations, baseline_row, args.primary_method, output_dir / "feature_selection.png")

    rec = recommend(
        [e for e in evaluations if e.method != "random"], args.min_cka, args.min_trust
    )
    if rec is not None:
        speedup = baseline.train_seconds / max(rec.train_seconds, 1e-6)
        print(
            f"\nRECOMMENDATION: '{rec.label}' keeps {rec.n_features}/{total} features "
            f"(input_dim {rec.input_dim} vs {baseline.input_dim}) at CKA={rec.cka_similarity:.3f}, "
            f"trustworthiness={rec.trustworthiness:.3f}, ~{speedup:.2f}x faster training."
        )
        print(f"  keep: {', '.join(rec.features)}")
    else:
        print(
            f"\nNo subset met CKA>={args.min_cka} and trustworthiness>={args.min_trust}. "
            f"Loosen thresholds, raise --eval-keep/--keep-sizes, or increase --sample-rows."
        )

    payload = {
        "args": vars(args),
        "baseline": asdict(baseline_row),
        "rankings": rankings,
        "evaluations": [asdict(e) for e in evaluations],
        "recommendation": asdict(rec) if rec is not None else None,
    }
    (output_dir / "feature_selection_results.json").write_text(json.dumps(payload, indent=2, default=str))
    log(f"Wrote {output_dir / 'feature_selection_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
