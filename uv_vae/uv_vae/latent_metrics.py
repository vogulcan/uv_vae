"""Latent-space quality metrics for post-training evaluation.

Computes metrics beyond active units and val loss that measure how well the
latent space encodes the data and how suitable it is for downstream UMAP +
HDBSCAN clustering.  Every metric is unsupervised — no row-level labels needed.

Metrics:

1. **Activation quality** — are all latent dims doing useful work?
   - Active units (Burda et al. 2016, "Importance Weighted Autoencoders",
     ICLR 2016)
   - Mean KL per dim (Burgess et al. 2018, "Understanding disentangling in
     β-VAE", arXiv 1804.03599) — average information usage across dims
   - Collapsed dim percentage: fraction of dims with KL < threshold
     (Lucas et al. 2019, "Don't Blame the ELBO!", arXiv 1911.02469)
   - Participation Ratio / effective dimensionality (Gao et al. 2017,
     "A Theory of Multineuron Dimensionality, Dynamics and Measurement";
     Litwin-Kumar et al. 2017, "Optimal degrees of synaptic connectivity";
     Mazzaglia et al. 2022, "The Free Energy Principle for Perception and
     Action") — continuous version of active units: PR = (Σλ)²/Σλ², where
     λ are eigenvalues of the posterior-mean covariance matrix
   - Off-diagonal covariance mean/max of posterior mean (Kumar et al. 2017,
     "Variational Inference of Disentangled Latent Concepts", DIP-VAE)

2. **Structure preservation** — does the latent space preserve input geometry?
   - Trustworthiness (Venna & Kaski 2006, "Local multidimensional scaling",
     Neural Networks)

3. **Reconstruction quality**
   - Numeric MSE (masked, per-feature and aggregate)

This module is additive — it does not modify any original uv_vae files.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.manifold import trustworthiness
from torch.utils.data import DataLoader

from uv_vae.model import TabularVAE

# Re-exported (not redefined) so the post-hoc report here and the per-epoch diagnostics
# in the trainers always score collapse against the same threshold.
from uv_vae.early_stopping import DEFAULT_KL_COLLAPSE_THRESHOLD  # noqa: F401


# -------------------------------------------------------------------------
# 1. Activation quality (single streaming pass)
# -------------------------------------------------------------------------

@dataclass
class ActivationMetrics:
    active_units: int
    latent_dim: int
    active_unit_threshold: float
    effective_dimensionality: float
    posterior_mean_variance: list[float]
    kl_per_dim: list[float]
    kl_total: float
    mean_kl: float
    kl_collapse_threshold: float
    collapsed_dims: int
    collapsed_pct: float
    off_diagonal_cov_mean: float
    off_diagonal_cov_max: float
    rows_evaluated: int

    def to_dict(self) -> dict:
        return {
            "active_units": self.active_units,
            "latent_dim": self.latent_dim,
            "active_unit_threshold": self.active_unit_threshold,
            "effective_dimensionality": round(self.effective_dimensionality, 4),
            "posterior_mean_variance": [round(v, 8) for v in self.posterior_mean_variance],
            "kl_per_dim": [round(v, 6) for v in self.kl_per_dim],
            "kl_total": round(self.kl_total, 6),
            "mean_kl": round(self.mean_kl, 6),
            "kl_collapse_threshold": self.kl_collapse_threshold,
            "collapsed_dims": self.collapsed_dims,
            "collapsed_pct": round(self.collapsed_pct, 2),
            "off_diagonal_cov_mean": round(self.off_diagonal_cov_mean, 8),
            "off_diagonal_cov_max": round(self.off_diagonal_cov_max, 8),
            "rows_evaluated": self.rows_evaluated,
        }


def compute_activation_metrics(
    model: TabularVAE,
    loader: DataLoader,
    device: torch.device,
    active_unit_threshold: float = 0.01,
    kl_collapse_threshold: float = DEFAULT_KL_COLLAPSE_THRESHOLD,
) -> ActivationMetrics:
    """Compute activation quality metrics in a single streaming pass.

    Accumulates mu and logvar statistics in float64 for numerical stability,
    then derives: active units, mean KL, collapsed_pct, participation ratio,
    and off-diagonal covariance magnitude.
    """
    was_training = model.training
    model.eval()
    latent_dim = model.config.latent_dim

    count = 0
    mu_sum = torch.zeros(latent_dim, dtype=torch.float64, device=device)
    mu_sq_sum = torch.zeros(latent_dim, dtype=torch.float64, device=device)
    kl_sum = torch.zeros(latent_dim, dtype=torch.float64, device=device)
    mu_outer_sum = torch.zeros(latent_dim, latent_dim, dtype=torch.float64, device=device)

    with torch.inference_mode():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                cat_inputs = batch[0].to(device, non_blocking=True)
                num_inputs = batch[1].to(device, non_blocking=True)
            else:
                raise TypeError(f"Unexpected batch type: {type(batch)}")

            mu, logvar = model.encode(cat_inputs, num_inputs)
            mu64 = mu.double()
            logvar64 = logvar.double()

            mu_sum += mu64.sum(dim=0)
            mu_sq_sum += (mu64 * mu64).sum(dim=0)
            mu_outer_sum += mu64.T @ mu64
            kl_sum += (-0.5 * (1.0 + logvar64 - mu64.pow(2) - logvar64.exp())).sum(dim=0)
            count += int(mu64.shape[0])

    model.train(mode=was_training)

    if count == 0:
        raise RuntimeError("No batches were available for latent metrics")

    mu_mean = mu_sum / count
    mu_variance = torch.clamp((mu_sq_sum / count) - mu_mean.pow(2), min=0.0)
    kl_per_dim = kl_sum / count

    active_mask = mu_variance > active_unit_threshold
    active_units = int(active_mask.sum().item())

    cov_matrix = (mu_outer_sum / count) - mu_mean.unsqueeze(1) * mu_mean.unsqueeze(0)
    cov_np = cov_matrix.cpu().numpy()

    # Eigendecomposition used only for participation ratio — not returned directly.
    eigenvalues = np.linalg.eigvalsh(cov_np)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0.0)
    eig_sum = eigenvalues.sum()
    eig_sq_sum = (eigenvalues ** 2).sum()
    effective_dim = float((eig_sum ** 2) / eig_sq_sum) if eig_sq_sum > 0 else 0.0

    diag_mask = np.eye(latent_dim, dtype=bool)
    off_diag = np.abs(cov_np[~diag_mask])
    off_diag_mean = float(off_diag.mean()) if off_diag.size > 0 else 0.0
    off_diag_max = float(off_diag.max()) if off_diag.size > 0 else 0.0

    kl_np = kl_per_dim.cpu().numpy()
    kl_total = float(kl_np.sum())
    mean_kl = kl_total / latent_dim if latent_dim > 0 else 0.0
    collapsed_dims = int((kl_np < kl_collapse_threshold).sum())
    collapsed_pct = collapsed_dims / latent_dim * 100.0 if latent_dim > 0 else 0.0

    return ActivationMetrics(
        active_units=active_units,
        latent_dim=latent_dim,
        active_unit_threshold=active_unit_threshold,
        effective_dimensionality=effective_dim,
        posterior_mean_variance=[float(v) for v in mu_variance.cpu().tolist()],
        kl_per_dim=[float(v) for v in kl_np.tolist()],
        kl_total=kl_total,
        mean_kl=mean_kl,
        kl_collapse_threshold=kl_collapse_threshold,
        collapsed_dims=collapsed_dims,
        collapsed_pct=collapsed_pct,
        off_diagonal_cov_mean=off_diag_mean,
        off_diagonal_cov_max=off_diag_max,
        rows_evaluated=count,
    )


# -------------------------------------------------------------------------
# 3. Structure preservation (input space vs latent space)
# -------------------------------------------------------------------------

@dataclass
class StructureMetrics:
    trustworthiness_k10: float
    trustworthiness_k50: float
    rows_evaluated: int

    def to_dict(self) -> dict:
        return {
            "trustworthiness_k10": round(self.trustworthiness_k10, 6),
            "trustworthiness_k50": round(self.trustworthiness_k50, 6),
            "rows_evaluated": self.rows_evaluated,
        }


def compute_structure_metrics(
    input_features: np.ndarray,
    latent_embeddings: np.ndarray,
) -> StructureMetrics:
    """Compare input-space and latent-space geometry via trustworthiness.

    Uses a subsample — caller is responsible for ensuring both arrays are the
    same rows and small enough for pairwise distance computation (~10K rows max).
    """
    n = input_features.shape[0]
    if n != latent_embeddings.shape[0]:
        raise ValueError("input_features and latent_embeddings must have the same row count")

    max_k = max(1, n // 2 - 1)
    k10 = min(10, max_k)
    k50 = min(50, max_k)

    trust_k10 = float(trustworthiness(input_features, latent_embeddings, n_neighbors=k10))
    trust_k50 = float(trustworthiness(input_features, latent_embeddings, n_neighbors=k50))

    return StructureMetrics(
        trustworthiness_k10=trust_k10,
        trustworthiness_k50=trust_k50,
        rows_evaluated=n,
    )


# -------------------------------------------------------------------------
# 4. Reconstruction quality
# -------------------------------------------------------------------------

@dataclass
class ReconstructionMetrics:
    numeric_mse_total: float
    numeric_mse_per_feature: dict[str, float]
    rows_evaluated: int

    def to_dict(self) -> dict:
        return {
            "numeric_mse_total": round(self.numeric_mse_total, 8),
            "numeric_mse_per_feature": {k: round(v, 8) for k, v in self.numeric_mse_per_feature.items()},
            "rows_evaluated": self.rows_evaluated,
        }


def compute_reconstruction_metrics(
    model: TabularVAE,
    loader: DataLoader,
    device: torch.device,
    numeric_feature_names: list[str],
    use_amp: bool = False,
) -> ReconstructionMetrics:
    """Per-feature numeric reconstruction quality (masked MSE).

    Uses the posterior mean (mu) for reconstruction, not a sampled z — this
    gives the deterministic reconstruction quality without stochastic noise.
    """
    was_training = model.training
    model.eval()

    num_features = len(numeric_feature_names)
    num_se_sum = np.zeros(num_features, dtype=np.float64) if num_features > 0 else np.zeros(0)
    num_observed = np.zeros(num_features, dtype=np.float64) if num_features > 0 else np.zeros(0)
    total_rows = 0

    with torch.inference_mode():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                cat_inputs = batch[0].to(device, non_blocking=True)
                num_inputs = batch[1].to(device, non_blocking=True)
                num_mask = batch[2].to(device, non_blocking=True) if len(batch) > 2 else torch.ones_like(num_inputs)
            else:
                raise TypeError(f"Unexpected batch type: {type(batch)}")

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                mu, _ = model.encode(cat_inputs, num_inputs)
                numeric_out, _ = model.decode(mu)

            total_rows += int(cat_inputs.shape[0])

            if num_features > 0:
                se = ((numeric_out - num_inputs) ** 2) * num_mask
                num_se_sum += se.sum(dim=0).cpu().numpy().astype(np.float64)
                num_observed += num_mask.sum(dim=0).cpu().numpy().astype(np.float64)

    model.train(mode=was_training)

    num_mse_per = {}
    for i, name in enumerate(numeric_feature_names):
        obs = num_observed[i]
        num_mse_per[name] = float(num_se_sum[i] / max(obs, 1.0))
    num_mse_total = float(num_se_sum.sum() / max(num_observed.sum(), 1.0))

    return ReconstructionMetrics(
        numeric_mse_total=num_mse_total,
        numeric_mse_per_feature=num_mse_per,
        rows_evaluated=total_rows,
    )


# -------------------------------------------------------------------------
# Combined evaluation
# -------------------------------------------------------------------------

def evaluate_checkpoint(
    checkpoint_path: str,
    parquet_path: str,
    feature_spec_path: str,
    row_filter: str | None = None,
    sample_rows: int = 10000,
    batch_size: int = 4096,
    seed: int = 42,
    threads: int | None = None,
    active_unit_threshold: float = 0.01,
) -> dict:
    """Load a checkpoint and compute the full latent-space quality report.

    Samples `sample_rows` from the parquet, encodes through the model, and
    computes activation, structure, and reconstruction metrics.

    For structure metrics (trustworthiness), the input features are the
    normalised numeric + embedded categorical vectors — the same
    representation the encoder sees.
    """
    from uv_vae.data import connect_duckdb, get_row_count, sample_frame
    from uv_vae.features import load_feature_specs
    from uv_vae.model import VAEConfig
    from uv_vae.preprocess import encode_categorical_column, encode_numeric_column
    from uv_vae.training import seed_everything

    seed_everything(seed)

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = TabularVAE(VAEConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    feature_report = payload["feature_report"]
    preprocess_report = payload["preprocess_report"]
    all_specs = load_feature_specs(feature_spec_path)
    spec_map = {s.name: s for s in all_specs}
    cat_specs = [spec_map[n] for n in feature_report["active_categorical_features"]]
    num_specs = [spec_map[n] for n in feature_report["active_numeric_features"]]
    numeric_means = preprocess_report["numeric_means"]
    numeric_stds = preprocess_report["numeric_stds"]

    feature_names = [s.name for s in cat_specs] + [s.name for s in num_specs]

    with connect_duckdb(threads=1) as conn:
        total = get_row_count(conn, parquet_path, where=row_filter)
        n = min(sample_rows, total)
        df = sample_frame(
            conn=conn,
            parquet_path=parquet_path,
            feature_names=feature_names,
            sample_rows=n,
            seed=seed,
            where=row_filter,
        )

    print(f"  eval sample: {df.height:,} rows from {parquet_path}", file=sys.stderr, flush=True)

    n_rows = df.height
    if cat_specs:
        cat_matrix = np.stack([encode_categorical_column(df, s) for s in cat_specs], axis=1)
    else:
        cat_matrix = np.zeros((n_rows, 0), dtype=np.int64)
    cat_tensor = torch.from_numpy(cat_matrix).long()

    means_arr = np.array([numeric_means[s.name] for s in num_specs], dtype=np.float32)
    stds_arr = np.array([numeric_stds[s.name] for s in num_specs], dtype=np.float32)
    if num_specs:
        num_arrays = []
        mask_arrays = []
        for spec in num_specs:
            values, mask = encode_numeric_column(df, spec)
            num_arrays.append(values)
            mask_arrays.append(mask)
        num_matrix = np.stack(num_arrays, axis=1)
        mask_matrix = np.stack(mask_arrays, axis=1)
        num_matrix = np.where(mask_matrix > 0, num_matrix, means_arr)
        num_matrix = (num_matrix - means_arr) / stds_arr
    else:
        num_matrix = np.zeros((n_rows, 0), dtype=np.float32)
        mask_matrix = np.zeros((n_rows, 0), dtype=np.float32)
    num_tensor = torch.from_numpy(num_matrix).float()
    mask_tensor = torch.from_numpy(mask_matrix).float()

    from torch.utils.data import DataLoader as DL, TensorDataset
    ds = TensorDataset(cat_tensor, num_tensor, mask_tensor)
    loader = DL(ds, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")

    activation = compute_activation_metrics(model, loader, device, active_unit_threshold)

    recon = compute_reconstruction_metrics(
        model, loader, device,
        numeric_feature_names=[s.name for s in num_specs],
        use_amp=device.type == "cuda",
    )

    input_parts = []
    if cat_specs:
        with torch.inference_mode():
            embedded_cats = []
            for i, name in enumerate(model.categorical_names):
                embedded_cats.append(
                    model.embeddings[name](cat_tensor[:, i].to(device)).cpu().numpy()
                )
            input_parts.append(np.concatenate(embedded_cats, axis=1))
    if num_specs:
        input_parts.append(num_matrix.astype(np.float32))

    input_features = np.concatenate(input_parts, axis=1) if input_parts else np.zeros((n_rows, 1))

    with torch.inference_mode():
        mu_chunks = []
        for start in range(0, n_rows, batch_size):
            end = min(start + batch_size, n_rows)
            mu, _ = model.encode(
                cat_tensor[start:end].to(device),
                num_tensor[start:end].to(device),
            )
            mu_chunks.append(mu.cpu().numpy())
    latent_embeddings = np.concatenate(mu_chunks, axis=0)

    structure = compute_structure_metrics(input_features, latent_embeddings)

    return {
        "checkpoint_path": checkpoint_path,
        "parquet_path": parquet_path,
        "sample_rows": n_rows,
        "activation": activation.to_dict(),
        "reconstruction": recon.to_dict(),
        "structure": structure.to_dict(),
    }
