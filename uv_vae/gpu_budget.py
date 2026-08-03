"""Hard cap on how much GPU memory one uv_vae process may hold.

The node has a single RTX PRO 5000 Blackwell (48 GB) shared with other work, so no
run is allowed to expand into the whole card. ``apply()`` installs a ceiling
(default 16 GB, ``UV_VAE_GPU_MEM_GB``) on every allocator that can reach the GPU:

* **torch** -- ``set_per_process_memory_fraction`` caps the caching allocator, so
  torch raises a normal OOM at the ceiling instead of taking the device.
* **RMM** (used by cuDF and cuML) -- reinitialised with a bounded pool. Without
  this, cuDF's default pool grows toward all free memory and the torch ceiling
  buys nothing.

The two allocators **share one budget**: RMM is capped first, and torch is given
whatever is left. They must not each be handed the full ceiling -- that was a bug
in the first version of this module, where a 16 GB budget permitted 16 GB of torch
plus an 8 GB RMM pool, i.e. 24 GB. ``budget_gb`` is the total for the process, and
``BudgetReport.torch_budget_gb + BudgetReport.rmm_pool_gb`` must equal it.

Both caps are per *process*. When several runs share the card, divide the budget
between them (the tmux runners do this from ``concurrency``).

``max_batch_rows`` answers the other half of the question: given a ceiling, how
many rows may a training batch hold? Batch size is a swept variable in
``Batch_Size_Learning_Rate_Testing``, so this module never silently rewrites it --
see ``UV_VAE_GPU_OOM_POLICY`` on :func:`resolve_batch_size`.

Nothing here changes training math. Import is safe without CUDA, torch, or RMM.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

BUDGET_ENV = "UV_VAE_GPU_MEM_GB"
POLICY_ENV = "UV_VAE_GPU_OOM_POLICY"
RMM_SHARE_ENV = "UV_VAE_RMM_SHARE"
DEFAULT_BUDGET_GB = 16.0

# Slice of the budget reserved for RMM (cuDF / cuML); torch gets the rest.
#
# 0.25 suits the TRAINING path, which is what the sweeps run: cuDF only holds one
# decoded record batch (read_chunk_rows, default 100k) at a time, while torch holds
# the model, a full batch of activations and the optimizer state -- so torch is by
# far the heavier consumer. The clustering pipeline inverts this (cuML UMAP/HDBSCAN
# over millions of rows against torch doing single-batch inference), so it should
# pass a larger share explicitly, or set UV_VAE_RMM_SHARE.
DEFAULT_RMM_SHARE = 0.25

# Fraction of the budget a training batch may occupy. The rest absorbs
# fragmentation, cuBLAS workspaces, the CUDA context (~0.5 GB) and optimizer state.
DEFAULT_BATCH_HEADROOM = 0.60

# Activations are counted once for the forward pass; autograd keeps most of them
# alive for the backward pass and the optimizer needs transient scratch on top.
AUTOGRAD_OVERHEAD = 2.5

BYTES_PER_GB = 1024 ** 3

# Size of the RMM pool this process has already installed, if any. ``rmm.reinitialize``
# swaps the memory resource out from under whatever cuML has already allocated, so a
# pool is installed once and never replaced by a smaller one -- see :func:`_cap_rmm`.
_INSTALLED_POOL_BYTES: int | None = None


@dataclass
class BudgetReport:
    """What was actually enforced -- copied into each run's training_report.json."""

    enabled: bool
    budget_gb: float
    device_total_gb: float | None = None
    torch_fraction: float | None = None
    # torch_budget_gb + rmm_pool_gb == budget_gb. Recorded separately so a run's
    # training_report.json shows how the one budget was divided.
    torch_budget_gb: float | None = None
    rmm_pool_gb: float = 0.0
    # Why RMM ended up with the pool it did. A zero pool has three quite different
    # causes -- absent, share=0, or a failed reinitialise -- and they used to print
    # identically, which hid a mis-applied cap for a whole sweep.
    rmm_note: str | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "budget_gb": self.budget_gb,
            "device_total_gb": self.device_total_gb,
            "torch_fraction": self.torch_fraction,
            "torch_budget_gb": self.torch_budget_gb,
            "rmm_pool_gb": self.rmm_pool_gb,
            "rmm_note": self.rmm_note,
            "notes": list(self.notes),
        }


def resolve_budget_gb(budget_gb: float | None = None) -> float:
    """Explicit argument wins, then ``UV_VAE_GPU_MEM_GB``, then 16 GB."""
    if budget_gb is not None:
        return float(budget_gb)
    raw = os.environ.get(BUDGET_ENV, "").strip()
    if not raw:
        return DEFAULT_BUDGET_GB
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_BUDGET_GB
    return value if value > 0 else DEFAULT_BUDGET_GB


def resolve_rmm_share(rmm_share: float | None = None) -> float:
    """Slice of the budget RMM may take -- 0.0 when RMM is not even installed.

    Both ``apply`` and ``max_batch_rows`` call this so they agree on how much of
    the budget is left for torch. Without RMM present the answer is 0.0 and torch
    gets the whole budget.
    """
    try:
        import rmm  # noqa: F401
    except ImportError:
        return 0.0

    if rmm_share is None:
        raw = os.environ.get(RMM_SHARE_ENV, "").strip()
        try:
            rmm_share = float(raw) if raw else DEFAULT_RMM_SHARE
        except ValueError:
            rmm_share = DEFAULT_RMM_SHARE
    # Never starve torch entirely, and never go negative.
    return min(max(float(rmm_share), 0.0), 0.9)


def torch_budget_gb(budget_gb: float | None = None, rmm_share: float | None = None) -> float:
    """The part of the budget torch actually gets, after RMM's slice."""
    budget = resolve_budget_gb(budget_gb)
    return budget * (1.0 - resolve_rmm_share(rmm_share))


def apply(budget_gb: float | None = None, rmm_share: float | None = None,
          verbose: bool = True) -> BudgetReport:
    """Cap torch and RMM so their COMBINED footprint stays within ``budget_gb``.

    RMM is capped first and torch is given the remainder. Handing each allocator
    the full budget would permit 1.5-2x the intended footprint, which is the whole
    thing this function exists to prevent.

    Safe to call more than once and safe to call with no GPU -- it reports
    ``enabled=False`` and changes nothing.
    """
    budget = resolve_budget_gb(budget_gb)
    report = BudgetReport(enabled=False, budget_gb=budget)

    try:
        import torch
    except ImportError:
        report.notes.append("torch not importable; no cap applied")
        return _finish(report, verbose)

    if not torch.cuda.is_available():
        report.notes.append("CUDA not available; no cap applied")
        return _finish(report, verbose)

    total_bytes = torch.cuda.get_device_properties(0).total_memory
    report.device_total_gb = round(total_bytes / BYTES_PER_GB, 2)
    budget_bytes = budget * BYTES_PER_GB

    if budget_bytes >= total_bytes:
        report.notes.append(
            f"budget {budget:.1f} GB >= device {report.device_total_gb} GB; "
            f"capping at the device instead"
        )
        budget_bytes = float(total_bytes)

    # RMM first: torch's share is whatever RMM did not reserve, and RMM must be
    # bounded before cuDF/cuML allocate anything from it.
    rmm_bytes = _cap_rmm(budget_bytes, resolve_rmm_share(rmm_share), report)

    torch_bytes = budget_bytes - rmm_bytes
    fraction = torch_bytes / total_bytes
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    report.torch_fraction = round(fraction, 4)
    report.torch_budget_gb = round(torch_bytes / BYTES_PER_GB, 2)
    report.enabled = True

    return _finish(report, verbose)


def _cap_rmm(budget_bytes: float, share: float, report: BudgetReport) -> int:
    """Bound the RMM pool that cuDF/cuML allocate from; return the bytes reserved.

    Returns 0 when RMM is absent (the CPU-only and torch-only paths) or when the
    reinitialisation fails -- in both cases the caller gives torch the whole
    budget, which is correct because nothing else will be allocating.

    When a pool is already installed and this call asks for one no larger, the
    existing size is returned and nothing is reinitialised: the pool is installed
    once per process and only ever grows.
    """
    global _INSTALLED_POOL_BYTES

    if share <= 0.0:
        report.rmm_note = "rmm not importable or share=0; whole budget goes to torch"
        report.notes.append(report.rmm_note)
        return 0

    try:
        import rmm
    except ImportError:  # resolve_rmm_share already checked, but stay defensive
        report.rmm_note = "rmm not importable; cuDF/cuML pool not capped"
        report.notes.append(report.rmm_note)
        return 0

    pool_bytes = int(budget_bytes * share)
    pool_bytes -= pool_bytes % 256  # RMM requires 256-byte aligned pool sizes
    if pool_bytes < 256:
        report.rmm_note = f"share {share} of a {budget_bytes / BYTES_PER_GB:.2f} GB budget rounds to an empty pool"
        report.notes.append(report.rmm_note)
        return 0

    # BOTH sizes must be 256-aligned, not just the maximum. pool_bytes // 4 is a
    # multiple of 64 and only sometimes of 256; when it is not, reinitialize raises
    # "Initial pool size required to be a multiple of 256 bytes" and the cap is lost
    # entirely. That is the bug that let a sweep run uncapped: the 0.9 share it asked
    # for could not install (initial 3,865,470,528) while the 0.25 training default
    # that later overwrote it could (initial 1,073,741,824), so the only cap that ever
    # took hold was the wrong one.
    initial_bytes = pool_bytes // 4
    initial_bytes -= initial_bytes % 256

    # A pool already installed in this process is never replaced by a smaller one.
    # reinitialize swaps the memory resource underneath live cuML allocations, so a
    # late caller applying a different share -- run_variant_cluster_pipeline installs
    # the TRAINING default (0.25) at import time, and the sweep lazy-imports it for
    # SigProfiler mid-run -- would otherwise drop the ceiling below the pool already
    # in use and OOM every subsequent fit.
    if _INSTALLED_POOL_BYTES is not None and pool_bytes <= _INSTALLED_POOL_BYTES:
        report.rmm_pool_gb = round(_INSTALLED_POOL_BYTES / BYTES_PER_GB, 2)
        report.rmm_note = (
            f"kept the {report.rmm_pool_gb} GB pool already installed in this process "
            f"rather than shrinking it to {pool_bytes / BYTES_PER_GB:.2f} GB"
        )
        report.notes.append(report.rmm_note)
        return _INSTALLED_POOL_BYTES

    try:
        rmm.reinitialize(
            pool_allocator=True,
            initial_pool_size=initial_bytes,
            maximum_pool_size=pool_bytes,
        )
        _INSTALLED_POOL_BYTES = pool_bytes
        report.rmm_pool_gb = round(pool_bytes / BYTES_PER_GB, 2)
        # Let cuDF spill to host RAM rather than fail when the pool is exhausted.
        os.environ.setdefault("CUDF_SPILL", "1")
        return pool_bytes
    except Exception as exc:  # RMM raises a variety of driver-level errors
        report.rmm_note = f"rmm.reinitialize failed ({type(exc).__name__}: {exc})"
        report.notes.append(report.rmm_note)
        return 0


def _finish(report: BudgetReport, verbose: bool) -> BudgetReport:
    if verbose:
        import sys

        if report.enabled:
            message = (
                f"[gpu_budget] cap {report.budget_gb:.1f} GB of "
                f"{report.device_total_gb} GB = torch {report.torch_budget_gb} GB "
                f"(fraction {report.torch_fraction})"
            )
            if report.rmm_pool_gb:
                message += f" + rmm pool {report.rmm_pool_gb} GB"
            else:
                # Never print a bare "not installed" here: an uncapped RMM is the
                # difference between a sweep that runs and one that OOMs, and the
                # reason has to be visible in the log at the moment it is decided.
                message += f" + rmm 0 GB ({report.rmm_note or 'not installed'})"
        else:
            message = f"[gpu_budget] not enforced: {'; '.join(report.notes) or 'no GPU'}"
        print(message, file=sys.stderr, flush=True)
    return report


def bytes_per_row(
    n_categorical: int,
    n_numeric: int,
    embedding_dims: list[int] | dict[str, int],
    hidden_dims: list[int],
    latent_dim: int,
    categorical_cardinalities: list[int] | dict[str, int] | None = None,
    amp: bool = True,
) -> int:
    """Estimate GPU bytes one row costs during a training step.

    Counts the input tensors plus every activation the encoder/decoder produces,
    scaled by :data:`AUTOGRAD_OVERHEAD` for the values autograd keeps alive. This
    is a planning estimate, not an allocator trace -- it is deliberately
    pessimistic so ``max_batch_rows`` errs toward a batch that fits.
    """
    embeddings = list(embedding_dims.values()) if isinstance(embedding_dims, dict) else list(embedding_dims)
    if categorical_cardinalities is None:
        cardinalities: list[int] = []
    elif isinstance(categorical_cardinalities, dict):
        cardinalities = list(categorical_cardinalities.values())
    else:
        cardinalities = list(categorical_cardinalities)

    # Inputs live in fp32/int64 regardless of AMP -- autocast happens downstream.
    input_elements = 8 * n_categorical + 4 * n_numeric * 2  # cat int64, num + mask fp32

    concat_dim = sum(embeddings) + n_numeric
    encoder = concat_dim + sum(hidden_dims) + 2 * latent_dim
    decoder = latent_dim + sum(hidden_dims) + n_numeric + sum(cardinalities)
    activation_elements = encoder + decoder

    element_bytes = 2 if amp else 4
    activation_bytes = activation_elements * element_bytes * AUTOGRAD_OVERHEAD

    return int(input_elements + activation_bytes)


@dataclass
class ModelShape:
    """Input shape of the VAE, derived from a feature spec rather than retyped.

    The hardcoded shape this replaces was wrong in two ways: ``n_numeric=30``
    (ml_features.json has 29, and only 15 survive the all-null drop on the
    production featuremap) and uniform ``cardinalities=[16]*11`` /
    ``embedding_dims=[8]*11`` (real cardinalities are 5, 8 or 16, giving embedding
    dims of 6 or 8). Both errors inflate the per-row estimate, so the batch ceiling
    came out lower than the card can actually take.
    """

    n_categorical: int
    n_numeric: int
    embedding_dims: list[int]
    categorical_cardinalities: list[int]
    dropped_features: list[str] = field(default_factory=list)

    def bytes_per_row(self, hidden_dims: list[int], latent_dim: int, amp: bool = True) -> int:
        return bytes_per_row(
            n_categorical=self.n_categorical,
            n_numeric=self.n_numeric,
            embedding_dims=self.embedding_dims,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            categorical_cardinalities=self.categorical_cardinalities,
            amp=amp,
        )


def shape_from_specs(
    feature_spec_path,
    non_null_counts: dict[str, int] | None = None,
) -> ModelShape:
    """Read the model's input shape out of ``ml_features.json``.

    ``non_null_counts`` applies the same drop :func:`uv_vae.data.split_specs`
    performs at training time, so the estimate matches the model that actually
    gets built. Pass ``None`` for the pre-drop shape, which is the pessimistic
    (safe) answer when the parquet has not been scanned.

    Imported lazily so this module keeps its promise of importing without torch,
    CUDA, polars or duckdb present.
    """
    from uv_vae.data import split_specs
    from uv_vae.features import load_feature_specs
    from uv_vae.preprocess import infer_embedding_dim

    specs = load_feature_specs(feature_spec_path)
    if non_null_counts is None:
        non_null_counts = {spec.name: 1 for spec in specs}
    categorical_specs, numeric_specs, dropped = split_specs(specs, non_null_counts)

    cardinalities = [spec.cardinality for spec in categorical_specs]
    return ModelShape(
        n_categorical=len(categorical_specs),
        n_numeric=len(numeric_specs),
        embedding_dims=[infer_embedding_dim(value) for value in cardinalities],
        categorical_cardinalities=cardinalities,
        dropped_features=dropped,
    )


def shape_from_parquet(
    feature_spec_path,
    parquet_path,
    where: str | None = None,
    threads: int | None = None,
) -> ModelShape:
    """:func:`shape_from_specs` with the all-null columns actually measured.

    Falls back to the pre-drop shape if the scan fails for any reason -- a
    planning estimate must never be the thing that stops a run from starting.
    """
    from uv_vae.data import connect_duckdb, get_non_null_counts
    from uv_vae.features import load_feature_specs

    try:
        specs = load_feature_specs(feature_spec_path)
        conn = connect_duckdb(threads=threads)
        try:
            counts = get_non_null_counts(
                conn, parquet_path, [spec.name for spec in specs], where=where
            )
        finally:
            conn.close()
        return shape_from_specs(feature_spec_path, counts)
    except Exception:
        return shape_from_specs(feature_spec_path)


def max_batch_rows(
    per_row_bytes: int,
    budget_gb: float | None = None,
    headroom: float = DEFAULT_BATCH_HEADROOM,
    rmm_share: float | None = None,
) -> int:
    """Largest batch that fits inside ``headroom`` of TORCH's share of the budget.

    Measured against ``torch_budget_gb``, not the whole budget: a batch is a torch
    allocation, and RMM's slice is not available to it. When RMM is not installed
    the two are the same number.
    """
    usable = torch_budget_gb(budget_gb, rmm_share) * BYTES_PER_GB * headroom
    return max(1, int(usable // max(1, per_row_bytes)))


def resolve_batch_size(
    requested: int,
    per_row_bytes: int,
    budget_gb: float | None = None,
    headroom: float = DEFAULT_BATCH_HEADROOM,
) -> tuple[int, dict]:
    """Check ``requested`` against the budget and apply ``UV_VAE_GPU_OOM_POLICY``.

    Returns ``(batch_size, report)``. Policies:

    ``warn`` (default)
        Keep the requested batch and log the projection. The torch cap from
        :func:`apply` still turns an overrun into a clean OOM rather than a
        device takeover. This is the default *because batch size is a swept
        variable* -- quietly shrinking it would corrupt the batch-size sweep.
    ``clamp``
        Reduce to the largest batch that fits. Use for production runs where
        finishing matters more than the exact batch size.
    ``error``
        Refuse to start. Use to fail fast before burning hours on a config that
        cannot fit.
    """
    policy = os.environ.get(POLICY_ENV, "warn").strip().lower() or "warn"
    ceiling = max_batch_rows(per_row_bytes, budget_gb=budget_gb, headroom=headroom)
    projected_gb = requested * per_row_bytes / BYTES_PER_GB

    report = {
        "requested_batch_size": int(requested),
        "per_row_bytes": int(per_row_bytes),
        "projected_batch_gb": round(projected_gb, 2),
        "max_batch_rows": int(ceiling),
        "budget_gb": resolve_budget_gb(budget_gb),
        "policy": policy,
        "action": "kept",
    }

    if requested <= ceiling:
        return int(requested), report

    if policy == "clamp":
        report["action"] = "clamped"
        report["effective_batch_size"] = int(ceiling)
        return int(ceiling), report

    if policy == "error":
        report["action"] = "error"
        raise RuntimeError(
            f"batch_size={requested:,} needs about {projected_gb:.1f} GB of GPU memory but the "
            f"budget is {report['budget_gb']:.1f} GB (max {ceiling:,} rows). Lower --batch-size, "
            f"raise {BUDGET_ENV}, or set {POLICY_ENV}=clamp."
        )

    report["action"] = "warned"
    return int(requested), report


def describe_environment() -> dict:
    """Device/driver facts worth recording alongside a run's results."""
    info: dict = {"cuda_available": False}
    try:
        import torch
    except ImportError:
        info["note"] = "torch not importable"
        return info

    info["torch_version"] = torch.__version__
    info["cuda_available"] = bool(torch.cuda.is_available())
    if not info["cuda_available"]:
        return info

    properties = torch.cuda.get_device_properties(0)
    info["device_name"] = properties.name
    info["device_total_gb"] = round(properties.total_memory / BYTES_PER_GB, 2)
    info["capability"] = f"sm_{properties.major}{properties.minor}"
    info["cuda_runtime"] = torch.version.cuda
    try:
        info["arch_list"] = torch.cuda.get_arch_list()
    except Exception:
        pass
    return info
