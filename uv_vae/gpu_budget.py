"""Hard cap on how much GPU memory one uv_vae process may hold.

The node has a single RTX PRO 5000 Blackwell (48 GB) shared with other work, so no
run is allowed to expand into the whole card. ``apply()`` installs a ceiling
(default 16 GB, ``UV_VAE_GPU_MEM_GB``) on every allocator that can reach the GPU:

* **torch** -- ``set_per_process_memory_fraction`` caps the caching allocator, so
  torch raises a normal OOM at the ceiling instead of taking the device.
* **RMM** (used by cuDF and cuML) -- reinitialised with a bounded pool. Without
  this, cuDF's default pool grows toward all free memory and the torch ceiling
  buys nothing.

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
DEFAULT_BUDGET_GB = 16.0

# Fraction of the budget a training batch may occupy. The rest absorbs
# fragmentation, cuBLAS workspaces, the CUDA context (~0.5 GB) and optimizer state.
DEFAULT_BATCH_HEADROOM = 0.60

# Activations are counted once for the forward pass; autograd keeps most of them
# alive for the backward pass and the optimizer needs transient scratch on top.
AUTOGRAD_OVERHEAD = 2.5

BYTES_PER_GB = 1024 ** 3


@dataclass
class BudgetReport:
    """What was actually enforced -- copied into each run's training_report.json."""

    enabled: bool
    budget_gb: float
    device_total_gb: float | None = None
    torch_fraction: float | None = None
    rmm_pool_gb: float | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "budget_gb": self.budget_gb,
            "device_total_gb": self.device_total_gb,
            "torch_fraction": self.torch_fraction,
            "rmm_pool_gb": self.rmm_pool_gb,
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


def apply(budget_gb: float | None = None, verbose: bool = True) -> BudgetReport:
    """Cap torch (and RMM, if present) at ``budget_gb`` on the current device.

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

    fraction = budget_bytes / total_bytes
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    report.torch_fraction = round(fraction, 4)
    report.enabled = True

    _cap_rmm(budget_bytes, report)
    return _finish(report, verbose)


def _cap_rmm(budget_bytes: float, report: BudgetReport) -> None:
    """Bound the RMM pool that cuDF/cuML allocate from.

    Skipped silently when RMM is absent (the CPU-only and torch-only paths). The
    pool is deliberately smaller than the torch ceiling: both allocators draw on
    the same device, so they have to share one budget rather than each claim it.
    """
    try:
        import rmm
    except ImportError:
        report.notes.append("rmm not installed; cuDF/cuML pool not capped")
        return

    # Half the budget to RMM, half left for torch. cuDF only holds one decoded
    # record batch at a time, so it is the lighter of the two consumers.
    pool_bytes = int(budget_bytes * 0.5)
    # RMM requires pool sizes aligned to 256 bytes.
    pool_bytes -= pool_bytes % 256
    try:
        rmm.reinitialize(
            pool_allocator=True,
            initial_pool_size=pool_bytes // 4,
            maximum_pool_size=pool_bytes,
        )
        report.rmm_pool_gb = round(pool_bytes / BYTES_PER_GB, 2)
        # Let cuDF spill to host RAM rather than fail when the pool is exhausted.
        os.environ.setdefault("CUDF_SPILL", "1")
    except Exception as exc:  # RMM raises a variety of driver-level errors
        report.notes.append(f"rmm.reinitialize failed ({type(exc).__name__}: {exc})")


def _finish(report: BudgetReport, verbose: bool) -> BudgetReport:
    if verbose:
        import sys

        if report.enabled:
            message = (
                f"[gpu_budget] cap {report.budget_gb:.1f} GB of "
                f"{report.device_total_gb} GB (torch fraction {report.torch_fraction})"
            )
            if report.rmm_pool_gb is not None:
                message += f", rmm pool {report.rmm_pool_gb} GB"
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


def max_batch_rows(
    per_row_bytes: int,
    budget_gb: float | None = None,
    headroom: float = DEFAULT_BATCH_HEADROOM,
) -> int:
    """Largest batch that fits inside ``headroom`` of the budget."""
    budget = resolve_budget_gb(budget_gb)
    usable = budget * BYTES_PER_GB * headroom
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
