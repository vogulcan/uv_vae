"""Preflight the GPU box before committing hours to a sweep.

Checks the things that silently produce wrong or unreproducible results rather
than a crash:

1. torch sees the GPU, and its wheel was built for this card's architecture.
   An RTX PRO 5000 Blackwell is ``sm_120``; a cu12.6-or-older wheel has no
   ``sm_120`` kernels and falls back to slow JIT-from-PTX or fails outright.
2. ``CUBLAS_WORKSPACE_CONFIG`` is exported *before* python starts. It is read
   when the cuBLAS handle is created, so setting it from inside python (as
   ``training.seed_everything`` does) is too late and
   ``torch.use_deterministic_algorithms(True)`` then either raises or is a no-op.
3. Deterministic mode actually runs a matmul and an embedding backward.
4. The GPU budget cap installs, and a batch of the requested size fits.
5. cuDF / cuML / RMM presence, since they change which allocator owns the card.

Exit code is non-zero when any REQUIRED check fails, so it can gate a runner:

    python uv_vae/scripts/gpu_preflight.py --batch-size 131072 || exit 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from uv_vae import gpu_budget

# Architectures the Blackwell workstation cards report. sm_120 is RTX PRO / RTX 50xx.
BLACKWELL = {"sm_100", "sm_120"}


class Check:
    def __init__(self) -> None:
        self.results: list[dict] = []
        self.failed = False

    def add(self, name: str, ok: bool, detail: str, required: bool = True) -> None:
        self.results.append({"check": name, "ok": ok, "required": required, "detail": detail})
        if required and not ok:
            self.failed = True
        marker = "PASS" if ok else ("FAIL" if required else "WARN")
        print(f"  [{marker}] {name}: {detail}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight the GPU before a sweep")
    parser.add_argument("--batch-size", type=int, default=131072,
                        help="batch size you intend to train with")
    parser.add_argument("--budget-gb", type=float, default=None,
                        help=f"GPU ceiling; defaults to ${gpu_budget.BUDGET_ENV} or "
                             f"{gpu_budget.DEFAULT_BUDGET_GB:.0f}")
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dims", default="256,128")
    parser.add_argument("--n-categorical", type=int, default=11,
                        help="categorical features in ml_features.json")
    parser.add_argument("--n-numeric", type=int, default=30)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--allow-cpu", action="store_true",
                        help="report instead of failing when no GPU is present")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    check = Check()
    print("=== uv_vae GPU preflight ===", flush=True)

    environment = gpu_budget.describe_environment()
    print(f"  torch {environment.get('torch_version', '?')}, "
          f"cuda runtime {environment.get('cuda_runtime', '?')}", flush=True)

    # ---- 1. device visible -------------------------------------------------
    if not environment.get("cuda_available"):
        check.add("cuda_available", False,
                  "torch.cuda.is_available() is False -- check the driver, and that "
                  "CUDA_VISIBLE_DEVICES is not set to an empty string",
                  required=not args.allow_cpu)
        _emit(check, environment, args)
        return 1 if check.failed else 0

    check.add("cuda_available", True,
              f"{environment['device_name']} "
              f"({environment['device_total_gb']} GB, {environment['capability']})")

    # ---- 2. wheel built for this architecture ------------------------------
    capability = environment.get("capability", "")
    arch_list = environment.get("arch_list", []) or []
    if capability in BLACKWELL:
        supported = capability in arch_list
        check.add(
            "arch_support", supported,
            f"{capability} {'is' if supported else 'is NOT'} in the wheel's arch list "
            f"({', '.join(arch_list) or 'unknown'}). Blackwell needs a cu128 or newer "
            f"torch build; pyproject already points at the cu128 index.",
        )
    else:
        check.add("arch_support", capability in arch_list,
                  f"{capability} vs wheel arch list {', '.join(arch_list) or 'unknown'}",
                  required=False)

    # ---- 3. CUBLAS_WORKSPACE_CONFIG exported before python -----------------
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")
    check.add(
        "cublas_workspace", workspace in {":4096:8", ":16:8"},
        f"CUBLAS_WORKSPACE_CONFIG={workspace!r}. Must be exported by the shell "
        f"before python starts -- setting it inside seed_everything() is too late, "
        f"because cuBLAS reads it when the handle is created.",
    )

    # ---- 4. deterministic mode really works --------------------------------
    import torch

    try:
        from uv_vae.training import seed_everything

        seed_everything(42)
        left = torch.randn(512, 256, device="cuda")
        right = torch.randn(256, 128, device="cuda")
        (left @ right).sum().item()

        embedding = torch.nn.Embedding(64, 16).cuda()
        indices = torch.randint(0, 64, (1024,), device="cuda")
        embedding(indices).sum().backward()
        torch.cuda.synchronize()
        check.add("deterministic_algorithms", True,
                  "matmul + embedding backward ran under use_deterministic_algorithms(True)")
    except Exception as exc:
        check.add("deterministic_algorithms", False,
                  f"{type(exc).__name__}: {exc}")

    # ---- 5. budget cap + batch fit -----------------------------------------
    report = gpu_budget.apply(args.budget_gb, verbose=False)
    check.add("budget_cap", report.enabled,
              f"capped at {report.budget_gb:.1f} GB of {report.device_total_gb} GB "
              f"(torch fraction {report.torch_fraction}"
              + (f", rmm pool {report.rmm_pool_gb} GB" if report.rmm_pool_gb else "")
              + ")")

    hidden_dims = [int(part) for part in args.hidden_dims.split(",") if part.strip()]
    # ml_features.json categoricals are small; embedding dims land at 4-16.
    embedding_dims = [8] * args.n_categorical
    cardinalities = [16] * args.n_categorical
    per_row = gpu_budget.bytes_per_row(
        n_categorical=args.n_categorical,
        n_numeric=args.n_numeric,
        embedding_dims=embedding_dims,
        hidden_dims=hidden_dims,
        latent_dim=args.latent_dim,
        categorical_cardinalities=cardinalities,
        amp=True,
    )
    ceiling = gpu_budget.max_batch_rows(per_row, budget_gb=args.budget_gb)
    fits = args.batch_size <= ceiling
    projected = args.batch_size * per_row / gpu_budget.BYTES_PER_GB
    check.add(
        "batch_fits", fits,
        f"batch_size={args.batch_size:,} needs about {projected:.1f} GB; the budget "
        f"allows up to {ceiling:,} rows. "
        + ("" if fits else f"Lower --batch-size, raise ${gpu_budget.BUDGET_ENV}, or set "
                           f"${gpu_budget.POLICY_ENV}=clamp."),
        required=False,
    )

    # ---- 6. a real allocation of that size ---------------------------------
    try:
        probe_rows = min(args.batch_size, ceiling)
        probe = torch.empty((probe_rows, args.n_numeric), dtype=torch.float32, device="cuda")
        del probe
        torch.cuda.empty_cache()
        check.add("allocation_probe", True,
                  f"allocated and freed a {probe_rows:,}-row numeric block under the cap")
    except Exception as exc:
        check.add("allocation_probe", False, f"{type(exc).__name__}: {exc}")

    # ---- 7. optional GPU dataframe stack -----------------------------------
    for module_name, why in (
        ("cudf", "streaming.py uses it to encode chunks on GPU"),
        ("cuml", "only wanted for GPU UMAP/HDBSCAN; it must NOT load during training"),
        ("rmm", "needed to bound the cuDF/cuML pool"),
    ):
        try:
            __import__(module_name)
            check.add(f"optional_{module_name}", True, f"present -- {why}", required=False)
        except ImportError:
            check.add(f"optional_{module_name}", False, f"absent -- {why}", required=False)

    _emit(check, environment, args)
    print("\n=== preflight " + ("FAILED" if check.failed else "OK") + " ===", flush=True)
    return 1 if check.failed else 0


def _emit(check: Check, environment: dict, args: argparse.Namespace) -> None:
    if not args.json_out:
        return
    payload = {"environment": environment, "checks": check.results, "failed": check.failed}
    path = Path(args.json_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"  wrote {path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
