"""Run the real trainer twice on a real parquet and prove the results are identical.

``tests/test_reproducibility.py`` asserts the same property on a 512-row synthetic
fixture, which is fast enough for CI but exercises none of the things that
actually break determinism at scale: multi-threaded DuckDB scans, the streaming
train/val split, AMP loss scaling, and real CUDA kernels. This script is the
full-size version -- run it on the GPU node before trusting any sweep result.

    python uv_vae/scripts/check_reproducibility.py \
        --parquet-path /path/to/combined.featuremap.parquet \
        --output-root /tmp/repro --streaming --sample-rows 2000000 --epochs 2

Exit code 0 means the two runs are bit-identical. Non-zero means they are not,
and the report names the first tensor that diverged and by how much.

Interpreting a failure on CUDA, most likely cause first:

  * ``CUBLAS_WORKSPACE_CONFIG`` was not exported BEFORE python started. cuBLAS
    reads it when it creates its handle, so ``training.seed_everything()``
    setting it is too late. Export it in the shell:  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  * TF32 left enabled (``seed_everything`` disables it; a later library may re-enable it).
  * AMP: the streaming trainer forces mixed precision on CUDA. Loss scaling is
    deterministic for a fixed run, but a divergence here points at it -- compare
    with ``--no-streaming`` (the in-RAM trainer does not use AMP).
  * A non-deterministic kernel that ``torch.use_deterministic_algorithms(True)``
    did not catch.

Note this only compares runs to EACH OTHER on one machine. It does not claim CPU
and GPU results match -- they do not, and AMP alone guarantees that.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from uv_vae import gpu_budget
from uv_vae.training import DEFAULT_TRAINING_ROW_FILTER, TrainingConfig, train


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify two identical runs produce identical results")
    parser.add_argument("--parquet-path", required=True)
    parser.add_argument("--feature-spec-path", default=str(REPO_ROOT / "ml_features.json"))
    parser.add_argument("--output-root", default="repro_check",
                        help="scratch dir for the two runs; removed afterwards unless --keep")
    parser.add_argument("--row-filter", default=DEFAULT_TRAINING_ROW_FILTER)
    parser.add_argument("--streaming", action="store_true",
                        help="use the streaming trainer (what the sweeps use). Otherwise the "
                             "in-RAM sampled trainer, which is faster to check and has no AMP.")
    parser.add_argument("--sample-rows", type=int, default=500_000,
                        help="in-RAM mode only; streaming always uses all filtered rows")
    parser.add_argument("--epochs", type=int, default=2,
                        help="2 is enough -- divergence shows up in the first optimizer step")
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dims", default="256,128")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--kl-weight", type=float, default=0.05)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--keep", action="store_true", help="keep the two run dirs")
    parser.add_argument("--json-out", default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace, output_dir: Path) -> TrainingConfig:
    return TrainingConfig(
        parquet_path=args.parquet_path,
        feature_spec_path=args.feature_spec_path,
        output_dir=str(output_dir),
        row_filter=args.row_filter,
        sample_rows=0 if args.streaming else args.sample_rows,
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        hidden_dims=[int(p) for p in args.hidden_dims.split(",") if p.strip()],
        learning_rate=args.learning_rate,
        kl_weight=args.kl_weight,
        train_fraction=args.train_fraction,
        seed=args.seed,
        threads=args.threads,
    )


def run_once(args: argparse.Namespace, output_dir: Path, label: str) -> Path:
    log(f"--- run {label} ---")
    started = perf_counter()
    config = build_config(args, output_dir)
    if args.streaming:
        from uv_vae.early_stopping import EarlyStoppingConfig
        from uv_vae.streaming import train_with_early_stopping_streaming

        # patience=0 disables early stopping: a stop triggered at a different
        # epoch would be a real difference, but it makes the diff harder to read.
        run_dir = train_with_early_stopping_streaming(config, EarlyStoppingConfig(patience=0))
    else:
        run_dir = train(config)
    log(f"    finished in {perf_counter() - started:.1f}s -> {run_dir}")
    return run_dir


def compare_weights(left_dir: Path, right_dir: Path) -> tuple[bool, list[dict]]:
    left = torch.load(left_dir / "model.pt", map_location="cpu", weights_only=False)
    right = torch.load(right_dir / "model.pt", map_location="cpu", weights_only=False)
    left_state, right_state = left["model_state_dict"], right["model_state_dict"]

    if left_state.keys() != right_state.keys():
        return False, [{"tensor": "<keys>", "note": "the two runs built different models"}]

    differences: list[dict] = []
    for name, left_tensor in left_state.items():
        right_tensor = right_state[name]
        if torch.equal(left_tensor, right_tensor):
            continue
        delta = (left_tensor.float() - right_tensor.float()).abs()
        differences.append({
            "tensor": name,
            "shape": list(left_tensor.shape),
            "max_abs_delta": float(delta.max()),
            "mean_abs_delta": float(delta.mean()),
            "elements_differing": int((delta > 0).sum()),
            "elements_total": int(delta.numel()),
        })
    return not differences, differences


def compare_history(left_dir: Path, right_dir: Path) -> tuple[bool, list[dict]]:
    left = json.loads((left_dir / "training_report.json").read_text())["history"]
    right = json.loads((right_dir / "training_report.json").read_text())["history"]
    if left == right:
        return True, []

    mismatches: list[dict] = []
    for index, (a, b) in enumerate(zip(left, right)):
        for key in sorted(set(a) | set(b)):
            if a.get(key) != b.get(key):
                mismatches.append({"epoch_index": index, "metric": key,
                                   "run_a": a.get(key), "run_b": b.get(key)})
    if len(left) != len(right):
        mismatches.append({"note": f"different epoch counts: {len(left)} vs {len(right)}"})
    return False, mismatches


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    environment = gpu_budget.describe_environment()
    import os

    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")

    print("=== reproducibility check ===")
    print(f"  trainer      : {'streaming (AMP on CUDA)' if args.streaming else 'in-RAM sampled'}")
    print(f"  parquet      : {args.parquet_path}")
    print(f"  epochs       : {args.epochs}   batch: {args.batch_size}   seed: {args.seed}")
    print(f"  device       : {environment.get('device_name', 'cpu')} "
          f"({environment.get('capability', 'n/a')})")
    print(f"  torch        : {environment.get('torch_version')} / cuda {environment.get('cuda_runtime')}")
    print(f"  CUBLAS_WORKSPACE_CONFIG : {workspace!r}")
    if environment.get("cuda_available") and workspace not in {":4096:8", ":16:8"}:
        print("  WARNING: CUBLAS_WORKSPACE_CONFIG is not set in the environment. cuBLAS reads")
        print("           it at handle creation, so a run started without it may be")
        print("           non-deterministic no matter what seed_everything() does.")
    print()

    first = run_once(args, output_root / "run_a", "A")
    second = run_once(args, output_root / "run_b", "B")

    weights_ok, weight_differences = compare_weights(first, second)
    history_ok, history_differences = compare_history(first, second)
    passed = weights_ok and history_ok

    print()
    print("=== result ===")
    if weights_ok:
        print("  weights : IDENTICAL (bitwise)")
    else:
        print(f"  weights : DIFFER in {len(weight_differences)} tensor(s)")
        for difference in weight_differences[:10]:
            if "note" in difference:
                print(f"      {difference['tensor']}: {difference['note']}")
            else:
                print(f"      {difference['tensor']} {difference['shape']}: "
                      f"max|d|={difference['max_abs_delta']:.3e} "
                      f"({difference['elements_differing']}/{difference['elements_total']} elements)")
        if len(weight_differences) > 10:
            print(f"      ... and {len(weight_differences) - 10} more")

    if history_ok:
        print("  history : IDENTICAL")
    else:
        print(f"  history : DIFFERS in {len(history_differences)} metric(s)")
        for difference in history_differences[:10]:
            print(f"      {difference}")

    print()
    print("  VERDICT : " + ("REPRODUCIBLE" if passed else "NOT REPRODUCIBLE"))
    if not passed:
        print()
        print("  Most likely cause on CUDA, in order:")
        print("    1. CUBLAS_WORKSPACE_CONFIG not exported before python started")
        print("    2. TF32 re-enabled by a library imported after seed_everything()")
        print("    3. AMP loss scaling (streaming only) -- retry without --streaming")
        print("    4. A non-deterministic kernel use_deterministic_algorithms(True) missed")

    if args.json_out:
        payload = {
            "passed": passed,
            "streaming": args.streaming,
            "environment": environment,
            "cublas_workspace_config": workspace,
            "config": vars(args),
            "weight_differences": weight_differences,
            "history_differences": history_differences,
        }
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        print(f"  report  : {path}")

    if not args.keep:
        shutil.rmtree(output_root, ignore_errors=True)
    else:
        print(f"  runs kept: {first}  {second}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
