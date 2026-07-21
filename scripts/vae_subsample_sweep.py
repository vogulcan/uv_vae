from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from scipy.spatial import procrustes
from sklearn.manifold import trustworthiness

from uv_vae.features import load_feature_specs
from uv_vae.inference import LatentInference
from uv_vae.preprocess import transform_frame
from uv_vae.training import DEFAULT_TRAINING_ROW_FILTER, DEFAULT_TRAINING_SAMPLE_ROWS, TrainingConfig, train


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def prompt_continue_after_training(
    run_dir: Path,
    fraction: float,
    sample_rows: int,
    final_epoch: dict,
) -> bool:
    """Print a training summary and ask whether to continue the sweep."""
    print(f"\n{'=' * 60}")
    print(f"  Training complete")
    print(f"  Sample fraction : {fraction * 100:.0f}%  ({sample_rows:,} rows)")
    print(f"  Model saved to  : {run_dir / 'model.pt'}")
    print(f"  Val total loss  : {final_epoch.get('val_total_loss', 'n/a'):.4f}")
    print(f"  Val numeric loss: {final_epoch.get('val_numeric_loss', 'n/a'):.4f}")
    print(f"  Val KL loss     : {final_epoch.get('val_kl_loss', 'n/a'):.4f}")
    print(f"{'=' * 60}")
    while True:
        try:
            resp = input("Continue sweep to next sample size? [y/n]: ").strip().lower()
        except EOFError:
            log("Non-interactive mode detected — continuing sweep automatically")
            return True
        if resp in ("y", "yes", ""):
            return True
        if resp in ("n", "no"):
            log("Sweep stopped by user")
            return False
        print("Please enter y or n")


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear Centered Kernel Alignment between two (N, D) embedding matrices.
    1 = identical geometry, 0 = no similarity.
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    numerator = np.linalg.norm(X.T @ Y, "fro") ** 2
    denom_x = np.linalg.norm(X.T @ X, "fro")
    denom_y = np.linalg.norm(Y.T @ Y, "fro")
    if denom_x < 1e-10 or denom_y < 1e-10:
        return 0.0
    return float(numerator / (denom_x * denom_y))


def latent_collapse_score(embeddings: np.ndarray) -> float:
    """
    Mean std across latent dimensions.
    Lower score = more collapsed latent dims.
    """
    return float(np.std(embeddings, axis=0).mean())


def compute_trustworthiness(input_matrix: np.ndarray, embeddings: np.ndarray, n_neighbors: int = 10) -> float:
    """Fraction of k-NN in latent space that were also neighbours in input space."""
    return float(trustworthiness(input_matrix, embeddings, n_neighbors=n_neighbors))


def get_input_matrix(inference: LatentInference, test_df: pl.DataFrame) -> np.ndarray:
    """Return the normalised numeric input matrix for the test set."""
    _, num_tensor = transform_frame(
        frame=test_df,
        categorical_specs=inference.categorical_specs,
        numeric_specs=inference.numeric_specs,
        numeric_means=inference.numeric_means,
        numeric_stds=inference.numeric_stds,
    )
    return num_tensor.numpy()


def load_val_metrics(run_dir: Path) -> dict:
    report_path = run_dir / "training_report.json"
    if not report_path.exists():
        return {}
    report = json.loads(report_path.read_text())
    history = report.get("history", [])
    return history[-1] if history else {}


def load_test_set(test_set_path: Path) -> pl.DataFrame:
    log(f"Loading test set from {test_set_path}")
    return pl.read_parquet(test_set_path)


def encode_test_set(checkpoint_path: Path, test_df: pl.DataFrame, batch_size: int) -> np.ndarray:
    inference = LatentInference.from_checkpoint(str(checkpoint_path), device="auto")
    features_df = test_df.select(inference.feature_names)
    return inference.encode_frame(features_df, batch_size=batch_size)


def compute_all_metrics(
    reference_emb: np.ndarray,
    subset_emb: np.ndarray,
    input_matrix: np.ndarray,
    val_metrics: dict,
    n_neighbors: int = 10,
) -> dict:
    _, _, procrustes_disparity = procrustes(reference_emb, subset_emb)
    return {
        "procrustes_disparity": float(procrustes_disparity),
        "linear_cka": linear_cka(reference_emb, subset_emb),
        "latent_collapse": latent_collapse_score(subset_emb),
        "trustworthiness": compute_trustworthiness(input_matrix, subset_emb, n_neighbors=n_neighbors),
        "val_total_loss": val_metrics.get("val_total_loss", float("nan")),
        "val_numeric_loss": val_metrics.get("val_numeric_loss", float("nan")),
        "val_kl_loss": val_metrics.get("val_kl_loss", float("nan")),
    }


def load_checkpoint_state(output_dir: Path) -> tuple[list[dict], set[float]]:
    """Load existing sweep results and determine which fractions are already done."""
    results_path = output_dir / "sweep_results.json"
    if not results_path.exists():
        return [], set()
    results = json.loads(results_path.read_text())
    completed = set()
    for r in results:
        checkpoint = Path(r["run_dir"]) / "model.pt"
        if checkpoint.exists():
            completed.add(r["fraction"])
    return results, completed


def find_reference_embedding(
    results: list[dict],
    test_df: pl.DataFrame,
    batch_size: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Re-encode the test set through the reference (largest fraction) VAE from a prior run."""
    if not results:
        return None, None
    ref = results[0]
    checkpoint_path = Path(ref["run_dir"]) / "model.pt"
    if not checkpoint_path.exists():
        return None, None
    log(f"Resuming — re-encoding test set through reference VAE ({ref['fraction']:.0%})")
    inference = LatentInference.from_checkpoint(str(checkpoint_path), device="auto")
    features_df = test_df.select(inference.feature_names)
    ref_emb = inference.encode_frame(features_df, batch_size=batch_size)
    input_matrix = get_input_matrix(inference, test_df)
    return ref_emb, input_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VAE subsample stability sweep")
    parser.add_argument("--parquet-path", required=True, help="Path to the source feature map parquet")
    parser.add_argument(
        "--output-dir",
        default="sweep_results",
        help="Directory for all sweep outputs (checkpoints, test set, plots)",
    )
    parser.add_argument("--test-set-path", required=True, help="Path to the fixed test set parquet")
    parser.add_argument(
        "--subsample-fractions",
        default="1.0,0.75,0.5,0.25,0.1,0.05",
        help="Comma-separated training data fractions, largest first",
    )
    parser.add_argument(
        "--row-filter",
        default=DEFAULT_TRAINING_ROW_FILTER,
        help="DuckDB SQL filter applied to the parquet before sampling",
    )
    parser.add_argument("--max-sample-rows", type=int, default=DEFAULT_TRAINING_SAMPLE_ROWS)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dims", default="256,128")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--kl-weight", type=float, default=0.05)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42,
                        help="Training seed (weight init, train/val split, batch order)")
    parser.add_argument("--data-seed", type=int, default=None,
                        help="Separate seed for DuckDB row sampling only. "
                             "When set, --seed controls all training RNG while this "
                             "controls which rows are selected. Omit to use --seed for both.")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--feature-spec-path", default="ml_features.json")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip the continue prompt after each training run (useful for batch HPC jobs)",
    )
    return parser.parse_args()


def run_sweep(args: argparse.Namespace) -> list[dict]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fractions = [float(f) for f in args.subsample_fractions.split(",") if f.strip()]
    fractions = sorted(set(fractions), reverse=True)

    test_df = load_test_set(Path(args.test_set_path))

    results, completed_fractions = load_checkpoint_state(output_dir)
    if completed_fractions:
        log(f"Resuming sweep — {len(completed_fractions)} fractions already done: "
            f"{', '.join(f'{f:.0%}' for f in sorted(completed_fractions, reverse=True))}")

    hidden_dims = [int(d) for d in args.hidden_dims.split(",") if d.strip()]
    reference_emb: np.ndarray | None = None
    input_matrix: np.ndarray | None = None

    if results:
        reference_emb, input_matrix = find_reference_embedding(
            results, test_df, batch_size=args.batch_size,
        )

    for fraction in fractions:
        if fraction in completed_fractions:
            log(f"Skipping fraction={fraction:.0%} — already completed")
            continue

        sample_rows = int(args.max_sample_rows * fraction)
        label = f"{fraction * 100:.0f}pct"
        log(f"Training VAE: fraction={fraction:.0%}, sample_rows={sample_rows:,}")

        config = TrainingConfig(
            parquet_path=args.parquet_path,
            feature_spec_path=args.feature_spec_path,
            output_dir=str(output_dir / f"run_{label}"),
            row_filter=args.row_filter,
            sample_rows=sample_rows,
            epochs=args.epochs,
            batch_size=args.batch_size,
            latent_dim=args.latent_dim,
            hidden_dims=hidden_dims,
            learning_rate=args.learning_rate,
            kl_weight=args.kl_weight,
            train_fraction=args.train_fraction,
            seed=args.seed,
            threads=args.threads,
            data_seed=args.data_seed,
        )

        t0 = perf_counter()
        run_dir = train(config)
        log(f"Training finished in {perf_counter() - t0:.1f}s — checkpoint: {run_dir / 'model.pt'}")

        val_metrics = load_val_metrics(run_dir)
        checkpoint_path = run_dir / "model.pt"

        log("Encoding test set")
        emb = encode_test_set(checkpoint_path, test_df, batch_size=args.batch_size)

        if reference_emb is None:
            reference_emb = emb
            inference = LatentInference.from_checkpoint(str(checkpoint_path), device="auto")
            input_matrix = get_input_matrix(inference, test_df)
            log("Reference embeddings set (full sample VAE)")

        log("Computing metrics")
        metrics = compute_all_metrics(
            reference_emb=reference_emb,
            subset_emb=emb,
            input_matrix=input_matrix,
            val_metrics=val_metrics,
        )
        result = {
            "fraction": fraction,
            "sample_rows": sample_rows,
            "run_dir": str(run_dir),
            **metrics,
        }
        results.append(result)

        log(
            f"  procrustes={metrics['procrustes_disparity']:.4f}  "
            f"cka={metrics['linear_cka']:.4f}  "
            f"trustworthiness={metrics['trustworthiness']:.4f}  "
            f"collapse={metrics['latent_collapse']:.4f}"
        )

        results_path = output_dir / "sweep_results.json"
        results_path.write_text(json.dumps(results, indent=2))
        log(f"Checkpoint saved — {len(results)}/{len(fractions)} fractions complete")

        if not args.non_interactive and fraction != fractions[-1]:
            if not prompt_continue_after_training(run_dir, fraction, sample_rows, val_metrics):
                break

    return results


def visualize_results(results: list[dict], output_dir: Path) -> None:
    if not results:
        return

    fractions = [r["fraction"] * 100 for r in results]
    sample_rows = [r["sample_rows"] for r in results]
    x_labels = [f"{f:.0f}%\n({n:,})" for f, n in zip(fractions, sample_rows)]

    metrics = {
        "Procrustes Disparity\n(lower = more similar to full VAE)": [r["procrustes_disparity"] for r in results],
        "Linear CKA\n(higher = more similar to full VAE)": [r["linear_cka"] for r in results],
        "Trustworthiness\n(higher = better local structure)": [r["trustworthiness"] for r in results],
        "Latent Collapse Score\n(higher = more active latent dims)": [r["latent_collapse"] for r in results],
        "Val Total Loss\n(lower = better reconstruction)": [r["val_total_loss"] for r in results],
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("VAE Subsample Stability Sweep", fontsize=14, fontweight="bold")

    for ax in axes.flat[len(metrics):]:
        ax.set_visible(False)
    for ax, (title, values) in zip(axes.flat, metrics.items()):
        clean_vals = [v if not (isinstance(v, float) and (v != v)) else None for v in values]
        plot_x = [x for x, v in zip(range(len(fractions)), clean_vals) if v is not None]
        plot_y = [v for v in clean_vals if v is not None]
        plot_labels = [x_labels[i] for i in plot_x]

        ax.plot(plot_x, plot_y, marker="o", linewidth=2, markersize=8, color="#2563eb")
        ax.set_xticks(plot_x)
        ax.set_xticklabels(plot_labels, fontsize=8)
        ax.set_xlabel("Training data fraction (rows)", fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

    plt.tight_layout()
    plot_path = output_dir / "sweep_results.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Plot saved to {plot_path}")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()

    log(f"Starting VAE subsample sweep — output: {output_dir}")
    results = run_sweep(args)

    log("Generating visualisation")
    visualize_results(results, output_dir)

    log(f"Sweep complete. Results: {output_dir / 'sweep_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
