"""Compare VAE latent spaces and training dynamics across a KL-weight sweep.

Run after run_kl_sweep.sh completes:

    python scripts/compare_kl_sweep.py \
        --sweep-dir /cta/users/patrickgao765/uv_vae/kl_sweep_<jobid> \
        --parquet-path /cta/users/patrickgao765/parquet_files/wt0-12-ppm0050.featuremap.parquet \
        --feature-spec-path ~/uv_vae/ml_features.json

Each KL weight's training output lives in:
    <sweep-dir>/kl_<value>/training/   (model.pt, summary.json, diagnostics_report.json)
    <sweep-dir>/kl_<value>/clustering/ (optional: analysis.parquet, cluster_summary.json)

Latent comparison: a fixed reference sample (--reference-rows drawn with --seed) is encoded
by every checkpoint. Each model is compared to the --reference-kl model (default 0.05, your
current baseline) using the project's standard metrics: Procrustes, CKA, trustworthiness,
Jaccard-kNN. Cluster agreement (ARI/NMI) is computed when analysis.parquet files are present
and the row sets can be matched by CHROM/POS/REF/ALT.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from uv_vae.data import connect_duckdb, sample_frame
from uv_vae.evaluation import compare_clusterings, compare_latent_spaces
from uv_vae.features import load_feature_specs
from uv_vae.inference import LatentInference
from uv_vae.training import DEFAULT_TRAINING_ROW_FILTER


def log(msg: str) -> None:
    import datetime
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def discover_runs(sweep_dir: Path) -> list[tuple[float, Path]]:
    """Return (kl_weight, run_dir) pairs sorted by kl_weight."""
    runs: list[tuple[float, Path]] = []
    for subdir in sweep_dir.iterdir():
        if not subdir.is_dir() or not subdir.name.startswith("kl_"):
            continue
        try:
            kl = float(subdir.name[3:])
        except ValueError:
            continue
        training_dir = subdir / "training"
        checkpoint = training_dir / "model.pt"
        if not checkpoint.exists():
            log(f"  WARNING: no model.pt in {training_dir} — skipping")
            continue
        runs.append((kl, subdir))
    return sorted(runs, key=lambda t: t[0])


def load_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "training" / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_diagnostics(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "training" / "diagnostics_report.json"
    if not path.exists():
        return []
    report = json.loads(path.read_text())
    return report.get("per_epoch", [])


def load_training_history(run_dir: Path) -> list[dict[str, Any]]:
    """Per-epoch val_total_loss, train_total_loss from summary.json history."""
    summary = load_summary(run_dir)
    return (summary.get("early_stopping") or {}).get("history") or []


def load_cluster_labels(run_dir: Path) -> np.ndarray | None:
    """Load HDBSCAN cluster labels from analysis.parquet if it exists."""
    analysis = run_dir / "clustering" / "analysis.parquet"
    if not analysis.exists():
        return None
    try:
        df = pl.read_parquet(analysis, columns=["cluster"])
        return df["cluster"].to_numpy()
    except Exception:
        return None


def load_cluster_keys(run_dir: Path) -> pl.DataFrame | None:
    """Load CHROM/POS/REF/ALT + cluster from analysis.parquet for row-matching."""
    analysis = run_dir / "clustering" / "analysis.parquet"
    if not analysis.exists():
        return None
    try:
        return pl.read_parquet(analysis, columns=["CHROM", "POS", "REF", "ALT", "cluster"])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Reference sample
# ---------------------------------------------------------------------------

def load_reference_frame(
    parquet_path: str,
    feature_spec_path: str,
    row_filter: str,
    n_rows: int,
    seed: int,
    threads: int,
) -> pl.DataFrame:
    specs = load_feature_specs(feature_spec_path)
    feature_names = [s.name for s in specs]
    with connect_duckdb(threads=threads) as conn:
        frame = sample_frame(
            conn=conn,
            parquet_path=parquet_path,
            feature_names=feature_names,
            sample_rows=n_rows,
            seed=seed,
            where=row_filter,
        )
    log(f"Reference sample: {frame.height:,} rows × {frame.width} features")
    return frame


# ---------------------------------------------------------------------------
# Encoding + comparison
# ---------------------------------------------------------------------------

def encode_checkpoint(
    checkpoint_path: Path,
    reference_frame: pl.DataFrame,
    feature_spec_path: str,
    parquet_path: str,
    batch_size: int,
) -> np.ndarray:
    inf = LatentInference.from_checkpoint(
        checkpoint_path=checkpoint_path,
        feature_spec_path=feature_spec_path,
        parquet_path=parquet_path,
    )
    return inf.encode_frame(reference_frame, batch_size=batch_size)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class KLResult:
    kl_weight: float
    epochs_run: int
    stopped_early: bool
    final_active_units: int
    latent_dim: int
    best_val_total_loss: float
    final_kl_total: float        # sum of per-dim KL at last epoch
    train_seconds: float
    # latent comparison to reference model
    procrustes_distance: float = float("nan")
    cka_similarity: float = float("nan")
    trustworthiness: float = float("nan")
    continuity: float = float("nan")
    jaccard_knn10: float = float("nan")
    jaccard_knn30: float = float("nan")
    # cluster comparison to reference model (nan if no clustering)
    cluster_ari: float = float("nan")
    cluster_nmi: float = float("nan")


# ---------------------------------------------------------------------------
# Matched cluster comparison
# ---------------------------------------------------------------------------

def compare_cluster_assignments(
    ref_keys: pl.DataFrame,
    cmp_keys: pl.DataFrame,
) -> dict[str, float]:
    """Join on CHROM/POS/REF/ALT, return ARI/NMI on matched rows."""
    joined = ref_keys.join(cmp_keys, on=["CHROM", "POS", "REF", "ALT"], how="inner",
                           suffix="_cmp")
    if joined.height < 10:
        return {"ari": float("nan"), "nmi": float("nan")}
    ref_labels = joined["cluster"].to_numpy()
    cmp_labels = joined["cluster_cmp"].to_numpy()
    return compare_clusterings(ref_labels, cmp_labels)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_comparison_table(results: list[KLResult], reference_kl: float) -> None:
    print("\n" + "=" * 100)
    print(f"KL-WEIGHT SWEEP COMPARISON  (latent metrics vs kl_weight={reference_kl})")
    print("=" * 100)
    hdr = (
        f"{'kl_weight':>10}  {'epochs':>6}  {'early?':>6}  {'AU':>4}  "
        f"{'val_loss':>9}  {'kl_total':>9}  "
        f"{'procrustes':>10}  {'CKA':>7}  {'trust':>7}  {'jac30':>7}  "
        f"{'clust_ARI':>9}  {'time(s)':>8}"
    )
    print(hdr)
    print("-" * 100)
    for r in results:
        ref_marker = " *" if abs(r.kl_weight - reference_kl) < 1e-9 else "  "
        print(
            f"{r.kl_weight:>10g}{ref_marker}"
            f"  {r.epochs_run:>6d}"
            f"  {'yes' if r.stopped_early else 'no':>6}"
            f"  {r.final_active_units:>4d}/{r.latent_dim}"
            f"  {r.best_val_total_loss:>9.4f}"
            f"  {r.final_kl_total:>9.4f}"
            f"  {r.procrustes_distance:>10.4f}"
            f"  {r.cka_similarity:>7.4f}"
            f"  {r.trustworthiness:>7.4f}"
            f"  {r.jaccard_knn30:>7.4f}"
            f"  {r.cluster_ari:>9.4f}"
            f"  {r.train_seconds:>8.0f}"
        )
    print("=" * 100)
    print("  * = reference model (latent metrics are identity by definition)")
    print("  AU = active units / latent_dim   |   procrustes/CKA/trust = vs reference model")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_sweep(
    results: list[KLResult],
    diagnostics_by_kl: dict[float, list[dict[str, Any]]],
    history_by_kl: dict[float, list[dict[str, Any]]],
    reference_kl: float,
    output_path: Path,
) -> None:
    kl_values = [r.kl_weight for r in results]
    cmap = plt.cm.viridis
    colors = {kl: cmap(i / max(len(kl_values) - 1, 1)) for i, kl in enumerate(kl_values)}

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("KL-weight sweep — VAE training dynamics and latent fidelity", fontsize=12)

    # --- Panel 1: val_loss per epoch -----------------------------------------
    ax = axes[0, 0]
    for kl, history in history_by_kl.items():
        if not history:
            continue
        epochs = [h.get("epoch", i + 1) for i, h in enumerate(history)]
        losses = [h.get("val_total_loss", float("nan")) for h in history]
        ax.plot(epochs, losses, color=colors[kl], label=f"β={kl:g}", linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val total loss")
    ax.set_title("Validation loss curves")
    ax.legend(fontsize=7, ncol=2)

    # --- Panel 2: active units per epoch -------------------------------------
    ax = axes[0, 1]
    for kl, diags in diagnostics_by_kl.items():
        if not diags:
            continue
        epochs = [d.get("epoch", i + 1) for i, d in enumerate(diags)]
        au = [d.get("active_units", float("nan")) for d in diags]
        ax.plot(epochs, au, color=colors[kl], label=f"β={kl:g}", linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Active units")
    ax.set_title("Active units per epoch")
    ax.legend(fontsize=7, ncol=2)

    # --- Panel 3: KL total per epoch -----------------------------------------
    ax = axes[0, 2]
    for kl, diags in diagnostics_by_kl.items():
        if not diags:
            continue
        epochs = [d.get("epoch", i + 1) for i, d in enumerate(diags)]
        kl_total = [d.get("kl_total", float("nan")) for d in diags]
        ax.plot(epochs, kl_total, color=colors[kl], label=f"β={kl:g}", linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL total (nats)")
    ax.set_title("Total KL divergence per epoch")
    ax.legend(fontsize=7, ncol=2)

    # --- Panel 4: final active units vs kl_weight ----------------------------
    ax = axes[1, 0]
    au_vals = [r.final_active_units for r in results]
    latent_dim = results[0].latent_dim if results else 16
    bar_colors = [colors[r.kl_weight] for r in results]
    bars = ax.bar([str(kl) for kl in kl_values], au_vals, color=bar_colors)
    ax.axhline(latent_dim, color="red", linestyle="--", linewidth=0.8, label=f"latent_dim={latent_dim}")
    ax.set_xlabel("KL weight (β)")
    ax.set_ylabel("Active units")
    ax.set_title("Final active units (collapse detector)")
    ax.legend(fontsize=8)
    for bar, val in zip(bars, au_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(val), ha="center", va="bottom", fontsize=8)

    # --- Panel 5: procrustes + trustworthiness vs kl_weight ------------------
    ax = axes[1, 1]
    proc = [r.procrustes_distance for r in results]
    trust = [r.trustworthiness for r in results]
    ax2 = ax.twinx()
    l1, = ax.plot(kl_values, proc, "o-", color="steelblue", label="Procrustes ↓", linewidth=1.5)
    l2, = ax2.plot(kl_values, trust, "s--", color="darkorange", label="Trustworthiness ↑", linewidth=1.5)
    ref_idx = next((i for i, r in enumerate(results) if abs(r.kl_weight - reference_kl) < 1e-9), None)
    if ref_idx is not None:
        ax.axvline(kl_values[ref_idx], color="gray", linestyle=":", linewidth=0.8, label=f"ref β={reference_kl:g}")
    ax.set_xscale("log")
    ax.set_xlabel("KL weight β (log scale)")
    ax.set_ylabel("Procrustes distance", color="steelblue")
    ax2.set_ylabel("Trustworthiness", color="darkorange")
    ax.set_title(f"Latent fidelity vs β  (ref = β={reference_kl:g})")
    lines = [l1, l2]
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8)

    # --- Panel 6: ARI/NMI vs kl_weight (cluster agreement) ------------------
    ax = axes[1, 2]
    ari_vals = [r.cluster_ari for r in results]
    nmi_vals = [r.cluster_nmi for r in results]
    has_cluster = any(np.isfinite(v) for v in ari_vals)
    if has_cluster:
        ax.plot(kl_values, ari_vals, "o-", color="green", label="ARI ↑", linewidth=1.5)
        ax.plot(kl_values, nmi_vals, "s--", color="purple", label="NMI ↑", linewidth=1.5)
        ax.set_xscale("log")
        ax.set_xlabel("KL weight β (log scale)")
        ax.set_ylabel("Cluster agreement (vs reference β)")
        ax.set_title(f"Cluster agreement vs β  (ref = β={reference_kl:g})")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)
    else:
        ax.text(0.5, 0.5, "No clustering output found\n(run with CLUSTER=1 or\ncheck clustering/ subdirs)",
                ha="center", va="center", transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_title("Cluster agreement (not available)")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved plot → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare VAE checkpoints across a KL-weight sweep")
    parser.add_argument("--sweep-dir", required=True,
                        help="Root directory produced by run_kl_sweep.sh (contains kl_<value>/ subdirs)")
    parser.add_argument("--parquet-path", required=True,
                        help="Source parquet (or glob) to draw the reference sample from")
    parser.add_argument("--feature-spec-path", default="ml_features.json")
    parser.add_argument("--row-filter", default=DEFAULT_TRAINING_ROW_FILTER)
    parser.add_argument("--reference-kl", type=float, default=0.05,
                        help="KL weight treated as the reference model. "
                             "All other models are compared to this one. "
                             "Defaults to 0.05 (current pipeline baseline).")
    parser.add_argument("--reference-rows", type=int, default=4000,
                        help="Rows in the fixed reference sample encoded by every checkpoint.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sweep_dir = Path(args.sweep_dir).resolve()
    output_dir = sweep_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover runs
    runs = discover_runs(sweep_dir)
    if not runs:
        print(f"ERROR: no completed runs found under {sweep_dir}", file=sys.stderr)
        return 1
    log(f"Found {len(runs)} completed run(s): β ∈ {[kl for kl, _ in runs]}")

    # Load reference frame
    reference_frame = load_reference_frame(
        parquet_path=args.parquet_path,
        feature_spec_path=args.feature_spec_path,
        row_filter=args.row_filter,
        n_rows=args.reference_rows,
        seed=args.seed,
        threads=args.threads,
    )

    # Encode reference frame with every checkpoint
    latents: dict[float, np.ndarray] = {}
    for kl, run_dir in runs:
        checkpoint = run_dir / "training" / "model.pt"
        log(f"Encoding reference sample with β={kl:g} checkpoint")
        try:
            latents[kl] = encode_checkpoint(
                checkpoint_path=checkpoint,
                reference_frame=reference_frame,
                feature_spec_path=args.feature_spec_path,
                parquet_path=args.parquet_path,
                batch_size=args.batch_size,
            )
        except Exception as exc:
            log(f"  WARNING: encoding failed for β={kl:g}: {exc}")

    # Identify reference model — fall back to closest available kl_weight
    available_kls = [kl for kl, _ in runs if kl in latents]
    if not available_kls:
        print("ERROR: no checkpoints could be encoded", file=sys.stderr)
        return 1
    ref_kl = min(available_kls, key=lambda k: abs(k - args.reference_kl))
    if abs(ref_kl - args.reference_kl) > 1e-9:
        log(f"Requested reference β={args.reference_kl} not found; using β={ref_kl:g} as reference")
    ref_latent = latents[ref_kl]

    # Cluster keys for reference model (for ARI/NMI matching)
    ref_cluster_keys = {kl: load_cluster_keys(run_dir) for kl, run_dir in runs}

    # Build results
    results: list[KLResult] = []
    diagnostics_by_kl: dict[float, list[dict]] = {}
    history_by_kl: dict[float, list[dict]] = {}

    for kl, run_dir in runs:
        summary = load_summary(run_dir)
        diags = load_diagnostics(run_dir)
        history = load_training_history(run_dir)
        diagnostics_by_kl[kl] = diags
        history_by_kl[kl] = history

        es = (summary.get("early_stopping") or {})
        final_epoch_diag = diags[-1] if diags else {}

        r = KLResult(
            kl_weight=kl,
            epochs_run=es.get("epochs_run", len(history)),
            stopped_early=bool(es.get("stopped_early", False)),
            final_active_units=int(es.get("final_active_units", final_epoch_diag.get("active_units", 0))),
            latent_dim=int(summary.get("latent_dim", 16)),
            best_val_total_loss=float(es.get("best_val_total_loss", float("nan"))),
            final_kl_total=float(final_epoch_diag.get("kl_total", float("nan"))),
            train_seconds=float(summary.get("train_seconds", float("nan"))),
        )

        # Latent comparison
        if kl in latents:
            if abs(kl - ref_kl) < 1e-9:
                r.procrustes_distance = 0.0
                r.cka_similarity = 1.0
                r.trustworthiness = 1.0
                r.continuity = 1.0
                r.jaccard_knn10 = 1.0
                r.jaccard_knn30 = 1.0
            else:
                metrics = compare_latent_spaces(ref_latent, latents[kl])
                r.procrustes_distance = metrics["procrustes_distance"]
                r.cka_similarity = metrics["cka_similarity"]
                r.trustworthiness = metrics["trustworthiness"]
                r.continuity = metrics["continuity"]
                r.jaccard_knn10 = metrics["jaccard_knn10"]
                r.jaccard_knn30 = metrics["jaccard_knn30"]

        # Cluster comparison (matched on CHROM/POS/REF/ALT)
        ref_keys = ref_cluster_keys.get(ref_kl)
        cmp_keys = ref_cluster_keys.get(kl)
        if ref_keys is not None and cmp_keys is not None and abs(kl - ref_kl) >= 1e-9:
            cmp = compare_cluster_assignments(ref_keys, cmp_keys)
            r.cluster_ari = cmp["ari"]
            r.cluster_nmi = cmp["nmi"]
        elif abs(kl - ref_kl) < 1e-9:
            r.cluster_ari = 1.0
            r.cluster_nmi = 1.0

        results.append(r)

    print_comparison_table(results, ref_kl)

    plot_sweep(
        results=results,
        diagnostics_by_kl=diagnostics_by_kl,
        history_by_kl=history_by_kl,
        reference_kl=ref_kl,
        output_path=output_dir / "kl_sweep_comparison.png",
    )

    payload = {
        "sweep_dir": str(sweep_dir),
        "reference_kl": ref_kl,
        "reference_rows": args.reference_rows,
        "results": [asdict(r) for r in results],
    }
    out_json = output_dir / "kl_sweep_comparison.json"
    out_json.write_text(json.dumps(payload, indent=2, default=str))
    log(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
