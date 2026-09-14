#!/usr/bin/env python
"""Per-sample inference: VAE encode -> parametric UMAP -> HDBSCAN -> SigProfiler.

For each of the 95 per-sample parquet files:
  1. Filter rows (st='MIXED', et='MIXED', FILT=1) via DuckDB
  2. Encode through the cohort VAE (LatentInference.from_checkpoint)
  3. Project through the parametric UMAP encoder (PyTorch MLP 16->2)
  4. Assign cluster labels via HDBSCAN approximate_predict on the 1M-row cohort model
  5. Build SBS96 per-cluster counts matrix
  6. Run SigProfiler (uv_only)
  7. Generate 4 plots: umap_cluster, umap_substitution, umap_sigprofiler, umap_cosine

SigProfiler and plotting are parallelised across samples (CPU workers) after the GPU pipeline
finishes for each sample.

Usage:
    python umap_hdbscan_sweep/per_parquet_inference.py \\
        --parquet-glob '/data/lab/ppmseq_parquets/*.parquet' \\
        --checkpoint  $HOME/pure-internship/uv_vae/runs/train_multi_20260802T192756Z/training/run_20260802T192814Z/model.pt \\
        --umap-model  $HOME/pure-internship/umap_hdbscan_sweep/umap/results/final_models/13_BEST_25M_nn15_md0.1_umap.pt \\
        --feature-spec $HOME/pure-internship/uv_vae/ml_features.json \\
        --coords      $HOME/pure-internship/umap_hdbscan_sweep/umap_tests/hdbscan_scaling/coords.npy \\
        --context     $HOME/pure-internship/uv_vae/runs/train_multi_20260802T192756Z/stage1_embed/context.parquet \\
        --output-dir  $HOME/pure-internship/umap_hdbscan_sweep/per_parquet_inference
"""
from __future__ import annotations

import argparse
import gc
import glob
import itertools
import json
import multiprocessing as mp
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "uv_vae").is_dir()),
                 Path(__file__).resolve().parents[1])
for _c in (REPO_ROOT / "uv_vae" / "scripts", REPO_ROOT / "uv_vae", REPO_ROOT, Path(__file__).resolve().parent):
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch

# COSMIC palette  (must not import cuML here — subprocess imports it per-worker)
SUBSTITUTIONS = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
SUB_COLOURS = {"C>A": "#03BCEE", "C>G": "#000000", "C>T": "#E32926",
               "T>A": "#CAC9C9", "T>C": "#A1CE63", "T>G": "#EBC6C4"}
UV_ONLY_SIGS = ["SBS7A", "SBS7B", "SBS7C", "SBS7D", "SBS38"]
UV_ONLY_COLOURS = {"SBS7A": "#E32926", "SBS7B": "#03BCEE", "SBS7C": "#A1CE63",
                   "SBS7D": "#EBC6C4", "SBS38": "#7B4FA3"}
NOISE_COLOUR = "#cccccc"
TAB_COLOURS = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors) + list(plt.cm.tab20c.colors)


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def source_fingerprint(path: str) -> dict:
    """Identity of the source file at the moment it was read.

    file_row_number is positional -- duckdb derives it from physical layout rather than
    reading it from the file. It therefore keys a join correctly only while the parquet
    is byte-identical; a rewrite, re-sort or recompaction silently repoints every id at
    a different read. Recording this lets a later join verify the file it is joining
    against is the one that was labelled, instead of trusting that nothing changed.

    The hash covers the parquet footer (schema + row-group offsets + row counts), which
    any rewrite disturbs, rather than the whole multi-GB file.
    """
    import hashlib
    st = os.stat(path)
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        fh.seek(max(0, st.st_size - 1_000_000))
        h.update(fh.read())
    return {
        "path": str(path),
        "size_bytes": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "footer_sha256": h.hexdigest(),
        "read_at": datetime.now(timezone.utc).isoformat(),
    }


def release_memory() -> None:
    """Collect, then hand freed arenas back to the OS.

    glibc's allocator keeps freed blocks on its own free lists, so RSS stayed at
    51 GB between samples even after every large object was dropped. malloc_trim
    is what actually returns them; without it the next sample's allocation grows
    the heap instead of reusing the space.
    """
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def rss_gb() -> float:
    """Resident set size, so a creeping leak shows in the log before it stalls."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024 / 1024
    except Exception:
        pass
    return float("nan")


# ── parametric UMAP loader ──────────────────────────────────────────────────────

def load_parametric_umap(pt_path: str | Path, input_dim: int = 16,
                          hidden: tuple = (256, 256, 128), output_dim: int = 2,
                          device: str = "cpu") -> "torch.nn.Module":
    from parametric_umap import ParametricEncoder
    enc = ParametricEncoder(input_dim=input_dim, hidden=hidden, output_dim=output_dim)
    ckpt = torch.load(pt_path, map_location=device, weights_only=True)
    enc.load_state_dict(ckpt["state_dict"])
    enc.to(device)
    enc.eval()
    return enc


def umap_transform(encoder: "torch.nn.Module", latent: np.ndarray,
                   batch_size: int = 65536, device: str = "cpu") -> np.ndarray:
    out = []
    tensor = torch.tensor(latent, dtype=torch.float32)
    with torch.inference_mode():
        for i in range(0, len(tensor), batch_size):
            batch = tensor[i:i + batch_size].to(device)
            out.append(encoder(batch).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


# ── HDBSCAN CPU model (saveable, supports approximate_predict) ──────────────────

def fit_cpu_hdbscan(coords: np.ndarray, fit_indices: np.ndarray,
                    mcs: int, ms: int, epsilon: float) -> "hdbscan.HDBSCAN":
    import hdbscan as hdbscan_pkg
    fit_coords = np.ascontiguousarray(coords[fit_indices], dtype=np.float64)
    clusterer = hdbscan_pkg.HDBSCAN(
        min_cluster_size=mcs, min_samples=ms,
        cluster_selection_method="eom",
        cluster_selection_epsilon=epsilon,
        prediction_data=True,
    )
    clusterer.fit(fit_coords)
    return clusterer


# ── DuckDB / VAE encode helpers ─────────────────────────────────────────────────

ROW_FILTER = "st = 'MIXED' AND et = 'MIXED' AND FILT = 1"

SBS96_COLUMNS = ["REF", "ALT", "X_PREV1", "X_NEXT1"]


def load_and_filter_parquet(parquet_path: str, columns: list[str] | None = None,
                            max_rows: int | None = None,
                            memory_limit: str | None = None) -> pl.DataFrame:
    """Filtered rows, reading only the columns downstream actually touches.

    SELECT * pulled all ~70 columns when the VAE needs its feature-spec subset and
    SBS96 needs four more; the unused string columns dominated resident memory on a
    50M-row sample.
    """
    import duckdb
    limit_clause = f"LIMIT {max_rows}" if max_rows else ""
    select = "*" if not columns else ", ".join(f'"{c}"' for c in columns)
    conn = duckdb.connect()
    if memory_limit:
        conn.execute(f"SET memory_limit='{memory_limit}'")
    arrow = conn.execute(
        f"SELECT {select} FROM read_parquet('{parquet_path}') WHERE {ROW_FILTER} {limit_clause}"
    ).arrow()
    conn.close()
    frame = pl.from_arrow(arrow)
    del arrow
    return frame


def stream_filtered_parquet(parquet_path: str, columns: list[str] | None,
                            rows_per_batch: int, memory_limit: str | None = None):
    """Filtered rows in batches, never materialising the whole sample.

    The cohort is not uniform: samples range from ~50M rows to 226M+, and the largest
    do not fit in RAM as a single frame no matter how the downstream steps are batched.
    Streaming makes peak memory a function of rows_per_batch instead of sample size.
    """
    import duckdb
    # file_row_number is the physical row index in the source file. It is what makes the
    # outputs joinable back: positional correspondence between the label array and the
    # parquet is NOT guaranteed (duckdb parallelises the scan across row groups), and
    # re-deriving identity later is the mistake recover_source_columns.py exists to undo.
    cols = ["file_row_number"] + (list(columns) if columns else [])
    select = ", ".join(f'"{c}"' if c != "file_row_number" else c for c in cols) if columns else \
             "file_row_number, *"
    conn = duckdb.connect()
    try:
        if memory_limit:
            conn.execute(f"SET memory_limit='{memory_limit}'")
        result = conn.execute(
            f"SELECT {select} FROM read_parquet('{parquet_path}', file_row_number=true) "
            f"WHERE {ROW_FILTER}")
        # fetch_record_batch is deprecated in favour of to_arrow_reader; miletus and
        # tosun run different duckdb versions, so prefer the new name and fall back
        # rather than pinning. Both take the batch size positionally.
        make_reader = getattr(result, "to_arrow_reader", None)
        reader = (make_reader(rows_per_batch) if make_reader is not None
                  else result.fetch_record_batch(rows_per_batch))
        for batch in reader:
            yield pl.from_arrow(batch)
    finally:
        conn.close()


# ── SBS96 helpers ───────────────────────────────────────────────────────────────

_COMPLEMENT = str.maketrans("ACGT", "TGCA")

def _normalize_context(ref: str, alt: str, prev1: str, next1: str) -> str | None:
    try:
        if ref in ("C", "T"):
            return f"{prev1}[{ref}>{alt}]{next1}"
        else:
            r2 = ref.translate(_COMPLEMENT)
            a2 = alt.translate(_COMPLEMENT)
            p2 = next1.translate(_COMPLEMENT)
            n2 = prev1.translate(_COMPLEMENT)
            return f"{p2}[{r2}>{a2}]{n2}"
    except Exception:
        return None



_BYTE2CODE = np.full(256, 4, dtype=np.uint8)   # anything not ACGT -> 4 ("N")
for _i, _b in enumerate(b"ACGT"):
    _BYTE2CODE[_b] = _i
_CODE2CHAR = ["A", "C", "G", "T", "N"]


def build_sbs96_lut(channel_map: dict[str, int]) -> np.ndarray:
    """625-entry table over (prev, ref, alt, next) codes, each 0-4.

    Built by calling _normalize_context on every combination, so the mapping is the
    same one the row-wise version applied -- just evaluated 625 times up front instead
    of once per read.
    """
    lut = np.full(625, -1, dtype=np.int8)
    for p, r, a, n in itertools.product(range(5), repeat=4):
        ch = _normalize_context(_CODE2CHAR[r], _CODE2CHAR[a], _CODE2CHAR[p], _CODE2CHAR[n])
        if ch is not None:
            lut[((p * 5 + r) * 5 + a) * 5 + n] = channel_map.get(ch, -1)
    return lut


def _base_codes(column) -> np.ndarray:
    return _BYTE2CODE[np.asarray(column, dtype="S1").view(np.uint8)]


def frame_to_sbs96_index(frame: pl.DataFrame, lut: np.ndarray) -> np.ndarray:
    """Map each row to its SBS96 channel index via the precomputed lookup table.

    The row-wise version this replaces called .to_list() on four columns, which
    materialised ~4 x n Python str objects (~10 GB per 50M-row sample) that CPython
    does not return to the OS -- RSS grew every sample until the loader thrashed.
    Staying in numpy keeps the whole step at ~n bytes.
    """
    r, a = _base_codes(frame["REF"]), _base_codes(frame["ALT"])
    p, n = _base_codes(frame["X_PREV1"]), _base_codes(frame["X_NEXT1"])
    return lut[((p.astype(np.int32) * 5 + r) * 5 + a) * 5 + n]


def sbs96_counts(sbs96_idx: np.ndarray, labels: np.ndarray,
                 n_clusters: int) -> np.ndarray:
    counts = np.zeros((96, n_clusters), dtype=np.int64)
    valid = (sbs96_idx >= 0) & (labels >= 0)
    flat = sbs96_idx[valid].astype(np.int64) * n_clusters + labels[valid].astype(np.int64)
    bc = np.bincount(flat, minlength=96 * n_clusters).reshape(96, n_clusters)
    counts += bc
    return counts


# ── plotting (one function per plot type) ──────────────────────────────────────

def save_fig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_cluster(xy: np.ndarray, labels: np.ndarray, n_clusters: int,
                 out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    noise = labels < 0
    ax.scatter(xy[noise, 0], xy[noise, 1], c=NOISE_COLOUR, s=0.3, alpha=0.3, rasterized=True)
    for ci in range(n_clusters):
        mask = labels == ci
        if not mask.any():
            continue
        ax.scatter(xy[mask, 0], xy[mask, 1], c=[TAB_COLOURS[ci % len(TAB_COLOURS)]],
                   s=0.5, alpha=0.5, rasterized=True)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    save_fig(fig, out_path)


def plot_coloured(xy: np.ndarray, labels: np.ndarray, colour_map: dict[int, str],
                  out_path: Path, title: str,
                  legend_items: list[tuple[str, str]] | None = None) -> None:
    colours = np.array([colour_map.get(l, NOISE_COLOUR) for l in labels])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xy[:, 0], xy[:, 1], c=colours, s=0.5, alpha=0.5, rasterized=True)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    if legend_items:
        handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                               markersize=7, label=lbl) for lbl, c in legend_items]
        ax.legend(handles=handles, loc="upper right", fontsize=7)
    save_fig(fig, out_path)


def plot_cosine(xy: np.ndarray, labels: np.ndarray, cosine_map: dict[int, float],
                out_path: Path, title: str) -> None:
    values = np.array([cosine_map.get(l, np.nan) for l in labels])
    fig, ax = plt.subplots(figsize=(8, 6))
    noise = np.isnan(values)
    ax.scatter(xy[noise, 0], xy[noise, 1], c=NOISE_COLOUR, s=0.3, alpha=0.3, rasterized=True)
    if (~noise).any():
        sc = ax.scatter(xy[~noise, 0], xy[~noise, 1], c=values[~noise],
                        cmap="viridis", vmin=0, vmax=1, s=0.5, alpha=0.6, rasterized=True)
        plt.colorbar(sc, ax=ax, label="Cosine similarity")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    save_fig(fig, out_path)


# ── SigProfiler / plot worker (runs in separate process) ───────────────────────

def _sigprofiler_and_plots_worker(kwargs: dict) -> dict:
    """Runs in a subprocess to avoid GPU/memory conflicts and SigProfiler re-entrancy."""
    sample_name = kwargs["sample_name"]
    sample_dir = Path(kwargs["sample_dir"])
    sbs96_matrix_path = Path(kwargs["sbs96_matrix_path"])
    labels_path = Path(kwargs["labels_path"])
    xy_path = Path(kwargs["xy_path"])
    row_order = kwargs["row_order"]
    n_clusters = kwargs["n_clusters"]
    genome_build = kwargs["genome_build"]
    cosmic_version = kwargs["cosmic_version"]
    sig_db = kwargs["sig_db"]
    ncpus = kwargs["ncpus"]

    import sys, importlib
    for _c in (str(REPO_ROOT / "uv_vae" / "scripts"), str(REPO_ROOT / "uv_vae"), str(REPO_ROOT), str(Path(__file__).resolve().parent)):
        if _c not in sys.path:
            sys.path.insert(0, _c)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np, polars as pl, re

    t0 = perf_counter()
    result = {"sample": sample_name, "ok": False}

    try:
        labels = np.load(labels_path)
        xy = np.load(xy_path)

        # ── SigProfiler ──────────────────────────────────────────────────────────
        import run_variant_cluster_pipeline as rvcp
        sigprof_dir = sample_dir / f"sigprofilerassignment_uv_only_{genome_build.lower()}_v{cosmic_version}"
        rvcp.Analyzer.cosmic_fit(
            samples=str(sbs96_matrix_path),
            output=str(sigprof_dir / "output"),
            signature_database=str(sig_db),
            genome_build=genome_build, cosmic_version=float(cosmic_version),
            make_plots=False, collapse_to_SBS96=True, connected_sigs=False, verbose=False,
            input_type="matrix", context_type="96", export_probabilities=True,
            sample_reconstruction_plots=False, cpu=ncpus,
            add_background_signatures=False,
        )

        # ── read back SigProfiler outputs ─────────────────────────────────────
        act_path = (sigprof_dir / "output" / "Assignment_Solution" / "Activities"
                    / "Assignment_Solution_Activities.txt")
        stats_path = (sigprof_dir / "output" / "Assignment_Solution" / "Solution_Stats"
                      / "Assignment_Solution_Samples_Stats.txt")

        cosine_map: dict[int, float] = {}
        dom_sig_map: dict[int, str] = {}
        if stats_path.exists():
            frame = pl.read_csv(stats_path, separator="\t")
            # Samples_Stats calls it "Sample Names"; Activities calls it "Samples".
            name_col = next((c for c in ("Sample Names", "Samples") if c in frame.columns), None)
            for row in (frame.iter_rows(named=True) if name_col else []):
                m = re.search(r"(\d+)$", str(row.get(name_col, "")))
                if m:
                    cosine_map[int(m.group(1))] = float(row.get("Cosine Similarity", 0))
        if act_path.exists():
            frame = pl.read_csv(act_path, separator="\t")
            sig_cols = frame.columns[1:]
            for row in frame.iter_rows(named=True):
                m = re.search(r"(\d+)$", str(row.get("Samples", "")))
                if m:
                    counts_arr = np.array([float(row[s]) for s in sig_cols])
                    if counts_arr.sum() > 0:
                        dom_sig_map[int(m.group(1))] = sig_cols[int(np.argmax(counts_arr))]

        # load SBS96 matrix for dominant substitution
        matrix = pl.read_csv(sbs96_matrix_path, separator="\t")
        channels = matrix[:, 0].to_list()
        subs_per_channel = [c[c.index("[") + 1:c.index("]")] for c in channels]
        dom_sub_map: dict[int, str] = {}
        for col in matrix.columns[1:]:
            m = re.search(r"(\d+)$", col)
            if m:
                arr = matrix[col].to_numpy()
                if arr.sum() > 0:
                    dom_sub_map[int(m.group(1))] = subs_per_channel[int(np.argmax(arr))]

        # ── plots ────────────────────────────────────────────────────────────
        plots_dir = sample_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        noise = labels < 0
        noise_pct = noise.mean() * 100

        # 1. HDBSCAN cluster
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(xy[noise, 0], xy[noise, 1], c=NOISE_COLOUR, s=0.5, alpha=0.3, rasterized=True)
        for ci in range(n_clusters):
            mask = labels == ci
            if not mask.any(): continue
            ax.scatter(xy[mask, 0], xy[mask, 1], c=[TAB_COLOURS[ci % len(TAB_COLOURS)]],
                       s=0.5, alpha=0.5, rasterized=True)
        ax.set_title(f"{sample_name} — clusters  ({noise_pct:.1f}% noise)", fontsize=10)
        ax.axis("off")
        fig.savefig(plots_dir / "umap_cluster.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 2. Substitution
        sub_cmap = {ci: SUB_COLOURS.get(s, NOISE_COLOUR) for ci, s in dom_sub_map.items()}
        colours = np.array([sub_cmap.get(l, NOISE_COLOUR) for l in labels])
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(xy[:, 0], xy[:, 1], c=colours, s=0.5, alpha=0.5, rasterized=True)
        handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=SUB_COLOURS[s],
                               markersize=7, label=s) for s in SUBSTITUTIONS]
        ax.legend(handles=handles, loc="upper right", fontsize=7)
        ax.set_title(f"{sample_name} — dominant substitution", fontsize=10)
        ax.axis("off")
        fig.savefig(plots_dir / "umap_substitution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 3. SigProfiler
        sig_cmap = {ci: UV_ONLY_COLOURS.get(s, NOISE_COLOUR) for ci, s in dom_sig_map.items()}
        colours_sp = np.array([sig_cmap.get(l, NOISE_COLOUR) for l in labels])
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(xy[:, 0], xy[:, 1], c=colours_sp, s=0.5, alpha=0.5, rasterized=True)
        handles_sp = [plt.Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=UV_ONLY_COLOURS[s],
                                  markersize=7, label=s) for s in UV_ONLY_SIGS]
        ax.legend(handles=handles_sp, loc="upper right", fontsize=7)
        ax.set_title(f"{sample_name} — dominant SigProfiler signature (uv_only)", fontsize=10)
        ax.axis("off")
        fig.savefig(plots_dir / "umap_sigprofiler.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 4. Cosine similarity
        cos_vals = np.array([cosine_map.get(l, np.nan) for l in labels])
        no_val = np.isnan(cos_vals)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(xy[no_val, 0], xy[no_val, 1], c=NOISE_COLOUR, s=0.3, alpha=0.3, rasterized=True)
        if (~no_val).any():
            sc = ax.scatter(xy[~no_val, 0], xy[~no_val, 1], c=cos_vals[~no_val],
                            cmap="viridis", vmin=0, vmax=1, s=0.5, alpha=0.6, rasterized=True)
            plt.colorbar(sc, ax=ax, label="Cosine similarity")
        mean_cos = float(np.nanmean(cos_vals)) if (~no_val).any() else float("nan")
        ax.set_title(f"{sample_name} — cosine similarity  (mean {mean_cos:.3f})", fontsize=10)
        ax.axis("off")
        fig.savefig(plots_dir / "umap_cosine.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        result.update({"ok": True, "seconds": round(perf_counter() - t0, 1),
                       "n_clusters": n_clusters, "noise_pct": noise_pct,
                       "mean_cosine": mean_cos})
    except Exception as exc:
        result["error"] = str(exc)
        import traceback; traceback.print_exc()

    return result


# ── main pipeline ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet-glob", required=True, help="glob for per-sample parquets")
    p.add_argument("--checkpoint", required=True, help="path to model.pt VAE checkpoint")
    p.add_argument("--feature-spec", required=True, help="path to ml_features.json")
    p.add_argument("--umap-model", required=True, help="path to parametric UMAP .pt file")
    p.add_argument("--coords", required=True,
                   help="cohort coords.npy -- the space the HDBSCAN model lives in")
    p.add_argument("--hdbscan-model", default=None,
                   help="joblib .pkl of an already-fitted HDBSCAN (e.g. the low-noise "
                        "cohort model). Without this the script fits its own.")
    p.add_argument("--fit-indices", default=None,
                   help="fit_indices.npy the model was fit on. Defaults to "
                        "fit_indices.npy beside --hdbscan-model.")
    p.add_argument("--context", required=True, help="context.parquet for sbs96_index")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--mcs", type=int, default=2500)
    p.add_argument("--ms", type=int, default=15)
    p.add_argument("--epsilon", type=float, default=0.0,
                   help="cluster_selection_epsilon for the fallback fit. Ignored "
                        "when --hdbscan-model is given. 0.0 is the deployed cell; "
                        "the 0.05 this used to default to came from a parameter "
                        "cell that was abandoned.")
    p.add_argument("--fit-rows", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--genome-build", default="GRCh38")
    p.add_argument("--cosmic-version", default="3.5")
    p.add_argument("--sigprofiler-cpu", type=int, default=4, help="CPUs per SigProfiler job")
    p.add_argument("--n-workers", type=int, default=4, help="parallel SigProfiler+plot workers")
    p.add_argument("--stream-batch-rows", type=int, default=5_000_000,
                   help="rows per streaming batch through encode/UMAP/predict/count. "
                        "Peak memory tracks this, not the sample size -- the cohort "
                        "ranges from ~50M to 226M+ rows per file.")
    p.add_argument("--encode-batch", type=int, default=4096,
                   help="GPU forward-pass batch inside each encode slice")
    p.add_argument("--duckdb-memory-limit", default="32GB",
                   help="DuckDB memory_limit per load; it otherwise takes 80%% of RAM "
                        "and competes with the frame it is materialising.")
    p.add_argument("--gpu-budget-gb", type=float, default=44.0,
                   help="GPU budget for this process; RMM gets 0.9 of it. The library "
                        "default is 16 GB, too small for a 48 GB card.")
    p.add_argument("--predict-batch-rows", type=int, default=5_000_000,
                   help="rows per fast_predict batch. Peak GPU is ~240 B/row, so 5M "
                        "holds it near 1.2 GB. Lower this if RBC still OOMs.")
    p.add_argument("--umap-input-dim", type=int, default=16)
    p.add_argument("--umap-batch", type=int, default=65536)
    p.add_argument("--skip-done", action="store_true",
                   help="skip samples that already have plots/umap_cosine.png")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(glob.glob(args.parquet_glob))
    if not parquet_files:
        log(f"No files matched: {args.parquet_glob}")
        return 1
    log(f"Found {len(parquet_files)} parquet files")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")

    # ── build SBS96 channel order from uv_only signature DB ─────────────────────
    import run_variant_cluster_pipeline as rvcp
    sig_db_dir = out / "signature_db"
    sig_db_dir.mkdir(exist_ok=True)
    sig_db = sig_db_dir / f"uv_only_SBS_{args.genome_build}.tsv"
    row_order = rvcp.write_uv_only_signature_database(
        args.genome_build, args.cosmic_version, sig_db)
    channel_map = {ch: i for i, ch in enumerate(row_order)}
    sbs96_lut = build_sbs96_lut(channel_map)
    log(f"SigProfiler DB: {sig_db.name}  ({len(row_order)} channels)")

    # ── HDBSCAN model: reuse the cohort model, don't refit ──────────────────────
    # fast_predict.build_tables/build_index need the EXACT fit set the model was fit on,
    # in the same row order -- the tables index into it positionally. So the fit indices
    # travel with the model rather than being re-derived from (seed, fit_rows) and hoped
    # to match: a silent mismatch would relabel every read against the wrong neighbours.
    import joblib
    coords = np.load(args.coords, mmap_mode="r")

    if args.hdbscan_model:
        model_path = Path(args.hdbscan_model)
        if not model_path.exists():
            log(f"ERROR: --hdbscan-model not found: {model_path}")
            return 1
        log(f"Loading HDBSCAN model: {model_path}")
        clusterer = joblib.load(model_path)

        idx_path = Path(args.fit_indices) if args.fit_indices else model_path.parent / "fit_indices.npy"
        if idx_path.exists():
            fit_idx = np.load(idx_path)
            log(f"  fit indices from {idx_path.name}  ({len(fit_idx):,} rows)")
        else:
            # Fall back to the shared derivation both scripts use. Only valid when the
            # model really was fit with this seed/fit_rows, so make that check loud.
            fit_idx = np.sort(np.random.default_rng(args.seed).choice(
                coords.shape[0], size=min(args.fit_rows, coords.shape[0]), replace=False))
            log(f"  WARNING: {idx_path.name} absent; re-deriving fit indices from "
                f"seed={args.seed} fit_rows={args.fit_rows}")
            n_model = getattr(clusterer, "labels_", np.empty(0)).shape[0]
            if n_model != len(fit_idx):
                log(f"ERROR: model was fit on {n_model:,} rows but the derived index has "
                    f"{len(fit_idx):,}. Pass --fit-indices with the model's own indices.")
                return 1
    else:
        model_path = out / f"hdbscan_mcs{args.mcs}_ms{args.ms}_eps{args.epsilon:.3f}.pkl"
        idx_path = out / "fit_indices.npy"
        if model_path.exists() and idx_path.exists():
            log(f"Loading existing HDBSCAN model from {model_path.name}")
            clusterer = joblib.load(model_path)
            fit_idx = np.load(idx_path)
        else:
            fit_idx = np.sort(np.random.default_rng(args.seed).choice(
                coords.shape[0], size=min(args.fit_rows, coords.shape[0]), replace=False))
            log(f"Fitting CPU HDBSCAN on {len(fit_idx):,} rows  (mcs={args.mcs} ms={args.ms}) …")
            t0 = perf_counter()
            clusterer = fit_cpu_hdbscan(coords, fit_idx, args.mcs, args.ms, args.epsilon)
            log(f"  {perf_counter()-t0:.1f}s  {int(clusterer.labels_.max())+1} clusters  "
                f"{(clusterer.labels_<0).mean()*100:.1f}% noise")
            joblib.dump(clusterer, model_path)
            np.save(idx_path, fit_idx)
            log(f"  model + fit_indices saved → {out}")

    n_cohort_clusters = int(clusterer.labels_.max()) + 1 if (clusterer.labels_ >= 0).any() else 0
    log(f"HDBSCAN model: {n_cohort_clusters} cohort clusters, fit on {len(fit_idx):,} rows")

    # min_samples must be the model's own, not the CLI default: build_tables uses it for
    # the core-distance estimate and build_index for k=2*ms.
    model_ms = int(getattr(clusterer, "min_samples", None) or getattr(clusterer, "min_cluster_size", args.ms))
    if model_ms != args.ms:
        log(f"  using model's min_samples={model_ms} (CLI --ms {args.ms} ignored)")

    # ── build fast_predict tables + index once (reused across all samples) ───────
    import fast_predict as _fp
    _fit_coords_fp = np.ascontiguousarray(coords[fit_idx], dtype=np.float32)
    del coords

    _predict_backend = "rbc"
    try:
        import cuml  # noqa: F401
        import sweep_core
        # Stage "apply", not "sweep": this process loads torch (VAE + UMAP encoder)
        # alongside cuML, so the budget has to be split (rmm_share 0.5) rather than
        # handed to RMM whole.
        #
        # Pin the budget in the environment too. gpu_budget.apply() computes torch's
        # share as (budget - rmm_pool); LatentInference.from_checkpoint calls it again
        # with no argument, and the 16 GB default against an already-reserved RMM pool
        # yields a negative fraction and a ValueError.
        os.environ["UV_VAE_GPU_MEM_GB"] = str(args.gpu_budget_gb)
        sweep_core.apply_gpu_budget("apply", budget_gb=args.gpu_budget_gb,
                                    require_free=False)
    except ImportError:
        _predict_backend = "sklearn"
        log("cuML not available, fast_predict will use sklearn KDTree backend")

    _fp_tables = _fp.build_tables(clusterer, len(fit_idx), model_ms)
    _fp_index  = _fp.build_index(_fit_coords_fp, 2 * model_ms, _predict_backend)
    log(f"fast_predict tables+index built  backend={_predict_backend}  k={2*model_ms}")

    # ── load parametric UMAP encoder ────────────────────────────────────────────
    log(f"Loading parametric UMAP encoder from {Path(args.umap_model).name} …")
    umap_encoder = load_parametric_umap(args.umap_model, input_dim=args.umap_input_dim, device=device)
    log(f"  encoder loaded")

    # ── load VAE (shared across samples) ────────────────────────────────────────
    log(f"Loading VAE from {Path(args.checkpoint).name} …")
    from uv_vae.inference import LatentInference
    inf = LatentInference.from_checkpoint(args.checkpoint, feature_spec_path=args.feature_spec)
    log("  VAE loaded")

    # Read only the columns that are actually consumed: the VAE's own feature columns
    # plus the four SBS96 context columns. Derived from the loaded specs rather than
    # hardcoded, so a feature-spec change cannot silently drop an input column.
    read_columns = None
    try:
        spec_columns = list(inf.feature_names)
        read_columns = sorted(set(spec_columns) | set(SBS96_COLUMNS))
        log(f"  reading {len(read_columns)} columns "
            f"({len(spec_columns)} feature + {len(SBS96_COLUMNS)} SBS96 context)")
    except Exception as exc:  # noqa: BLE001
        log(f"  WARNING: could not derive column subset ({exc}); reading all columns")

    # ── per-sample GPU pipeline ──────────────────────────────────────────────────
    worker_jobs: list[dict] = []

    for idx, pq_path in enumerate(parquet_files):
        sample_name = Path(pq_path).stem
        sample_dir = out / sample_name
        sample_dir.mkdir(exist_ok=True)

        cosine_done = (sample_dir / "plots" / "umap_cosine.png").exists()
        if args.skip_done and cosine_done:
            log(f"[{idx+1}/{len(parquet_files)}] {sample_name}  SKIP (done)")
            continue

        log(f"[{idx+1}/{len(parquet_files)}] {sample_name}   (RSS {rss_gb():.1f} GB)")

        # Stream the sample: encode -> project -> label -> count, one batch at a time.
        # Nothing sample-sized is ever resident except the outputs (labels ~4 B/row,
        # coords ~8 B/row), so a 226M-row file costs the same peak as a 50M-row one.
        log(f"  streaming {pq_path} …")
        t0 = perf_counter()
        label_parts: list[np.ndarray] = []
        xy_parts: list[np.ndarray] = []
        rowid_parts: list[np.ndarray] = []
        counts = np.zeros((96, n_cohort_clusters), dtype=np.int64)
        n_rows = 0
        try:
            for batch in stream_filtered_parquet(pq_path, read_columns,
                                                 args.stream_batch_rows,
                                                 args.duckdb_memory_limit):
                if batch.height == 0:
                    continue
                # Keep the source row id, then hand the VAE only its feature columns.
                rowid_parts.append(batch["file_row_number"].to_numpy().astype(np.int64))
                batch = batch.drop("file_row_number")
                latent = inf.encode_frame(batch, batch_size=args.encode_batch)
                xy = umap_transform(umap_encoder, latent.astype(np.float32),
                                    batch_size=args.umap_batch, device=device)
                del latent
                lab, _prob = _fp.predict(
                    _fp_tables, _fit_coords_fp,
                    np.ascontiguousarray(xy, dtype=np.float32),
                    backend=_predict_backend, batch_rows=args.predict_batch_rows,
                    index=_fp_index)
                lab = lab.astype(np.int32)
                counts += sbs96_counts(frame_to_sbs96_index(batch, sbs96_lut),
                                       lab, n_cohort_clusters)
                label_parts.append(lab)
                xy_parts.append(np.asarray(xy, dtype=np.float32))
                n_rows += batch.height
                del batch, xy, lab
                if (n_rows // max(args.stream_batch_rows, 1)) % 10 == 0:
                    log(f"    {n_rows:,} rows  (RSS {rss_gb():.1f} GB)")
        except Exception as exc:
            log(f"  ERROR streaming: {type(exc).__name__}: {exc}")
            del label_parts, xy_parts, rowid_parts
            release_memory()
            continue

        if n_rows == 0:
            log("  no rows after filter, skipping")
            continue
        log(f"  {n_rows:,} rows encoded+projected+labelled in {perf_counter()-t0:.1f}s"
            f"  (RSS {rss_gb():.1f} GB)")

        labels = np.concatenate(label_parts) if len(label_parts) > 1 else label_parts[0]
        xy = np.concatenate(xy_parts) if len(xy_parts) > 1 else xy_parts[0]
        rowids = np.concatenate(rowid_parts) if len(rowid_parts) > 1 else rowid_parts[0]
        del label_parts, xy_parts, rowid_parts
        n_sample_clusters = int(labels.max()) + 1 if (labels >= 0).any() else 0
        noise_pct = (labels < 0).mean() * 100
        log(f"  {n_sample_clusters} clusters  {noise_pct:.1f}% noise")

        # save labels + coords for worker (streaming order, positional)
        np.save(sample_dir / "labels.npy", labels)
        np.save(sample_dir / "umap_coords.npy", xy)

        # The joinable artefact: keyed on the source file's physical row number, sorted
        # so a merge against read_parquet(..., file_row_number=true) is a straight
        # equi-join. Rows the filter dropped are simply absent -- no placeholder rows,
        # and no assumption that array position means anything.
        order = np.argsort(rowids, kind="stable")
        pl.DataFrame({
            "file_row_number": rowids[order],
            "umap_1": xy[order, 0].astype(np.float32),
            "umap_2": xy[order, 1].astype(np.float32),
            "cluster_label": labels[order],
        }).write_parquet(sample_dir / "row_assignments.parquet")
        n_assigned = len(rowids)
        frn_lo, frn_hi = int(rowids.min()), int(rowids.max())
        log(f"  row_assignments.parquet written  ({n_assigned:,} rows, "
            f"file_row_number {frn_lo:,}-{frn_hi:,})")
        del xy, labels, rowids, order

        # Provenance for the join: which file these ids refer to, and what it looked
        # like when it was read.
        (sample_dir / "source_fingerprint.json").write_text(json.dumps({
            **source_fingerprint(pq_path),
            "row_filter": ROW_FILTER,
            "rows_assigned": n_assigned,
            "file_row_number_min": frn_lo,
            "file_row_number_max": frn_hi,
            "join_key": "file_row_number",
            "note": "file_row_number is positional; valid only while the source parquet "
                    "is byte-identical. Verify with verify_source_fingerprint.py.",
        }, indent=2))

        # counts were accumulated per batch during the stream
        present = np.nonzero(counts.sum(axis=0) > 0)[0]
        cluster_cols = [f"cluster_{int(c)}" for c in present]
        counts_present = counts[:, present]
        log(f"  SBS96 done  {len(present)} non-empty clusters  in {perf_counter()-t0:.1f}s")

        input_dir = sample_dir / "sigprofiler_input"
        input_dir.mkdir(exist_ok=True)
        matrix_path = input_dir / "cluster_sbs96_matrix.tsv"
        pl.DataFrame(
            {"Type": list(row_order), **{c: counts_present[:, i].tolist() for i, c in enumerate(cluster_cols)}}
        ).write_csv(matrix_path, separator="\t")
        log(f"  SBS96 matrix written  ({counts_present.sum():,} mutations)")
        del counts

        release_memory()

        worker_jobs.append({
            "sample_name": sample_name,
            "sample_dir": str(sample_dir),
            "sbs96_matrix_path": str(matrix_path),
            "labels_path": str(sample_dir / "labels.npy"),
            "xy_path": str(sample_dir / "umap_coords.npy"),
            "row_order": row_order,
            "n_clusters": n_cohort_clusters,
            "genome_build": args.genome_build,
            "cosmic_version": args.cosmic_version,
            "sig_db": str(sig_db),
            "ncpus": args.sigprofiler_cpu,
        })

    # ── parallel SigProfiler + plots ─────────────────────────────────────────────
    if not worker_jobs:
        log("No new samples to process in SigProfiler/plots phase.")
        return 0

    log(f"\nRunning SigProfiler + plots for {len(worker_jobs)} samples "
        f"({args.n_workers} workers, {args.sigprofiler_cpu} CPUs each) …")

    ctx = mp.get_context("spawn")
    results = []
    with ctx.Pool(processes=args.n_workers, maxtasksperchild=1) as pool:
        for i, res in enumerate(pool.imap_unordered(_sigprofiler_and_plots_worker, worker_jobs), 1):
            status = "OK" if res["ok"] else f"FAIL: {res.get('error','?')}"
            log(f"  [{i}/{len(worker_jobs)}] {res['sample']:30s}  {status}  "
                f"cos={res.get('mean_cosine', float('nan')):.3f}  {res.get('seconds',0):.0f}s")
            results.append(res)

    n_ok = sum(1 for r in results if r["ok"])
    log(f"\nDone: {n_ok}/{len(results)} samples succeeded  ->  {out}")
    (out / "run_summary.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
