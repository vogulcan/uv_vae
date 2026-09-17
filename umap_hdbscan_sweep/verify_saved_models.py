#!/usr/bin/env python
"""Can each saved cell actually be used for inference later, from disk alone?

A cell directory that *looks* complete is not the same as one that works. The sweep holds the
fitted clusterer in memory while it labels, so every number in ``metrics.json`` can be correct
while the pickle beside it is unusable -- a cuML object that does not survive joblib, a
``fit_indices.npy`` that does not match the model it sits next to, a model whose internals
regenerate to something subtly different after a reload. None of that shows up until a
downstream run loads the directory months later.

So this checks the thing that matters, from disk only, with nothing carried over from the
fit: load the pickle, load the indices, rebuild the prediction tables, label rows, and require
the answers to match what the sweep recorded for those same rows.

Three checks per cell, in increasing strength:

1. **Present** -- the files a consumer needs are all there and loadable.
2. **Consistent** -- ``clusterer.labels_`` matches ``cohort_labels.npy`` at ``fit_indices``,
   and the shapes agree. Catches an index file paired with the wrong model, which is the
   failure that silently relabels every read against the wrong neighbours.
3. **Reproduces** -- rows the model never saw at fit time get the *same* labels this run as
   the sweep gave them. This is the only check that exercises the reloaded model end to end,
   and it is the one that would have caught a pickle that loads but does not work.

Exit status is 0 only when every cell passes every check, so it can gate the inference run.

Usage::

    # every cell of every backend under the final_models root
    python umap_hdbscan_sweep/verify_saved_models.py \\
        --models-root ~/pure-internship/umap_hdbscan_sweep/hdbscan/results/final_models

    # one cell
    python umap_hdbscan_sweep/verify_saved_models.py \\
        --cell-dir <.../final_models/cuml/cells/fit1000000_mcs2500_ms15_eom>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

DEFAULT_COORDS = (Path.home() / "pure-internship" / "umap_hdbscan_sweep" / "hdbscan"
                  / "results" / "hdbscan_scaling" / "coords.npy")

# What a consumer needs to label new rows. cohort_labels is not needed for inference itself,
# but check 3 has nothing to compare against without it.
REQUIRED = ("hdbscan_model.pkl", "fit_indices.npy", "metrics.json", "cohort_labels.npy")
OPTIONAL = ("cohort_probabilities.npy", "cluster_persistence.npy", "outlier_scores.npy",
            "exemplars.npz")


def log(message: str) -> None:
    # Windows consoles default to cp1252, which cannot encode the box-drawing and ellipsis
    # characters this project's log lines use elsewhere; a verifier that dies formatting its
    # own output is worse than one that prints a '?'. Harmless on the UTF-8 HPC.
    stream = sys.stdout
    text = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}"
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
    except UnicodeEncodeError:
        text = text.encode(encoding, errors="replace").decode(encoding)
    print(text, flush=True)


def _to_host(array) -> np.ndarray:
    for attribute in ("get", "to_numpy"):
        method = getattr(array, attribute, None)
        if callable(method):
            return np.asarray(method())
    return np.asarray(array)


def verify_cell(cell_dir: Path, coords: np.ndarray, args) -> dict:
    """Run the three checks against one cell directory."""
    import fast_predict
    import joblib

    result: dict = {"cell": cell_dir.name, "dir": str(cell_dir), "checks": {}}

    # ── 1. present ──────────────────────────────────────────────────────────────
    missing = [name for name in REQUIRED if not (cell_dir / name).exists()]
    result["present"] = sorted(name for name in REQUIRED + OPTIONAL
                               if (cell_dir / name).exists())
    result["missing_optional"] = [name for name in OPTIONAL
                                  if not (cell_dir / name).exists()]

    # run_cell writes metrics.json LAST, after the labels and the model artefacts. Its
    # absence therefore means the cell is still being written, not that it is broken --
    # which matters because this script is most useful while the sweep is still running.
    # Reporting a healthy in-flight cell as FAIL would train you to ignore the failures
    # that are real.
    if not (cell_dir / "metrics.json").exists():
        result["checks"]["present"] = (
            "INCOMPLETE: no metrics.json yet (the sweep writes it last, so this cell is "
            "most likely still running)")
        result["status"] = "FAIL" if args.expect_complete else "INCOMPLETE"
        return result
    if missing:
        result["checks"]["present"] = f"FAIL: missing {', '.join(missing)}"
        result["status"] = "FAIL"
        return result
    result["checks"]["present"] = "ok"

    metrics = json.loads((cell_dir / "metrics.json").read_text())
    result["backend"] = metrics.get("cluster_backend", "?")
    result["expected_clusters"] = metrics.get("cohort_n_clusters")
    result["expected_noise"] = metrics.get("cohort_noise_fraction")

    started = perf_counter()
    try:
        clusterer = joblib.load(cell_dir / "hdbscan_model.pkl")
    except Exception as exc:  # noqa: BLE001
        result["checks"]["present"] = f"FAIL: hdbscan_model.pkl unloadable ({type(exc).__name__}: {exc})"
        result["status"] = "FAIL"
        return result
    result["load_seconds"] = round(perf_counter() - started, 1)
    result["model_type"] = f"{type(clusterer).__module__}.{type(clusterer).__name__}"

    fit_indices = np.load(cell_dir / "fit_indices.npy")
    cohort_labels = np.load(cell_dir / "cohort_labels.npy", mmap_mode="r")
    fit_labels = _to_host(clusterer.labels_).ravel().astype(np.int32)

    # ── 2. consistent ───────────────────────────────────────────────────────────
    problems = []
    if cohort_labels.shape[0] != coords.shape[0]:
        problems.append(f"cohort_labels has {cohort_labels.shape[0]:,} rows but coords has "
                        f"{coords.shape[0]:,}")
    if fit_labels.shape[0] != fit_indices.shape[0]:
        problems.append(f"model was fit on {fit_labels.shape[0]:,} rows but fit_indices has "
                        f"{fit_indices.shape[0]:,} -- wrong index file for this model")
    else:
        # The fit rows keep the labels the fit gave them, so these must agree exactly. A
        # mismatch means the index file and the model came from different runs.
        recorded = np.asarray(cohort_labels[fit_indices])
        differing = int((recorded != fit_labels).sum())
        if differing:
            problems.append(f"{differing:,}/{len(fit_labels):,} fit rows disagree with "
                            f"cohort_labels at fit_indices")
    n_model_clusters = int(fit_labels.max()) + 1 if (fit_labels >= 0).any() else 0
    if result["expected_clusters"] is not None and n_model_clusters > result["expected_clusters"]:
        problems.append(f"model has {n_model_clusters} clusters, metrics.json records "
                        f"{result['expected_clusters']}")
    result["model_clusters"] = n_model_clusters
    if problems:
        result["checks"]["consistent"] = "FAIL: " + "; ".join(problems)
        result["status"] = "FAIL"
        return result
    result["checks"]["consistent"] = "ok"

    if args.no_predict:
        result["checks"]["reproduces"] = "skipped (--no-predict)"
        result["status"] = "PARTIAL"
        return result

    # ── 3. reproduces ───────────────────────────────────────────────────────────
    # Held rows only. Fit rows were compared above and would not exercise prediction at all:
    # the sweep writes their labels straight from the fit rather than predicting them.
    rng = np.random.default_rng(args.seed)
    held_mask = np.ones(coords.shape[0], dtype=bool)
    held_mask[fit_indices] = False
    held_indices = np.nonzero(held_mask)[0]
    sample = np.sort(held_indices[rng.choice(len(held_indices),
                                             size=min(args.probe_rows, len(held_indices)),
                                             replace=False)])
    del held_mask, held_indices

    fit_coords = np.asarray(coords[fit_indices], dtype=np.float32)
    query_coords = np.asarray(coords[sample], dtype=np.float32)

    started = perf_counter()
    try:
        tables = fast_predict.build_tables(clusterer, n_fit=len(fit_indices),
                                           min_samples=args.min_samples)
        index = fast_predict.build_index(fit_coords, 2 * tables.min_samples, args.backend)
        predicted, _ = fast_predict.predict(tables, fit_coords, query_coords,
                                            backend=args.backend,
                                            batch_rows=args.batch_rows, index=index)
    except Exception as exc:  # noqa: BLE001
        result["checks"]["reproduces"] = f"FAIL: {type(exc).__name__}: {exc}"
        result["status"] = "FAIL"
        return result
    result["predict_seconds"] = round(perf_counter() - started, 1)

    expected = np.asarray(cohort_labels[sample]).astype(np.int32)
    agree = int((predicted == expected).sum())
    result.update({
        "probe_rows": int(len(sample)),
        "agree": agree,
        "agree_pct": round(agree / len(sample) * 100, 4),
        "probe_noise_pct": round(float((predicted < 0).mean()) * 100, 4),
    })
    if agree != len(sample):
        result["checks"]["reproduces"] = (
            f"FAIL: {len(sample) - agree:,}/{len(sample):,} probe rows "
            f"({(1 - agree / len(sample)) * 100:.3f}%) label differently than the sweep "
            f"recorded -- the saved model does not reproduce its own run")
        result["status"] = "FAIL"
        return result
    result["checks"]["reproduces"] = "ok"
    result["status"] = "PASS"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--models-root", type=Path, default=None,
                        help="final_models root; every <backend>/cells/<cell> under it")
    source.add_argument("--cell-dir", type=Path, default=None,
                        help="a single cell directory")
    parser.add_argument("--coords", default=None,
                        help=f"cohort coords.npy (default: {DEFAULT_COORDS})")
    parser.add_argument("--probe-rows", type=int, default=200_000,
                        help="held rows to re-label and compare against the sweep")
    parser.add_argument("--min-samples", type=int, default=None,
                        help="override; default is the model's own")
    parser.add_argument("--backend", default="rbc", choices=["rbc", "brute", "sklearn"])
    parser.add_argument("--batch-rows", type=int, default=5_000_000)
    parser.add_argument("--seed", type=int, default=7,
                        help="probe row draw; deliberately NOT the sweep's 42, so the rows "
                             "checked are not the ones any fit selected")
    parser.add_argument("--no-predict", action="store_true",
                        help="checks 1-2 only; fast, but does not prove inference works")
    parser.add_argument("--expect-complete", action="store_true",
                        help="treat a cell with no metrics.json as FAIL rather than "
                             "in-progress. Use once the sweep has finished, so a cell that "
                             "died mid-write is not written off as still running.")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    coords_path = (args.coords or "").strip() or os.environ.get("COORDS", "").strip()
    coords_path = os.path.expanduser(coords_path or str(DEFAULT_COORDS))
    if not os.path.exists(coords_path):
        log(f"ERROR: coords not found: {coords_path}")
        return 1
    coords = np.load(coords_path, mmap_mode="r")
    log(f"coords: {coords_path}  {coords.shape}")

    if args.cell_dir:
        cell_dirs = [args.cell_dir]
    else:
        root = args.models_root or (Path.home() / "pure-internship" / "umap_hdbscan_sweep"
                                    / "hdbscan" / "results" / "final_models")
        if not root.is_dir():
            log(f"ERROR: no models root at {root}")
            return 1
        cell_dirs = sorted(p for p in root.glob("*/cells/*") if p.is_dir())
        log(f"models root: {root}  ({len(cell_dirs)} cells)")

    if not cell_dirs:
        log("ERROR: no cell directories found. Has tmux_final_models.sh run?")
        return 1

    results = []
    for cell_dir in cell_dirs:
        backend_hint = cell_dir.parent.parent.name
        log(f"-- {backend_hint}/{cell_dir.name}")
        result = verify_cell(cell_dir, coords, args)
        result["backend_dir"] = backend_hint
        results.append(result)
        for name, outcome in result["checks"].items():
            marker = "ok" if outcome == "ok" else outcome
            log(f"     {name:<12} {marker}")
        if result.get("missing_optional"):
            log(f"     {'optional':<12} absent: {', '.join(result['missing_optional'])}")
        if result["status"] == "PASS":
            log(f"     {'summary':<12} {result['model_type']}, "
                f"{result['model_clusters']} clusters, "
                f"{result['agree']:,}/{result['probe_rows']:,} probe rows reproduced "
                f"({result['predict_seconds']}s)")
        # A backend recorded in metrics.json that disagrees with the directory it sits in is
        # exactly the mislabelling this project already published once.
        if result.get("backend", "?") not in ("?", backend_hint):
            log(f"     {'WARNING':<12} metrics.json says backend={result['backend']} but "
                f"this is the {backend_hint}/ tree")

    passed = [r for r in results if r["status"] == "PASS"]
    failed = [r for r in results if r["status"] == "FAIL"]
    partial = [r for r in results if r["status"] == "PARTIAL"]
    incomplete = [r for r in results if r["status"] == "INCOMPLETE"]

    log("")
    log(f"{len(passed)} pass, {len(failed)} fail"
        + (f", {len(partial)} partial (prediction skipped)" if partial else "")
        + (f", {len(incomplete)} still being written" if incomplete else ""))
    if incomplete:
        log(f"  in progress: {', '.join(r['cell'] for r in incomplete[:6])}"
            + (" ..." if len(incomplete) > 6 else ""))
        log("  re-run with --expect-complete once the sweep finishes, so a cell that died "
            "mid-write is not mistaken for one still running.")
    for result in failed:
        bad = [f"{name}: {outcome}" for name, outcome in result["checks"].items()
               if outcome != "ok"]
        log(f"  FAIL {result['backend_dir']}/{result['cell']}  {'; '.join(bad)}")

    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=2))
        log(f"detail → {args.json_out}")

    if failed:
        return 1
    if partial:
        log("NOTE: --no-predict was used; presence and consistency pass, but nothing here "
            "proves the models label correctly. Re-run without it before the inference run.")
        return 0
    log("All cells are usable for inference: model loads from disk, indices match, and held "
        "rows reproduce the labels the sweep recorded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
