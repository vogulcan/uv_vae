"""Build the per-file statistics cache and print the interleaving plan.

Run this **before** the first multi-file training run.  It is the cheapest way to
confirm that every parquet parses, that the row filter behaves as expected, and
that no sample is starved of draws at the chosen batch size -- and the cache it
writes is reused by every later run, so it is not throwaway work.

    python uv_vae/scripts/multi_parquet_stats.py \
        --parquet-paths '/data/lab/ppmseq_parquets/*.featuremap.parquet' \
        --feature-spec-path uv_vae/ml_features.json \
        --stats-cache-path ~/pure-internship/uv_vae/stats_cache.json \
        --batch-size 8192

It imports nothing from torch, so it runs on a login node.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from uv_vae.features import load_feature_specs
from uv_vae.multi_parquet import allocate_draws
from uv_vae.splitting import GLOBAL_SITE_HASH, STRATEGIES, SplitConfig
from uv_vae.stats_cache import load_or_compute_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-paths", nargs="+", required=True)
    parser.add_argument("--feature-spec-path", default="ml_features.json")
    parser.add_argument(
        "--row-filter", default="st = 'MIXED' AND et = 'MIXED' AND FILT = 1"
    )
    parser.add_argument("--stats-cache-path", default=None)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--epoch-shards", type=int, default=1)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-strategy", default=GLOBAL_SITE_HASH, choices=list(STRATEGIES))
    parser.add_argument(
        "--probe-row-groups",
        type=int,
        default=6,
        help="Random row groups per file to read when measuring the REALISED "
        "validation share. Site hashing is uniform over sites, not rows, and site "
        "depth varies (~8-9 reads, max in the thousands), so the row fraction is "
        "not the requested val_fraction. 0 skips the probe.",
    )
    parser.add_argument("--json-out", default=None)
    return parser.parse_args()


def probe_val_fraction(
    path: str, sample_id: str, split, row_filter: str, n_groups: int, seed: int = 0
) -> dict:
    """Measure the realised validation row/site share on a sample of row groups."""
    import numpy as np
    import polars as pl
    import pyarrow.parquet as pq

    from uv_vae.multi_parquet import columns_referenced
    from uv_vae.splitting import SITE_KEY_COLUMNS, val_mask

    handle = pq.ParquetFile(path, pre_buffer=False)
    available = set(handle.schema_arrow.names)
    columns = sorted(
        columns_referenced(row_filter, available)
        | {c for c in SITE_KEY_COLUMNS if c in available}
    )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(
        handle.metadata.num_row_groups,
        size=min(n_groups, handle.metadata.num_row_groups),
        replace=False,
    )

    frames = []
    positions = []
    starts = np.concatenate(
        [[0], np.cumsum([handle.metadata.row_group(i).num_rows
                         for i in range(handle.metadata.num_row_groups)])]
    )
    for group in chosen:
        table = handle.read_row_group(int(group), columns=columns)
        frame = pl.from_arrow(table).with_columns(
            pl.Series("__row_position",
                      np.arange(starts[group], starts[group] + table.num_rows, dtype=np.int64))
        )
        if row_filter:
            frame = pl.SQLContext(rows=frame).execute(
                f"SELECT * FROM rows WHERE {row_filter}", eager=True
            )
        if frame.height:
            frames.append(frame)
            positions.append(frame.get_column("__row_position").to_numpy())
    handle.close()

    if not frames:
        return {"sample_id": sample_id, "rows_probed": 0}

    frame = pl.concat(frames)
    mask = val_mask(split, sample_id, frame=frame, row_positions=np.concatenate(positions))
    result = {
        "sample_id": sample_id,
        "rows_probed": frame.height,
        "val_row_fraction": float(mask.mean()),
    }
    if all(column in frame.columns for column in SITE_KEY_COLUMNS):
        per_site = (
            frame.with_columns(pl.Series("__val", mask))
            .group_by(list(SITE_KEY_COLUMNS))
            .agg(pl.col("__val").mean().alias("f"))
        )
        fractions = per_site.get_column("f").to_numpy()
        result["sites_probed"] = int(per_site.height)
        result["val_site_fraction"] = float((fractions == 1.0).mean())
        result["straddling_site_fraction"] = float(((fractions > 0) & (fractions < 1)).mean())
    return result


def main() -> int:
    args = parse_args()
    from uv_vae.multi_streaming import resolve_parquet_paths

    paths = resolve_parquet_paths(args.parquet_paths)
    specs = load_feature_specs(args.feature_spec_path)
    split = SplitConfig(
        strategy=args.split_strategy, val_fraction=args.val_fraction, seed=42
    )

    stats = load_or_compute_stats(
        paths=paths,
        feature_specs=specs,
        row_filter=args.row_filter,
        cache_path=args.stats_cache_path,
        threads=args.threads,
    )

    counts = np.array([fs.rows for fs in stats.per_file], dtype=np.float64)
    weights = counts / counts.sum()
    draws = allocate_draws(weights, args.batch_size)

    print()
    print(f"{'sample':<32} {'rows':>15} {'weight':>9} {'draws':>7} {'val rows':>13}")
    print("-" * 80)
    for file_stats, weight, draw in zip(stats.per_file, weights, draws, strict=True):
        print(
            f"{file_stats.sample_id:<32} {file_stats.rows:>15,} {weight:>9.5f} "
            f"{draw:>7,} {int(file_stats.rows * split.val_fraction):>13,}"
        )
    print("-" * 80)
    print(f"{'TOTAL':<32} {stats.total_rows:>15,} {weights.sum():>9.5f} {draws.sum():>7,}")
    print()
    print(f"samples                : {len(paths)}")
    print(f"row filter             : {args.row_filter}")
    print(f"split                  : {split.strategy} @ val_fraction={split.val_fraction}")
    print(f"batch size             : {args.batch_size:,}")
    print(f"epoch shards           : {args.epoch_shards}")
    print(
        f"train rows per epoch   : "
        f"{int(stats.total_rows * (1 - split.val_fraction) / max(1, args.epoch_shards)):,}"
    )
    print(f"features kept          : {len(stats.categorical_specs)} categorical, "
          f"{len(stats.numeric_specs)} numeric")
    print(f"dropped (all-null)     : {len(stats.dropped_all_null_features)} "
          f"{stats.dropped_all_null_features}")

    probes: list[dict] = []
    if args.probe_row_groups > 0:
        print()
        print(
            f"realised split, probed on {args.probe_row_groups} random row groups per file:"
        )
        print(
            f"  {'sample':<30}{'rows':>12}{'val rows %':>12}{'val sites %':>13}{'straddling %':>14}"
        )
        for file_stats in stats.per_file:
            probe = probe_val_fraction(
                file_stats.path,
                file_stats.sample_id,
                split,
                args.row_filter,
                args.probe_row_groups,
            )
            probes.append(probe)
            if not probe.get("rows_probed"):
                print(f"  {probe['sample_id']:<30}{'no rows':>12}")
                continue
            print(
                f"  {probe['sample_id']:<30}{probe['rows_probed']:>12,}"
                f"{probe['val_row_fraction'] * 100:>12.3f}"
                f"{probe.get('val_site_fraction', float('nan')) * 100:>13.3f}"
                f"{probe.get('straddling_site_fraction', float('nan')) * 100:>14.3f}"
            )
        print(
            f"  (requested val_fraction={split.val_fraction}. Site hashing is uniform over "
            f"SITES;\n   the ROW share differs because site depth varies. Straddling sites "
            f"should be 0\n   for a site strategy -- anything else means validation leaks.)"
        )

    problems: list[str] = []
    for probe in probes:
        if probe.get("straddling_site_fraction", 0.0) > 0.001 and split.needs_site_columns:
            problems.append(
                f"{probe['sample_id']}: "
                f"{probe['straddling_site_fraction'] * 100:.2f}% of sites straddle "
                f"train/val under a site strategy -- validation is leaking"
            )
        realised = probe.get("val_row_fraction")
        if realised is not None and abs(realised - split.val_fraction) > 0.3 * split.val_fraction:
            problems.append(
                f"{probe['sample_id']}: realised validation row share "
                f"{realised * 100:.2f}% is far from the requested "
                f"{split.val_fraction * 100:.2f}%"
            )
    if int(draws.min()) < 1:
        starved = [
            fs.sample_id for fs, d in zip(stats.per_file, draws, strict=True) if d < 1
        ]
        problems.append(
            f"{len(starved)} sample(s) get zero draws at batch_size={args.batch_size:,}: "
            f"{starved[:5]}. Raise --batch-size."
        )
    if not stats.numeric_specs and not stats.categorical_specs:
        problems.append("no usable features remain after dropping all-null columns")
    smallest = min(weights)
    if smallest * args.batch_size < 5:
        problems.append(
            f"the smallest sample contributes only {smallest * args.batch_size:.1f} rows "
            f"per batch; consider a larger --batch-size for a stable per-sample signal"
        )

    print()
    if problems:
        for problem in problems:
            print(f"WARNING: {problem}", file=sys.stderr)
    else:
        print("OK: every sample parses, contributes rows, and gets a share of every batch.")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "parquet_paths": paths,
                    "row_filter": args.row_filter,
                    "total_rows": stats.total_rows,
                    "rows_by_sample": stats.rows_by_sample,
                    "weights": {
                        fs.sample_id: float(w)
                        for fs, w in zip(stats.per_file, weights, strict=True)
                    },
                    "draws_per_batch": {
                        fs.sample_id: int(d)
                        for fs, d in zip(stats.per_file, draws, strict=True)
                    },
                    "split": split.as_dict(),
                    "realised_split_probe": probes,
                    "active_categorical_features": [s.name for s in stats.categorical_specs],
                    "active_numeric_features": [s.name for s in stats.numeric_specs],
                    "dropped_all_null_features": stats.dropped_all_null_features,
                    "warnings": problems,
                },
                indent=2,
            )
        )
        print(f"wrote {args.json_out}")

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
