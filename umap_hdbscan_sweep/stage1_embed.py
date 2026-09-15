"""Stage 1 -- encode every deduplicated locus through the VAE into one latent array.

Output is a ``.npy`` written with ``open_memmap`` rather than a parquet of floats, because
stage 2 fits UMAP on the whole thing: a memmap lets cuML page it to the GPU without first
materialising 100M+ rows as Python objects, and it can be re-opened read-only by several
sweep processes at once.

Row order is the contract between the two files this writes:

    latent.npy      (N, latent_dim) float32 -- posterior means (mu), no sampling
    context.parquet (N rows)                -- CHROM, POS, REF, ALT, X_PREV1, X_NEXT1, locus_reads

Row *i* of the array is row *i* of the parquet. Everything downstream (cluster labels,
SigProfiler's trinucleotide counts) joins on that positional correspondence, so the
chromosome parts are always consumed in the manifest's order and never reordered.

mu is used rather than a sample from q(z|x) so the embedding is deterministic -- the same
choice ``LatentInference`` and every earlier clustering run already make.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

# Walk up to the directory that holds uv_vae/ rather than counting levels: these scripts
# get reorganised into subfolders (umap/, hdbscan/, tests/), and a hard-coded parents[N]
# silently resolves to the wrong root the moment one moves, failing on `import uv_vae`.
REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "uv_vae").is_dir()),
                 Path(__file__).resolve().parents[1])
for candidate in (REPO_ROOT / "uv_vae", REPO_ROOT, Path(__file__).resolve().parent):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np
import polars as pl
import pyarrow.parquet as pq

import sweep_core
from uv_vae.inference import LatentInference

CONTEXT_COLUMNS = ["CHROM", "POS", "REF", "ALT", "X_PREV1", "X_NEXT1", "locus_reads"]


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def dedup_parts(manifest_path: Path) -> list[Path]:
    """Chromosome parts in manifest order -- this order defines the row index downstream."""
    manifest = json.loads(manifest_path.read_text())
    return [Path(entry["path"]) for entry in manifest["per_chromosome"]]


def iter_row_group_frames(path: Path, columns: list[str]) -> tuple[int, object]:
    """Yield one polars frame per parquet row group, so peak memory is one group."""
    parquet_file = pq.ParquetFile(path)

    def generate():
        for group_index in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(group_index, columns=columns)
            yield pl.from_arrow(table)

    return parquet_file.metadata.num_rows, generate()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode deduplicated loci to VAE latents")
    parser.add_argument("--dedup-manifest", required=True, help="dedup_manifest.json from stage 0")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--feature-spec-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=262_144)
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--gpu-budget-gb", type=float, default=None,
                        help="GPU ceiling for THIS process; must leave room for any "
                             "trainer sharing the card.")
    return parser.parse_args()


def main() -> int:
    started = perf_counter()
    args = parse_args()

    manifest_path = Path(args.dedup_manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    parts = dedup_parts(manifest_path)
    total_rows = int(manifest["total_loci"])
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Encoding {total_rows:,} deduplicated loci from {len(parts)} chromosome parts")

    # Before the model is built, so torch's cap is in place for its first allocation.
    # Nothing here touches cuML, so the whole slice goes to torch (STAGE_RMM_SHARE).
    budget_report = None
    if args.device != "cpu":
        budget_report = sweep_core.apply_gpu_budget("embed", budget_gb=args.gpu_budget_gb)

    inference = LatentInference.from_checkpoint(
        checkpoint_path=str(Path(args.checkpoint_path).resolve()),
        feature_spec_path=args.feature_spec_path,
        parquet_path=str(parts[0]),
        device=args.device,
    )
    feature_names = list(inference.feature_names)
    available = set(pq.read_schema(parts[0]).names)
    context_columns = [name for name in CONTEXT_COLUMNS if name in available]
    read_columns = list(dict.fromkeys([*feature_names, *context_columns]))

    latent_path = output_dir / "latent.npy"
    context_path = output_dir / "context.parquet"
    latent_memmap: np.ndarray | None = None
    context_batches: list[pl.DataFrame] = []
    written = 0

    for part_index, part in enumerate(parts, start=1):
        part_started = perf_counter()
        part_rows, frames = iter_row_group_frames(part, read_columns)
        for frame in frames:
            latent = inference.encode_frame(frame.select(feature_names), batch_size=args.batch_size)

            if latent_memmap is None:
                latent_dim = int(latent.shape[1])
                # open_memmap writes a real .npy header, so the file is loadable with
                # np.load(mmap_mode="r") without any sidecar shape metadata.
                latent_memmap = np.lib.format.open_memmap(
                    latent_path, mode="w+", dtype=np.float32, shape=(total_rows, latent_dim)
                )
                log(f"Allocated {latent_path.name}: ({total_rows:,}, {latent_dim}) float32 "
                    f"= {total_rows * latent_dim * 4 / 1e9:.1f} GB")

            rows = latent.shape[0]
            if written + rows > total_rows:
                raise SystemExit(
                    f"stage 0 manifest claims {total_rows:,} loci but the parts hold more "
                    f"(at least {written + rows:,}) -- re-run stage 0, the manifest is stale"
                )
            latent_memmap[written:written + rows] = latent.astype(np.float32, copy=False)
            context_batches.append(frame.select(context_columns))
            written += rows

        log(f"  [{part_index}/{len(parts)}] {part.name}: {part_rows:,} rows "
            f"({perf_counter() - part_started:.0f}s), {written:,}/{total_rows:,} total")

    if latent_memmap is None:
        raise SystemExit("no rows encoded -- the dedup parts are empty")
    if written != total_rows:
        raise SystemExit(f"encoded {written:,} rows but the manifest claims {total_rows:,}")

    latent_memmap.flush()
    pl.concat(context_batches, how="vertical").write_parquet(context_path)

    summary = {
        "latent_path": str(latent_path),
        "context_path": str(context_path),
        "rows": written,
        "latent_dim": int(latent_memmap.shape[1]),
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "dedup_manifest": str(manifest_path),
        "feature_names": feature_names,
        "context_columns": context_columns,
        "device": args.device,
        "gpu_budget": budget_report,
        "seconds": round(perf_counter() - started, 1),
    }
    (output_dir / "embed_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"Wrote {written:,} x {summary['latent_dim']} latents in {summary['seconds']}s -> {latent_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
