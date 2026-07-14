from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from uv_vae.pipeline_vae import run_pipeline_to_vae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the production-style pipeline up to the VAE embedding stage")
    parser.add_argument("--checkpoint-path", required=True, help="Path to the trained VAE checkpoint")
    parser.add_argument("--parquet-path", required=True, help="Path to the source parquet feature map")
    parser.add_argument("--row-filter", required=True, help="DuckDB SQL filter applied before deduplication and embedding")
    parser.add_argument("--output-dir", default="artifacts/vae_stage", help="Directory for intermediate and latent outputs")
    parser.add_argument("--sample-rows", type=int, default=None, help="Optional number of deduplicated rows to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--embed-batch-size", type=int, default=4096)
    parser.add_argument("--scan-batch-rows", type=int, default=100_000)
    parser.add_argument("--memory-limit", default=None)
    parser.add_argument("--feature-spec-path", default=None, help="Optional feature spec path override for checkpoint loading")
    parser.add_argument("--deduped-output-path", default=None)
    parser.add_argument("--latent-output-path", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_pipeline_to_vae(
        checkpoint_path=args.checkpoint_path,
        parquet_path=args.parquet_path,
        row_filter=args.row_filter,
        output_dir=args.output_dir,
        sample_rows=args.sample_rows,
        seed=args.seed,
        threads=args.threads,
        device=args.device,
        embed_batch_size=args.embed_batch_size,
        scan_batch_rows=args.scan_batch_rows,
        memory_limit=args.memory_limit,
        feature_spec_path=args.feature_spec_path,
        deduped_output_path=args.deduped_output_path,
        latent_output_path=args.latent_output_path,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
