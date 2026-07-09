from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch

from uv_vae.data import connect_duckdb, stream_parquet_batches
from uv_vae.features import FeatureSpec, load_feature_specs
from uv_vae.model import TabularVAE, VAEConfig
from uv_vae.preprocess import transform_frame


def resolve_existing_path(path_str: str, checkpoint_path: Path) -> Path:
    raw_path = Path(path_str)
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append(Path.cwd() / raw_path)
        if len(checkpoint_path.resolve().parents) >= 3:
            candidates.append(checkpoint_path.resolve().parents[2] / raw_path)
        candidates.append(raw_path)

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve() if candidate.exists() else candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Unable to resolve path {path_str!r} relative to checkpoint {checkpoint_path}")


def resolve_device(device: str | None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for inference but is not available")
    return torch.device(device)


def unique_columns(columns: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for column in columns:
        if column in seen:
            continue
        seen.add(column)
        result.append(column)
    return result


@dataclass(frozen=True)
class EmbeddingResult:
    output_path: str
    rows_written: int
    latent_dim: int
    device: str
    parquet_path: str
    id_columns: list[str]


class LatentInference:
    def __init__(
        self,
        model: TabularVAE,
        checkpoint_path: Path,
        parquet_path: Path,
        categorical_specs: list[FeatureSpec],
        numeric_specs: list[FeatureSpec],
        numeric_means: dict[str, float],
        numeric_stds: dict[str, float],
        device: torch.device,
    ) -> None:
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.default_parquet_path = parquet_path
        self.categorical_specs = categorical_specs
        self.numeric_specs = numeric_specs
        self.numeric_means = numeric_means
        self.numeric_stds = numeric_stds
        self.device = device
        self.feature_names = [spec.name for spec in categorical_specs + numeric_specs]

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        feature_spec_path: str | None = None,
        parquet_path: str | None = None,
        device: str | None = None,
    ) -> "LatentInference":
        checkpoint_file = Path(checkpoint_path).resolve()
        payload = torch.load(checkpoint_file, map_location="cpu", weights_only=False)

        model = TabularVAE(VAEConfig(**payload["model_config"]))
        model.load_state_dict(payload["model_state_dict"])

        target_device = resolve_device(device)
        model.to(target_device)
        model.eval()

        checkpoint_feature_path = feature_spec_path or payload["training_config"]["feature_spec_path"]
        checkpoint_parquet_path = parquet_path or payload["training_config"]["parquet_path"]
        resolved_feature_path = resolve_existing_path(checkpoint_feature_path, checkpoint_file)
        resolved_parquet_path = resolve_existing_path(checkpoint_parquet_path, checkpoint_file)

        specs = load_feature_specs(resolved_feature_path)
        spec_map = {spec.name: spec for spec in specs}
        feature_report = payload["feature_report"]
        preprocess_report = payload["preprocess_report"]
        categorical_specs = [
            spec_map[name] for name in feature_report["active_categorical_features"]
        ]
        numeric_specs = [
            spec_map[name] for name in feature_report["active_numeric_features"]
        ]

        return cls(
            model=model,
            checkpoint_path=checkpoint_file,
            parquet_path=resolved_parquet_path,
            categorical_specs=categorical_specs,
            numeric_specs=numeric_specs,
            numeric_means=preprocess_report["numeric_means"],
            numeric_stds=preprocess_report["numeric_stds"],
            device=target_device,
        )

    @property
    def latent_dim(self) -> int:
        return self.model.config.latent_dim

    def encode_frame(self, frame: pl.DataFrame, batch_size: int) -> np.ndarray:
        categorical_inputs, numeric_inputs = transform_frame(
            frame=frame,
            categorical_specs=self.categorical_specs,
            numeric_specs=self.numeric_specs,
            numeric_means=self.numeric_means,
            numeric_stds=self.numeric_stds,
        )

        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, frame.height, batch_size):
                stop = start + batch_size
                batch_cat = categorical_inputs[start:stop].to(self.device, non_blocking=True)
                batch_num = numeric_inputs[start:stop].to(self.device, non_blocking=True)
                mu, _ = self.model.encode(batch_cat, batch_num)
                chunks.append(mu.detach().cpu().numpy().astype(np.float32, copy=False))

        if not chunks:
            return np.zeros((0, self.latent_dim), dtype=np.float32)
        return np.concatenate(chunks, axis=0)

    def embed_parquet(
        self,
        output_path: str | Path,
        parquet_path: str | Path | None = None,
        id_columns: list[str] | None = None,
        batch_size: int = 4096,
        scan_batch_rows: int = 100_000,
        threads: int | None = None,
        where: str | None = None,
        limit: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> EmbeddingResult:
        source_path = Path(parquet_path).resolve() if parquet_path else self.default_parquet_path
        target_path = Path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        selected_id_columns = id_columns or []
        selected_columns = unique_columns(selected_id_columns + self.feature_names)

        writer: pq.ParquetWriter | None = None
        rows_written = 0
        batch_index = 0
        try:
            with connect_duckdb(threads=threads) as conn:
                reader = stream_parquet_batches(
                    conn=conn,
                    parquet_path=source_path,
                    select_columns=selected_columns,
                    rows_per_batch=scan_batch_rows,
                    where=where,
                    limit=limit,
                )
                for record_batch in reader:
                    frame = pl.from_arrow(record_batch)
                    features_frame = frame.select(self.feature_names)
                    embeddings = self.encode_frame(features_frame, batch_size=batch_size)

                    output_frames = [
                        pl.DataFrame(
                            {
                                "row_index": np.arange(
                                    rows_written,
                                    rows_written + frame.height,
                                    dtype=np.int64,
                                )
                            }
                        )
                    ]
                    if selected_id_columns:
                        output_frames.append(frame.select(selected_id_columns))
                    output_frames.append(
                        pl.DataFrame(
                            {
                                f"latent_{index}": embeddings[:, index]
                                for index in range(self.latent_dim)
                            }
                        )
                    )
                    output_frame = pl.concat(output_frames, how="horizontal")
                    table = output_frame.to_arrow()
                    if writer is None:
                        writer = pq.ParquetWriter(target_path, table.schema)
                    writer.write_table(table)
                    rows_written += frame.height
                    batch_index += 1
                    if progress_callback is not None:
                        progress_callback(rows_written, batch_index)
        finally:
            if writer is not None:
                writer.close()

        if rows_written == 0:
            raise RuntimeError("No rows were written during inference")

        return EmbeddingResult(
            output_path=str(target_path),
            rows_written=rows_written,
            latent_dim=self.latent_dim,
            device=str(self.device),
            parquet_path=str(source_path),
            id_columns=selected_id_columns,
        )
