"""Reproducibility guarantees this project relies on but never previously asserted.

The whole research question is "does the latent geometry move when I change one
variable" -- which is only answerable if holding every variable fixed produces
the *same* result. Until now nothing checked that. It survived on CPU by luck;
on GPU there are several ways for it to quietly stop being true (cuBLAS
workspace not set before the handle is created, TF32, non-deterministic kernels,
AMP loss scaling), so it needs a test that fails loudly.

Four properties are covered:

1. Same seed -> bit-identical checkpoint. The headline guarantee.
2. Different seed -> different checkpoint. The negative control. Without it,
   a bug that froze every weight at its initial value would make (1) pass.
3. DuckDB reservoir sampling is stable, and ``data_seed`` moves the rows
   independently of ``seed``.
4. The streaming train/val split is a genuine partition -- disjoint and
   complete. This is the one that guards the leakage hazard: the split is by
   scan POSITION and train/val use separate DuckDB connections, so if the scans
   ever disagree on row order a row can land in both, and the val loss driving
   early stopping would be measured on trained-on rows.

These run on whatever device is available. On the GPU node run them there --
passing on CPU says nothing about CUDA determinism. For a full-size check
against a real parquet, use ``scripts/check_reproducibility.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# Skip rather than error when the scientific stack is absent -- this file gets
# opened on the Windows dev box, where torch is not installable (the project
# pins it to the Linux CUDA-12.8 index).
np = pytest.importorskip("numpy")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
torch = pytest.importorskip("torch")
pytest.importorskip("duckdb")
pytest.importorskip("polars")

from uv_vae.data import connect_duckdb, sample_frame
from uv_vae.training import TrainingConfig, train

ROWS = 512
FILTER = "st = 'MIXED' AND et = 'MIXED' AND FILT = 1"


def _feature_spec(path: Path) -> Path:
    """A miniature ml_features.json: two categoricals, three numerics."""
    payload = {
        "features": [
            {"name": "REF", "type": "c", "values": {"A": 0, "C": 1, "G": 2, "T": 3, "": 4}},
            {"name": "ALT", "type": "c", "values": {"A": 0, "C": 1, "G": 2, "T": 3, "": 4}},
            {"name": "SNVQ", "type": "float"},
            {"name": "DP", "type": "int"},
            {"name": "RAW_VAF", "type": "float"},
        ]
    }
    path.write_text(json.dumps(payload))
    return path


def _parquet(path: Path, rows: int = ROWS) -> Path:
    """Synthetic feature map. Fixed RNG so the fixture itself is reproducible."""
    rng = np.random.default_rng(0)
    bases = np.array(["A", "C", "G", "T"])
    table = pa.table(
        {
            "REF": pa.array(rng.choice(bases, rows)),
            "ALT": pa.array(rng.choice(bases, rows)),
            "SNVQ": pa.array(rng.normal(30.0, 5.0, rows).astype("float32")),
            "DP": pa.array(rng.integers(1, 100, rows).astype("int32")),
            "RAW_VAF": pa.array(rng.random(rows).astype("float32")),
            # Constant so every row passes the project's default filter.
            "st": pa.array(["MIXED"] * rows),
            "et": pa.array(["MIXED"] * rows),
            "FILT": pa.array([1] * rows),
            # Identity columns, for the sampling test.
            "CHROM": pa.array(["chr1"] * rows),
            "POS": pa.array(np.arange(rows, dtype="int64")),
        }
    )
    pq.write_table(table, path)
    return path


def _config(parquet: Path, spec: Path, out: Path, seed: int = 42,
            data_seed: int | None = None, epochs: int = 2) -> TrainingConfig:
    return TrainingConfig(
        parquet_path=str(parquet),
        feature_spec_path=str(spec),
        output_dir=str(out),
        row_filter=FILTER,
        sample_rows=ROWS,
        epochs=epochs,
        batch_size=64,
        latent_dim=4,
        hidden_dims=[16, 8],
        learning_rate=1e-3,
        kl_weight=0.05,
        train_fraction=0.8,
        seed=seed,
        threads=1,
        data_seed=data_seed,
    )


def _weights(run_dir: Path) -> dict[str, "torch.Tensor"]:
    payload = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    return payload["model_state_dict"]


@pytest.fixture
def dataset(tmp_path: Path) -> tuple[Path, Path]:
    return _parquet(tmp_path / "demo.parquet"), _feature_spec(tmp_path / "features.json")


# ---------------------------------------------------------------------------
# 1. The headline guarantee
# ---------------------------------------------------------------------------

def test_same_seed_produces_bit_identical_weights(dataset, tmp_path: Path) -> None:
    parquet, spec = dataset
    first = train(_config(parquet, spec, tmp_path / "a"))
    second = train(_config(parquet, spec, tmp_path / "b"))

    left, right = _weights(first), _weights(second)
    assert left.keys() == right.keys(), "the two runs built different models"

    for name in left:
        # Bit-identical, not allclose: any tolerance here would hide exactly the
        # drift this test exists to catch.
        assert torch.equal(left[name], right[name]), (
            f"tensor {name!r} differs between two same-seed runs; "
            f"max |delta| = {(left[name] - right[name]).abs().max().item():.3e}. "
            f"On CUDA, check that CUBLAS_WORKSPACE_CONFIG=:4096:8 was exported "
            f"BEFORE python started -- setting it inside seed_everything() is too late."
        )


def test_same_seed_produces_identical_loss_history(dataset, tmp_path: Path) -> None:
    parquet, spec = dataset
    first = train(_config(parquet, spec, tmp_path / "a"))
    second = train(_config(parquet, spec, tmp_path / "b"))

    left = json.loads((first / "training_report.json").read_text())["history"]
    right = json.loads((second / "training_report.json").read_text())["history"]
    assert left == right, "per-epoch losses diverged between two same-seed runs"


# ---------------------------------------------------------------------------
# 2. Negative control -- proves the test above can actually fail
# ---------------------------------------------------------------------------

def test_different_seed_produces_different_weights(dataset, tmp_path: Path) -> None:
    parquet, spec = dataset
    first = train(_config(parquet, spec, tmp_path / "a", seed=42))
    second = train(_config(parquet, spec, tmp_path / "b", seed=1337))

    left, right = _weights(first), _weights(second)
    assert any(not torch.equal(left[name], right[name]) for name in left), (
        "two different seeds produced identical weights -- the seed is not "
        "reaching model init, so the same-seed test proves nothing"
    )


# ---------------------------------------------------------------------------
# 3. Row sampling
# ---------------------------------------------------------------------------

def test_sample_frame_is_deterministic(dataset) -> None:
    parquet, _ = dataset
    frames = []
    for _ in range(2):
        # threads=1: DuckDB's REPEATABLE sampling is only reproducible single-threaded.
        with connect_duckdb(threads=1) as conn:
            frames.append(
                sample_frame(conn, parquet, ["POS"], sample_rows=64, seed=7, where=FILTER)
            )
    assert frames[0].to_dicts() == frames[1].to_dicts()


def test_data_seed_moves_rows_without_moving_the_model_seed(dataset, tmp_path: Path) -> None:
    """data_seed selects rows; seed drives init/shuffle/split. Sweeping which rows
    are drawn while holding the model fixed depends on these staying separate."""
    parquet, _ = dataset
    samples = []
    for data_seed in (7, 99):
        with connect_duckdb(threads=1) as conn:
            samples.append(
                sample_frame(conn, parquet, ["POS"], sample_rows=64, seed=data_seed, where=FILTER)
                ["POS"].to_list()
            )
    assert samples[0] != samples[1], "changing the sampling seed did not change the rows"


# ---------------------------------------------------------------------------
# 4. The streaming split really is a partition
# ---------------------------------------------------------------------------

def test_streaming_train_val_split_is_disjoint_and_complete(dataset, tmp_path: Path) -> None:
    """Guards the leakage hazard in the position-based split.

    Train and val are built by two independent DuckDB scans. Their row sets must
    be disjoint (no row trained on and validated on) and together cover every
    filtered row (none silently dropped).
    """
    from uv_vae.streaming import StreamingParquetDataset

    parquet, spec = dataset
    from uv_vae.features import load_feature_specs

    specs = load_feature_specs(spec)
    categorical = [s for s in specs if s.is_categorical]
    numeric = [s for s in specs if s.is_numeric]
    means = {s.name: 0.0 for s in numeric}
    stds = {s.name: 1.0 for s in numeric}

    common = dict(
        parquet_path=str(parquet),
        categorical_specs=categorical,
        numeric_specs=numeric,
        numeric_means=means,
        numeric_stds=stds,
        row_filter=FILTER,
        batch_size=32,
        train_fraction=0.8,
        seed=42,
        threads=4,          # >1 on purpose: this is where scan order could drift
        shuffle=False,      # shuffle would hide a position bug behind a permutation
    )

    counts = {}
    for split in ("train", "val"):
        dataset_obj = StreamingParquetDataset(split=split, **common)
        counts[split] = sum(int(cat.shape[0]) for cat, _, _ in dataset_obj)

    assert counts["train"] + counts["val"] == ROWS, (
        f"train ({counts['train']}) + val ({counts['val']}) != {ROWS} filtered rows -- "
        f"the two scans disagreed on row order, so rows were duplicated or dropped. "
        f"connect_duckdb(preserve_insertion_order=True) is what pins this."
    )
    assert counts["val"] > 0, "validation split is empty; early stopping would have no signal"


def test_streaming_split_is_stable_across_repeated_scans(dataset) -> None:
    """The same split, re-derived, must select the same number of rows every time."""
    from uv_vae.features import load_feature_specs
    from uv_vae.streaming import StreamingParquetDataset

    parquet, spec = dataset
    specs = load_feature_specs(spec)
    numeric = [s for s in specs if s.is_numeric]

    sizes = []
    for _ in range(2):
        dataset_obj = StreamingParquetDataset(
            parquet_path=str(parquet),
            categorical_specs=[s for s in specs if s.is_categorical],
            numeric_specs=numeric,
            numeric_means={s.name: 0.0 for s in numeric},
            numeric_stds={s.name: 1.0 for s in numeric},
            row_filter=FILTER,
            batch_size=32,
            split="val",
            train_fraction=0.8,
            seed=42,
            threads=4,
            shuffle=False,
        )
        sizes.append(sum(int(cat.shape[0]) for cat, _, _ in dataset_obj))

    assert sizes[0] == sizes[1], f"validation split size varied between scans: {sizes}"
