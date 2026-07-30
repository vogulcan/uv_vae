"""Parity gate on the cuDF decode path.

The GPU decode path in ``multi_parquet`` is only safe to enable if it is
**bit-identical** to the CPU one, for two separate reasons:

* the encoded arrays are pinned against ``preprocess.encode_categorical_column``
  by ``test_encoder_matches_preprocess_helpers``, so a divergence there breaks
  comparability with every single-file run;
* the split hash decides which rows are validation rows. A one-bit difference
  silently re-partitions the data, and a run trained on a different 90% is not
  comparable to any earlier result -- and nothing would report an error.

Every test here skips when cuDF/cuPy are absent, so the suite still passes on a
CPU-only machine. That means **these tests must be run on the GPU host before
UV_VAE_GPU_DECODE=1 is used for real work** -- a green suite on a laptop says
nothing about the GPU path.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pytest

from uv_vae.features import FeatureSpec
from uv_vae.multi_parquet import (
    ParquetSampleReader,
    RowEncoder,
    SampleSource,
)
from uv_vae.splitting import SplitConfig, split_mask

from uv_vae import gpu_decode

# Only the parity tests need a device. The filter grammar is pure Python and is
# where a silently under-filtering translation would be caught, so it must run on
# every machine, not just the GPU host.
needs_gpu = pytest.mark.skipif(
    not gpu_decode.gpu_decode_available(),
    reason="cudf/cupy not importable; GPU decode parity cannot be checked here",
)

FILTER = "st = 'MIXED' AND et = 'MIXED' AND FILT = 1"


def _specs() -> tuple[list[FeatureSpec], list[FeatureSpec]]:
    categorical = [
        FeatureSpec(name="REF", kind="c", values={"A": 1, "C": 2, "G": 3, "T": 4, "NA": 0}),
        FeatureSpec(name="ALT", kind="c", values={"A": 1, "C": 2, "G": 3, "T": 4, "NA": 0}),
    ]
    numeric = [
        FeatureSpec(name="SNVQ", kind="float"),
        FeatureSpec(name="RAW_VAF", kind="float"),
    ]
    return categorical, numeric


@pytest.fixture
def dataset(tmp_path):
    """A parquet with several row groups, nulls, and rows on both filter sides."""
    rng = np.random.default_rng(20260730)
    n = 6000
    bases = np.array(["A", "C", "G", "T"])
    snvq = rng.normal(30.0, 5.0, n).astype(np.float32)
    snvq[rng.random(n) < 0.1] = np.nan  # nulls must survive the mask path
    frame = pl.DataFrame(
        {
            "CHROM": rng.choice([f"chr{i}" for i in range(1, 23)], n),
            "POS": rng.integers(1, 250_000_000, n).astype(np.int64),
            "REF": rng.choice(bases, n),
            "ALT": rng.choice(bases, n),
            "SNVQ": snvq,
            "RAW_VAF": rng.random(n).astype(np.float32),
            "st": rng.choice(["MIXED", "PLUS"], n, p=[0.8, 0.2]),
            "et": rng.choice(["MIXED", "MINUS"], n, p=[0.8, 0.2]),
            "FILT": rng.choice([1, 0], n, p=[0.75, 0.25]).astype(np.int32),
        }
    )
    path = tmp_path / "sample.parquet"
    # Several row groups, so row-group addressing is actually exercised.
    pq.write_table(frame.to_arrow(), path, row_group_size=750)
    return path


def _reader(path, split, split_config, use_gpu):
    categorical, numeric = _specs()
    encoder = RowEncoder(
        categorical_specs=categorical,
        numeric_specs=numeric,
        numeric_means={s.name: 0.5 for s in numeric},
        numeric_stds={s.name: 2.0 for s in numeric},
    )
    return ParquetSampleReader(
        source=SampleSource(sample_id="sample", path=str(path), rows=1000),
        encoder=encoder,
        split_config=split_config,
        split=split,
        row_filter=FILTER,
        seed=42,
        use_gpu=use_gpu,
    )


@needs_gpu
@pytest.mark.parametrize("split", ["train", "val"])
@pytest.mark.parametrize(
    "strategy", ["global_site_hash", "per_sample_site_hash", "per_sample_row_hash"]
)
def test_gpu_decode_is_bit_identical_to_cpu(dataset, split, strategy) -> None:
    """Same row group, both paths, exact equality of all three arrays."""
    config = SplitConfig(strategy=strategy, val_fraction=0.1, seed=42)
    cpu = _reader(dataset, split, config, use_gpu=False)
    gpu = _reader(dataset, split, config, use_gpu=True)
    assert gpu._gpu is not None, "GPU reader silently fell back to the CPU path"

    compared = 0
    for group in range(cpu.num_row_groups):
        want = cpu._decode_group(group)
        got = gpu._decode_group(group)
        if want is None:
            assert got is None, f"group {group}: CPU dropped it, GPU did not"
            continue
        assert got is not None, f"group {group}: GPU dropped it, CPU did not"
        compared += 1

        # Row COUNT first: a filter or split divergence shows up here, and an
        # equality assert on mismatched shapes reports confusingly.
        assert want[0].shape == got[0].shape, f"group {group}: categorical shape"
        assert want[1].shape == got[1].shape, f"group {group}: numeric shape"

        np.testing.assert_array_equal(want[0], got[0], err_msg=f"group {group} cat")
        # Exact, not approximate: subtract-then-divide in float32 must not be
        # contracted into a differently-rounded fused op on the device.
        np.testing.assert_array_equal(want[1], got[1], err_msg=f"group {group} num")
        np.testing.assert_array_equal(want[2], got[2], err_msg=f"group {group} mask")

    assert compared > 0, "no row group survived the filter; the fixture proves nothing"


@needs_gpu
def test_gpu_split_mask_matches_cpu_exactly(dataset) -> None:
    """The split predicate alone, isolated from decode and encode."""
    config = SplitConfig(strategy="global_site_hash", val_fraction=0.1, seed=42)
    table = pq.read_table(dataset)
    frame = pl.from_arrow(table)

    cudf, cupy = gpu_decode.gpu_modules()
    gpu_frame = cudf.DataFrame.from_arrow(table)

    for split in ("train", "val"):
        want = split_mask(config, "sample", split, frame=frame)
        got = cupy.asnumpy(
            gpu_decode.gpu_split_mask(config, "sample", split, frame=gpu_frame)
        )
        np.testing.assert_array_equal(want, got, err_msg=f"{split} mask diverged")


def test_unsupported_filters_are_refused_not_approximated() -> None:
    """Anything outside the grammar must raise, never silently under-filter."""
    available = {"a", "b", "FILT", "st"}
    for sql in (
        "a = 1 OR b = 2",
        "(a = 1) AND b = 2",
        "a IN (1, 2)",
        "lower(st) = 'mixed'",
        "a + b = 3",
        "missing_column = 1",
    ):
        with pytest.raises(gpu_decode.UnsupportedFilter):
            gpu_decode.compile_filter(sql, available)


def test_supported_filter_grammar_compiles() -> None:
    available = {"st", "et", "FILT", "SNVQ"}
    for sql in (
        "st = 'MIXED' AND et = 'MIXED' AND FILT = 1",
        "FILT = 1",
        "SNVQ >= 20.5 AND SNVQ < 40",
        "st != 'PLUS'",
        "SNVQ IS NOT NULL",
    ):
        assert gpu_decode.compile_filter(sql, available) is not None
    assert gpu_decode.compile_filter(None, available) is None
    assert gpu_decode.compile_filter("   ", available) is None
