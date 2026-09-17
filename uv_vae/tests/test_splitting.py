"""Tests for the content-hash train/val split.

Several of these pin behaviour that a plausible-looking implementation gets
wrong: XOR salting, float-rounded thresholds, and Python's per-process
randomised ``hash()``.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import polars as pl
import pytest

from uv_vae.splitting import (
    GLOBAL_SITE_HASH,
    PER_SAMPLE_ROW_HASH,
    PER_SAMPLE_SITE_HASH,
    SplitConfig,
    fnv1a64,
    mix_keys,
    splitmix64,
    stable_seed,
    val_mask,
)

GOLDEN_GAMMA = 0x9E3779B97F4A7C15


def make_sites(n_sites: int = 4000, depth: int = 9, seed: int = 0) -> pl.DataFrame:
    """A frame shaped like the real featuremap: many reads per locus, adjacent."""
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(np.arange(1, 500_000), size=n_sites, replace=False))
    depths = rng.integers(1, depth * 2, size=n_sites)
    return pl.DataFrame(
        {
            "CHROM": np.repeat(["chr1", "chr2"], [depths.sum() - depths[n_sites // 2 :].sum(),
                                                  depths[n_sites // 2 :].sum()]),
            "POS": np.repeat(positions, depths),
            "REF": np.repeat(rng.choice(list("ACGT"), n_sites), depths),
            "ALT": np.repeat(rng.choice(list("ACGT"), n_sites), depths),
        }
    )


# --- threshold arithmetic ---------------------------------------------------

def test_threshold_is_exact_not_float_rounded():
    # int(0.1 * 2**64) rounds the float product to ...955264. The split fraction
    # is unaffected, but the number must come from the decimal the user wrote,
    # not from a platform's float rounding.
    assert SplitConfig(val_fraction=0.1).threshold == 1844674407370955161
    assert int(0.1 * 2**64) == 1844674407370955264


def test_val_fraction_derived_by_subtraction_gives_the_same_split():
    # A shell runner computes 1 - 0.9; the CLI passes a literal 0.1. In float
    # those are different numbers, and left alone they partition the data
    # differently -- so a statistics stage and a training stage would disagree
    # about which rows are validation rows.
    assert 1.0 - 0.9 != 0.1
    assert SplitConfig(val_fraction=1.0 - 0.9).threshold == SplitConfig(val_fraction=0.1).threshold


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_invalid_val_fraction_rejected(bad):
    with pytest.raises(ValueError):
        SplitConfig(val_fraction=bad)


def test_invalid_strategy_rejected():
    with pytest.raises(ValueError, match="Unknown split strategy"):
        SplitConfig(strategy="nonsense")


# --- hash primitives --------------------------------------------------------

def test_splitmix64_matches_reference_vectors():
    # The published sequence for a SplitMix64 generator seeded at 0 is produced by
    # applying the finaliser to successive multiples of the golden gamma.
    state = np.array([0, GOLDEN_GAMMA, (2 * GOLDEN_GAMMA) % 2**64], dtype=np.uint64)
    expected = np.array(
        [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F], dtype=np.uint64
    )
    assert np.array_equal(splitmix64(state), expected)


def test_fnv1a64_known_values():
    assert fnv1a64("") == 0xCBF29CE484222325
    assert fnv1a64("a") == 0xAF63DC4C8601EC8C


def test_mix_keys_is_order_sensitive_and_deterministic():
    a = np.array([1, 2, 3], dtype=np.uint64)
    b = np.array([4, 5, 6], dtype=np.uint64)
    assert np.array_equal(mix_keys(a, b), mix_keys(a, b))
    assert not np.array_equal(mix_keys(a, b), mix_keys(b, a))


def test_stable_seed_does_not_use_python_hash():
    """PYTHONHASHSEED must not change the answer.

    Python's hash() is randomised per process, and ``training.seed_everything``
    sets PYTHONHASHSEED with ``os.environ.setdefault`` -- far too late to affect
    the running interpreter. Seeding a row-group shuffle with hash(filename)
    therefore gives a different order on every run.
    """
    script = textwrap.dedent(
        """
        from uv_vae.splitting import stable_seed
        print(stable_seed("s0.parquet", 42, 3), hash("s0.parquet"))
        """
    )
    outputs = []
    for hashseed in ("0", "1", "12345"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={**dict(__import__("os").environ), "PYTHONHASHSEED": hashseed},
            check=True,
        )
        outputs.append(result.stdout.split())

    assert len({parts[0] for parts in outputs}) == 1, "stable_seed changed across processes"
    # Sanity: Python's own hash really does vary, so the test above is not vacuous.
    assert len({parts[1] for parts in outputs}) > 1


# --- split behaviour --------------------------------------------------------

@pytest.mark.parametrize(
    "strategy", [PER_SAMPLE_ROW_HASH, PER_SAMPLE_SITE_HASH, GLOBAL_SITE_HASH]
)
def test_val_fraction_is_approximately_honoured(strategy):
    frame = make_sites(n_sites=6000)
    positions = np.arange(frame.height)
    config = SplitConfig(strategy=strategy, val_fraction=0.1, seed=42)
    mask = val_mask(config, "sampleA", frame=frame, row_positions=positions)
    # Site strategies are uniform over SITES; the ROW fraction inherits variance
    # from unequal site depth, so the tolerance is wider than for row hashing.
    assert 0.08 < mask.mean() < 0.12


def test_site_hash_keeps_every_read_at_a_locus_together():
    frame = make_sites()
    config = SplitConfig(strategy=PER_SAMPLE_SITE_HASH, val_fraction=0.2, seed=1)
    mask = val_mask(config, "sampleA", frame=frame)
    per_site = (
        frame.with_columns(pl.Series("v", mask))
        .group_by(["CHROM", "POS", "REF", "ALT"])
        .agg(pl.col("v").n_unique().alias("sides"))
    )
    assert per_site.get_column("sides").max() == 1


def test_row_hash_splits_loci_across_both_sides():
    """The documented trade-off of the cheap strategy, pinned so it stays visible."""
    frame = make_sites()
    config = SplitConfig(strategy=PER_SAMPLE_ROW_HASH, val_fraction=0.2, seed=1)
    mask = val_mask(config, "sampleA", frame=frame, row_positions=np.arange(frame.height))
    per_site = (
        frame.with_columns(pl.Series("v", mask))
        .group_by(["CHROM", "POS", "REF", "ALT"])
        .agg(pl.col("v").n_unique().alias("sides"))
    )
    assert per_site.get_column("sides").max() == 2


def test_salts_produce_independent_draws():
    """Two 10% draws must overlap on ~1% of rows, not 0% and not 10%.

    ``hash(k, A) XOR hash(k, B)`` is a constant offset independent of k, so an
    implementation that XORs the salt onto the hash OUTPUT gives draws that are
    structured across salts -- two salts whose offset has a high bit set produce
    mutually exclusive splits. Salting must change the hash INPUT.
    """
    frame = make_sites(n_sites=20_000)
    config = SplitConfig(strategy=PER_SAMPLE_SITE_HASH, val_fraction=0.1, seed=42)
    other_seed = SplitConfig(strategy=PER_SAMPLE_SITE_HASH, val_fraction=0.1, seed=43)

    a = val_mask(config, "sampleA", frame=frame)
    b = val_mask(config, "sampleB", frame=frame)
    c = val_mask(other_seed, "sampleA", frame=frame)

    for name, overlap in (("sample", (a & b).mean()), ("seed", (a & c).mean())):
        assert 0.005 < overlap < 0.017, f"{name} salt overlap {overlap:.4f} is not ~0.01"


def test_global_site_hash_ignores_sample_identity():
    frame = make_sites()
    config = SplitConfig(strategy=GLOBAL_SITE_HASH, val_fraction=0.1, seed=42)
    assert np.array_equal(
        val_mask(config, "sampleA", frame=frame), val_mask(config, "sampleB", frame=frame)
    )


def test_split_is_independent_of_row_order():
    """The property that lets the split survive interleaved, shuffled reading."""
    frame = make_sites()
    config = SplitConfig(strategy=PER_SAMPLE_SITE_HASH, seed=7)
    permutation = np.random.default_rng(0).permutation(frame.height)
    base = val_mask(config, "sampleA", frame=frame)
    shuffled = val_mask(config, "sampleA", frame=frame[permutation])
    assert np.array_equal(base[permutation], shuffled)


def test_train_and_val_are_complementary():
    from uv_vae.splitting import split_mask

    frame = make_sites()
    config = SplitConfig(strategy=PER_SAMPLE_SITE_HASH, seed=3)
    train = split_mask(config, "s", "train", frame=frame)
    val = split_mask(config, "s", "val", frame=frame)
    assert np.array_equal(train, ~val)
    assert (train | val).all()


def test_site_strategies_declare_the_columns_they_need():
    assert SplitConfig(strategy=PER_SAMPLE_SITE_HASH).extra_columns == (
        "CHROM", "POS", "REF", "ALT",
    )
    assert SplitConfig(strategy=PER_SAMPLE_ROW_HASH).extra_columns == ()


def test_site_hash_without_site_columns_raises():
    config = SplitConfig(strategy=PER_SAMPLE_SITE_HASH)
    with pytest.raises(ValueError, match="CHROM"):
        val_mask(config, "s", frame=pl.DataFrame({"REF": ["A"], "ALT": ["C"]}))
