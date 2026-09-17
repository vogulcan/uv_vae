"""Tests for per-file, cached, analytically-combined normalisation statistics."""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from uv_vae.features import FeatureSpec
from uv_vae.stats_cache import (
    ColumnMoments,
    combine_moment_tree,
    combine_moments,
    compute_file_stats,
    load_or_compute_stats,
    sample_id_for,
    spec_fingerprint,
)

ROW_FILTER = "st = 'MIXED' AND et = 'MIXED' AND FILT = 1"

SPECS = [
    FeatureSpec(name="REF", kind="c", values={"": 0, "A": 1, "C": 2, "G": 3, "T": 4}),
    FeatureSpec(name="QUAL", kind="float"),
    FeatureSpec(name="DEPTH", kind="int"),
    FeatureSpec(name="ALWAYS_NULL", kind="float"),
    FeatureSpec(name="NULL_IN_ONE", kind="float"),
]


def write_sample(path, n: int, seed: int, null_in_one: bool, keep_fraction: float = 0.7):
    rng = np.random.default_rng(seed)
    keep = rng.random(n) < keep_fraction
    table = pa.table(
        {
            "REF": pa.array(rng.choice(list("ACGT"), n)),
            # A large offset is where sum(x^2) accumulation would lose precision.
            "QUAL": pa.array(rng.normal(1e7, 3.0, n)),
            "DEPTH": pa.array(rng.integers(1, 100, n).astype(np.int64)),
            "ALWAYS_NULL": pa.array([None] * n, type=pa.float64()),
            "NULL_IN_ONE": pa.array(
                [None] * n if null_in_one else rng.normal(5, 1, n).tolist(),
                type=pa.float64(),
            ),
            "st": pa.array(np.where(keep, "MIXED", "OTHER")),
            "et": pa.array(np.where(keep, "MIXED", "OTHER")),
            "FILT": pa.array(np.ones(n, dtype=np.int64)),
        }
    )
    pq.write_table(table, path)
    return table, keep


# --- Chan combination -------------------------------------------------------

def test_combination_matches_numpy_on_unequal_partitions():
    rng = np.random.default_rng(0)
    sizes = [5, 1_000, 37, 250_000, 2, 900, 100_000]
    arrays = [rng.normal(1e7 + i * 1000, 3.5, n) for i, n in enumerate(sizes)]
    parts = [
        ColumnMoments(n=len(a), mean=float(a.mean()), m2=float(((a - a.mean()) ** 2).sum()))
        for a in arrays
    ]
    combined = combine_moment_tree(parts)
    everything = np.concatenate(arrays)

    assert combined.n == everything.size
    assert combined.mean == pytest.approx(everything.mean(), rel=1e-12)
    assert combined.std == pytest.approx(everything.std(ddof=1), rel=1e-9)


def test_combining_empty_partitions_is_a_no_op():
    empty = ColumnMoments(n=0, mean=0.0, m2=0.0)
    part = ColumnMoments(n=10, mean=3.0, m2=9.0)
    assert combine_moments(empty, part) == part
    assert combine_moments(part, empty) == part
    assert combine_moment_tree([]) == empty
    assert combine_moment_tree([part]) == part


def test_two_singletons_combine_to_the_sample_standard_deviation():
    combined = combine_moments(
        ColumnMoments(n=1, mean=5.0, m2=0.0), ColumnMoments(n=1, mean=7.0, m2=0.0)
    )
    assert combined.mean == pytest.approx(6.0)
    assert combined.std == pytest.approx(np.std([5.0, 7.0], ddof=1))


def test_combination_is_order_independent():
    rng = np.random.default_rng(3)
    parts = []
    for n in (11, 5000, 3, 77):
        a = rng.normal(0, 1, n)
        parts.append(ColumnMoments(n=n, mean=float(a.mean()), m2=float(((a - a.mean()) ** 2).sum())))
    forward = combine_moment_tree(parts)
    backward = combine_moment_tree(list(reversed(parts)))
    assert forward.mean == pytest.approx(backward.mean, rel=1e-12)
    assert forward.std == pytest.approx(backward.std, rel=1e-12)


# --- single file ------------------------------------------------------------

def test_file_stats_respect_the_row_filter_and_null_counts(tmp_path):
    path = tmp_path / "a.featuremap.parquet"
    table, keep = write_sample(path, n=5000, seed=1, null_in_one=False)

    stats = compute_file_stats(path, "a", SPECS, row_filter=ROW_FILTER)
    assert stats.rows == int(keep.sum())
    assert stats.non_null_counts["ALWAYS_NULL"] == 0
    assert stats.non_null_counts["QUAL"] == int(keep.sum())

    qual = np.asarray(table["QUAL"])[keep]
    assert stats.moments["QUAL"].mean == pytest.approx(qual.mean(), rel=1e-12)
    assert stats.moments["QUAL"].std == pytest.approx(qual.std(ddof=1), rel=1e-9)


def test_each_column_carries_its_own_n(tmp_path):
    """SQL aggregates skip NULLs, so a column's n is its non-null count.

    Weighting by the file's row count instead would misweight every column that
    has any nulls at all.
    """
    path = tmp_path / "a.parquet"
    write_sample(path, n=2000, seed=4, null_in_one=True)
    stats = compute_file_stats(path, "a", SPECS, row_filter=ROW_FILTER)
    assert stats.moments["NULL_IN_ONE"].n == 0
    assert stats.moments["QUAL"].n == stats.rows


# --- combining files --------------------------------------------------------

def test_combined_stats_match_a_single_pass_over_both_files(tmp_path):
    paths = []
    frames = []
    for i, n in enumerate((3000, 11000)):
        path = tmp_path / f"s{i}.featuremap.parquet"
        table, keep = write_sample(path, n=n, seed=10 + i, null_in_one=False)
        paths.append(path)
        frames.append(np.asarray(table["QUAL"])[keep])

    combined = load_or_compute_stats(paths, SPECS, row_filter=ROW_FILTER, verbose=False)
    everything = np.concatenate(frames)

    assert combined.total_rows == everything.size
    assert combined.numeric_means["QUAL"] == pytest.approx(everything.mean(), rel=1e-12)
    assert combined.numeric_stds["QUAL"] == pytest.approx(everything.std(ddof=1), rel=1e-9)


def test_all_null_drop_uses_counts_summed_across_files(tmp_path):
    """A feature null in one sample but populated in another must survive.

    Deriving the model shape from a single file would build the wrong network.
    """
    paths = []
    for i, null_in_one in enumerate((True, False)):
        path = tmp_path / f"s{i}.featuremap.parquet"
        write_sample(path, n=2000, seed=20 + i, null_in_one=null_in_one)
        paths.append(path)

    combined = load_or_compute_stats(paths, SPECS, row_filter=ROW_FILTER, verbose=False)
    kept = {spec.name for spec in combined.numeric_specs}
    assert "NULL_IN_ONE" in kept, "dropped a feature that is populated in one sample"
    assert "ALWAYS_NULL" not in kept
    assert combined.dropped_all_null_features == ["ALWAYS_NULL"]


def test_degenerate_spread_becomes_unit_std(tmp_path):
    path = tmp_path / "c.parquet"
    n = 500
    pq.write_table(
        pa.table(
            {
                "REF": pa.array(["A"] * n),
                "QUAL": pa.array([7.0] * n),  # constant column
                "DEPTH": pa.array(np.arange(n, dtype=np.int64)),
                "ALWAYS_NULL": pa.array([None] * n, type=pa.float64()),
                "NULL_IN_ONE": pa.array([1.0] * n),
                "st": pa.array(["MIXED"] * n),
                "et": pa.array(["MIXED"] * n),
                "FILT": pa.array(np.ones(n, dtype=np.int64)),
            }
        ),
        path,
    )
    combined = load_or_compute_stats([path], SPECS, row_filter=ROW_FILTER, verbose=False)
    # Matches streaming.compute_streaming_stats: normalisation becomes a no-op
    # rather than a division by ~0.
    assert combined.numeric_stds["QUAL"] == 1.0
    assert combined.numeric_means["QUAL"] == pytest.approx(7.0)


def test_no_matching_rows_is_an_error(tmp_path):
    path = tmp_path / "d.parquet"
    write_sample(path, n=100, seed=1, null_in_one=False)
    with pytest.raises(RuntimeError, match="No rows"):
        load_or_compute_stats([path], SPECS, row_filter="FILT = 999", verbose=False)


def test_empty_path_list_is_an_error():
    with pytest.raises(ValueError):
        load_or_compute_stats([], SPECS, verbose=False)


# --- caching ----------------------------------------------------------------

def test_cache_is_written_and_reused(tmp_path):
    path = tmp_path / "s0.featuremap.parquet"
    write_sample(path, n=4000, seed=5, null_in_one=False)
    cache = tmp_path / "stats.json"

    first = load_or_compute_stats([path], SPECS, ROW_FILTER, cache_path=cache, verbose=False)
    assert cache.exists()
    entries = json.loads(cache.read_text())["entries"]
    assert len(entries) == 1

    second = load_or_compute_stats([path], SPECS, ROW_FILTER, cache_path=cache, verbose=False)
    assert first.total_rows == second.total_rows
    assert first.numeric_means == second.numeric_means
    assert first.numeric_stds == second.numeric_stds


def test_cache_misses_when_the_file_changes(tmp_path):
    path = tmp_path / "s0.featuremap.parquet"
    write_sample(path, n=4000, seed=5, null_in_one=False)
    cache = tmp_path / "stats.json"
    first = load_or_compute_stats([path], SPECS, ROW_FILTER, cache_path=cache, verbose=False)

    time.sleep(0.01)
    write_sample(path, n=9000, seed=6, null_in_one=False)
    os.utime(path, None)
    second = load_or_compute_stats([path], SPECS, ROW_FILTER, cache_path=cache, verbose=False)

    assert second.total_rows != first.total_rows
    assert len(json.loads(cache.read_text())["entries"]) == 2


def test_cache_misses_when_the_row_filter_changes(tmp_path):
    path = tmp_path / "s0.parquet"
    write_sample(path, n=3000, seed=7, null_in_one=False)
    cache = tmp_path / "stats.json"
    a = load_or_compute_stats([path], SPECS, ROW_FILTER, cache_path=cache, verbose=False)
    b = load_or_compute_stats([path], SPECS, "FILT = 1", cache_path=cache, verbose=False)
    assert a.total_rows != b.total_rows
    assert len(json.loads(cache.read_text())["entries"]) == 2


def test_a_corrupt_cache_is_ignored_rather_than_fatal(tmp_path):
    path = tmp_path / "s0.parquet"
    write_sample(path, n=1000, seed=8, null_in_one=False)
    cache = tmp_path / "stats.json"
    cache.write_text("{ not json")
    stats = load_or_compute_stats([path], SPECS, ROW_FILTER, cache_path=cache, verbose=False)
    assert stats.total_rows > 0


def test_adding_a_sample_only_scans_the_new_one(tmp_path):
    """The reason statistics are stored per file rather than per cohort."""
    cache = tmp_path / "stats.json"
    first = tmp_path / "s0.featuremap.parquet"
    write_sample(first, n=2000, seed=1, null_in_one=False)
    load_or_compute_stats([first], SPECS, ROW_FILTER, cache_path=cache, verbose=False)
    entries_before = set(json.loads(cache.read_text())["entries"])

    second = tmp_path / "s1.featuremap.parquet"
    write_sample(second, n=2000, seed=2, null_in_one=False)
    load_or_compute_stats([first, second], SPECS, ROW_FILTER, cache_path=cache, verbose=False)
    entries_after = set(json.loads(cache.read_text())["entries"])

    assert entries_before < entries_after
    assert len(entries_after) == 2, "the existing file was re-keyed instead of reused"


def test_fingerprint_changes_with_the_feature_spec():
    other = SPECS[:-1]
    assert spec_fingerprint(SPECS) != spec_fingerprint(other)
    assert spec_fingerprint(SPECS) == spec_fingerprint(list(SPECS))


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("wt0-12-ppm0050.featuremap.parquet", "wt0-12-ppm0050"),
        ("/a/b/sampleX.parquet", "sampleX"),
    ],
)
def test_sample_id_comes_from_the_filename(filename, expected):
    # The sample id salts the per-sample split, so it must not depend on glob
    # order or an enumeration index that would shift when a file is added.
    assert sample_id_for(filename) == expected
