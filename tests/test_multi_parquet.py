"""Tests for proportionally interleaved multi-file reading.

The fixtures imitate the real featuremaps' layout -- sorted by genomic position,
~9 adjacent reads per locus -- because that layout is the entire reason this
module exists.

Every row carries a unique ``RID``, declared as a numeric feature so it passes
through the encoder untouched.  That makes "which rows did the model actually
see?" directly checkable, which is what the coverage and epoch-sharding tests
turn on.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from uv_vae.features import FeatureSpec
from uv_vae.multi_parquet import (
    InterleavedRowSource,
    RowEncoder,
    SampleSource,
    allocate_draws,
    columns_referenced,
)
from uv_vae.splitting import PER_SAMPLE_ROW_HASH, PER_SAMPLE_SITE_HASH, SplitConfig

ROW_FILTER = "st = 'MIXED' AND et = 'MIXED' AND FILT = 1"

CATEGORICAL_SPECS = [
    FeatureSpec(name="REF", kind="c", values={"": 0, "A": 1, "C": 2, "G": 3, "T": 4}),
    FeatureSpec(name="ALT", kind="c", values={"": 0, "A": 1, "C": 2, "G": 3, "T": 4}),
]
NUMERIC_SPECS = [FeatureSpec(name="RID", kind="int"), FeatureSpec(name="QUAL", kind="float")]
FEATURE_SPECS = CATEGORICAL_SPECS + NUMERIC_SPECS


def make_encoder() -> RowEncoder:
    # mean 0 / std 1 so RID survives the encoder unchanged and rows stay traceable.
    return RowEncoder(
        CATEGORICAL_SPECS, NUMERIC_SPECS, {"RID": 0.0, "QUAL": 0.0}, {"RID": 1.0, "QUAL": 1.0}
    )


def write_featuremap(
    path,
    n_sites: int,
    seed: int,
    rid_offset: int,
    depth: int = 9,
    row_group_size: int = 512,
    keep_fraction: float = 0.6,
) -> int:
    """Write a parquet shaped like a real featuremap. Returns post-filter row count."""
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(np.arange(1, 10_000_000), size=n_sites, replace=False))
    depths = rng.integers(max(1, depth - 4), depth + 5, size=n_sites)
    total = int(depths.sum())

    # Rows at one locus are adjacent, exactly as in the production files.
    pos = np.repeat(positions, depths)
    ref = np.repeat(rng.choice(list("ACGT"), n_sites), depths)
    alt = np.repeat(rng.choice(list("ACGT"), n_sites), depths)
    keep = rng.random(total) < keep_fraction

    table = pa.table(
        {
            "CHROM": pa.array(np.where(pos < 5_000_000, "chr1", "chr2")),
            "POS": pa.array(pos.astype(np.int64)),
            "REF": pa.array(ref),
            "ALT": pa.array(alt),
            "RID": pa.array(np.arange(rid_offset, rid_offset + total, dtype=np.int64)),
            "QUAL": pa.array(rng.normal(30, 5, total).astype(np.float64)),
            "st": pa.array(np.where(keep, "MIXED", "OTHER")),
            "et": pa.array(np.where(keep, "MIXED", "OTHER")),
            "FILT": pa.array(np.ones(total, dtype=np.int64)),
        }
    )
    pq.write_table(table, path, row_group_size=row_group_size)
    return int(keep.sum())


@pytest.fixture
def samples(tmp_path):
    """Two deliberately unequal samples, so proportional weighting is observable."""
    specs = [("big", 900, 1), ("small", 300, 2)]
    sources = []
    offset = 0
    for name, n_sites, seed in specs:
        path = tmp_path / f"{name}.featuremap.parquet"
        rows = write_featuremap(path, n_sites=n_sites, seed=seed, rid_offset=offset)
        sources.append(SampleSource(sample_id=name, path=str(path), rows=rows))
        offset += 1_000_000
    return sources


def collect_rids(source: InterleavedRowSource) -> np.ndarray:
    return np.concatenate([batch[1][:, 0] for batch in source]).astype(np.int64)


def build(samples, split="train", **kwargs):
    params = dict(
        sources=samples,
        encoder=make_encoder(),
        split_config=SplitConfig(strategy=PER_SAMPLE_SITE_HASH, val_fraction=0.1, seed=42),
        split=split,
        row_filter=ROW_FILTER,
        batch_size=256,
        seed=42,
    )
    params.update(kwargs)
    return InterleavedRowSource(**params)


# --- allocation -------------------------------------------------------------

def test_allocate_draws_sums_to_batch_size():
    for size in (95, 256, 8192, 65536):
        weights = np.random.default_rng(0).random(95)
        assert allocate_draws(weights / weights.sum(), size).sum() == size


def test_allocate_draws_does_not_starve_small_samples():
    # Plain truncation floors a 0.02% sample to zero draws, so it would never
    # appear in any batch. Largest-remainder gives it its share.
    weights = np.array([0.9998, 0.0002])
    draws = allocate_draws(weights, 8192)
    assert draws.sum() == 8192
    assert draws[1] >= 1
    assert int(0.0002 * 8192) == 1  # documents what truncation would give


def test_allocate_draws_is_proportional_and_deterministic():
    weights = np.array([0.5, 0.3, 0.2])
    draws = allocate_draws(weights, 1000)
    assert draws.tolist() == [500, 300, 200]
    assert np.array_equal(draws, allocate_draws(weights, 1000))


@pytest.mark.parametrize("bad", [np.array([]), np.array([0.0, 0.0]), np.array([-1.0, 2.0])])
def test_allocate_draws_rejects_bad_weights(bad):
    with pytest.raises(ValueError):
        allocate_draws(bad, 100)


# --- column discovery -------------------------------------------------------

def test_columns_referenced_finds_filter_columns():
    available = {"st", "et", "FILT", "REF", "POS"}
    assert columns_referenced(ROW_FILTER, available) == {"st", "et", "FILT"}


def test_columns_referenced_ignores_literals_and_keywords():
    # Against a real schema, the string literal 'MIXED' and the keyword AND are
    # not columns, so the intersection drops them. Over-matching is harmless --
    # it would only read a column we do not need -- while under-matching cannot
    # pass silently, because the filter would raise on a missing column.
    schema = {"st", "et", "FILT", "REF", "ALT", "POS", "CHROM"}
    assert columns_referenced(ROW_FILTER, schema) == {"st", "et", "FILT"}
    assert "MIXED" not in columns_referenced(ROW_FILTER, schema)
    assert columns_referenced(None, {"st"}) == set()
    assert columns_referenced("", {"st"}) == set()


# --- composition ------------------------------------------------------------

def test_every_batch_contains_every_sample_in_proportion(samples):
    source = build(samples)
    report = source.describe()
    total = sum(s.rows for s in samples)
    for sample in samples:
        assert report.weights[sample.sample_id] == pytest.approx(sample.rows / total)
    assert sum(report.draws_per_batch.values()) == 256
    assert all(draw > 0 for draw in report.draws_per_batch.values())

    # Rows actually served must track the weights, not just the plan.
    batches = [batch for _, batch in zip(range(5), source)]
    assert all(batch[0].shape[0] == 256 for batch in batches)
    served = {r.source.sample_id: r.rows_served for r in source.readers}
    total_served = sum(served.values())
    for sample in samples:
        assert served[sample.sample_id] / total_served == pytest.approx(
            report.weights[sample.sample_id], abs=0.01
        )
    source.close()


def test_batch_size_below_sample_count_is_refused(samples):
    with pytest.raises(ValueError, match="smaller than"):
        build(samples, batch_size=1)


def test_batch_size_that_starves_a_sample_is_refused(tmp_path):
    big = SampleSource("big", str(tmp_path / "b.parquet"), rows=10_000_000)
    tiny = SampleSource("tiny", str(tmp_path / "t.parquet"), rows=1)
    write_featuremap(tmp_path / "b.parquet", n_sites=50, seed=1, rid_offset=0)
    write_featuremap(tmp_path / "t.parquet", n_sites=50, seed=2, rid_offset=100)
    with pytest.raises(ValueError, match="zero rows"):
        build([big, tiny], batch_size=4)


# --- coverage ---------------------------------------------------------------

def test_one_pass_uses_every_training_row_exactly_once(samples):
    source = build(samples)
    rids = collect_rids(source)
    assert len(rids) == len(set(rids.tolist())), "a row was emitted twice"

    expected = expected_rids(samples, "train")
    assert set(rids.tolist()) == expected
    source.close()


def test_train_and_val_partition_the_data(samples):
    train = build(samples, split="train")
    val = build(samples, split="val")
    train_rids = set(collect_rids(train).tolist())
    val_rids = set(collect_rids(val).tolist())
    assert train_rids & val_rids == set()
    assert train_rids | val_rids == expected_rids(samples, "both")
    assert 0.05 < len(val_rids) / (len(train_rids) + len(val_rids)) < 0.16
    train.close()
    val.close()


def expected_rids(samples, split: str) -> set[int]:
    """The RIDs a given split should contain, computed independently of the reader."""
    from uv_vae.splitting import split_mask

    config = SplitConfig(strategy=PER_SAMPLE_SITE_HASH, val_fraction=0.1, seed=42)
    out: set[int] = set()
    for sample in samples:
        frame = pl.read_parquet(sample.path)
        frame = pl.SQLContext(t=frame).execute(
            f"SELECT * FROM t WHERE {ROW_FILTER}", eager=True
        )
        if split == "both":
            out |= set(frame.get_column("RID").to_list())
            continue
        mask = split_mask(config, sample.sample_id, split, frame=frame)
        out |= set(frame.filter(pl.Series(mask)).get_column("RID").to_list())
    return out


# --- epoch sharding ---------------------------------------------------------

def test_epoch_shards_are_disjoint_and_cover_everything(samples):
    shards = 4
    source = build(samples, epoch_shards=shards)
    seen: list[set[int]] = []
    for epoch in range(shards):
        source.set_epoch(epoch)
        seen.append(set(collect_rids(source).tolist()))

    for i in range(shards):
        for j in range(i + 1, shards):
            assert seen[i] & seen[j] == set(), f"epoch {i} and {j} overlap"
    assert set().union(*seen) == expected_rids(samples, "train")
    source.close()


def test_a_shard_costs_roughly_one_over_e_of_a_pass(samples):
    full = build(samples, epoch_shards=1)
    full_rows = len(collect_rids(full))
    sharded = build(samples, epoch_shards=4)
    sharded.set_epoch(0)
    shard_rows = len(collect_rids(sharded))
    assert 0.15 * full_rows < shard_rows < 0.40 * full_rows
    full.close()
    sharded.close()


def test_sharded_epochs_reweight_to_what_is_actually_available(samples):
    """Draws follow the rows present in THIS epoch, not the whole-file totals.

    A stride of row groups does not hold exactly 1/E of every file. Weighting by
    whole-file counts makes readers exhaust at different times, leaving a tail of
    short batches that over-represents whoever still has rows.
    """
    source = build(samples, epoch_shards=4)
    source.set_epoch(1)
    report = source.describe()

    assert sum(report.draws_per_batch.values()) == 256
    assert all(draw > 0 for draw in report.draws_per_batch.values())
    assert pytest.approx(sum(report.epoch_weights.values())) == 1.0
    # epoch weights track this epoch's row groups, so they need not equal the
    # global weights -- but they must stay in the same neighbourhood.
    for sample_id, weight in report.epoch_weights.items():
        assert abs(weight - report.weights[sample_id]) < 0.25
    source.close()


def test_unsharded_epoch_weights_equal_the_global_weights(samples):
    source = build(samples, epoch_shards=1)
    report = source.describe()
    for sample_id, weight in report.epoch_weights.items():
        assert weight == pytest.approx(report.weights[sample_id])
    source.close()


def test_served_rows_track_the_epoch_weights(samples):
    source = build(samples, epoch_shards=2)
    source.set_epoch(0)
    list(source)
    report = source.last_report
    served = {r.source.sample_id: r.rows_served for r in source.readers}
    total = sum(served.values())
    ragged = report.ragged_rows / report.rows_emitted
    for sample_id, weight in report.epoch_weights.items():
        drift = abs(served[sample_id] / total - weight)
        assert drift <= ragged + 0.02, (
            f"{sample_id} drifted {drift:.4f} beyond what the ragged tail "
            f"({ragged:.4f}) explains"
        )
    source.close()


def test_next_cycle_reshuffles_rather_than_repeating(samples):
    """Row-group order is seeded on the CYCLE, so epoch e and e+E differ."""
    source = build(samples, epoch_shards=4)
    source.set_epoch(0)
    first = collect_rids(source).tolist()
    source.set_epoch(4)  # same stride index, next cycle
    second = collect_rids(source).tolist()
    assert first != second
    source.close()


# --- re-iteration -----------------------------------------------------------

def test_source_can_be_iterated_repeatedly(samples):
    """The validation loader is iterated once per epoch without set_epoch."""
    source = build(samples, split="val", epoch_varying=False)
    first = collect_rids(source)
    second = collect_rids(source)
    assert len(first) > 0
    assert np.array_equal(np.sort(first), np.sort(second))
    source.close()


def test_validation_plan_is_identical_across_epochs(samples):
    source = build(samples, split="val", epoch_varying=False)
    source.set_epoch(1)
    first = set(collect_rids(source).tolist())
    source.set_epoch(7)
    second = set(collect_rids(source).tolist())
    assert first == second, "early stopping compares val loss across epochs"
    source.close()


def test_cap_rows_limits_validation_io(samples):
    source = build(samples, split="val", epoch_varying=False)
    uncapped = len(collect_rids(source))
    limits = source.cap_rows(max_rows=uncapped // 4, keep_fraction=0.1)
    capped = len(collect_rids(source))
    assert limits, "expected per-sample row-group limits"
    assert capped < uncapped
    source.close()


# --- the point of the module: breaking locus clustering ---------------------

def test_interleaving_destroys_the_locus_clustering(samples):
    """Adjacent rows share a locus ~80% of the time on disk; a batch must not."""
    raw = pl.read_parquet(samples[0].path)
    raw = pl.SQLContext(t=raw).execute(f"SELECT * FROM t WHERE {ROW_FILTER}", eager=True)
    on_disk = raw.get_column("POS").to_numpy()
    on_disk_adjacency = float(np.mean(on_disk[1:] == on_disk[:-1]))

    source = build(samples, batch_size=512)
    batch = next(iter(source))
    rids = batch[1][:, 0].astype(np.int64)
    positions = rid_to_pos(samples)
    batch_pos = np.array([positions[int(r)] for r in rids])
    batch_adjacency = float(np.mean(batch_pos[1:] == batch_pos[:-1]))

    assert on_disk_adjacency > 0.5, "fixture is not clustered; the test is vacuous"
    assert batch_adjacency < 0.05
    # And the batch should span the genome, not one window.
    assert batch_pos.max() - batch_pos.min() > 0.5 * (on_disk.max() - on_disk.min())
    source.close()


def rid_to_pos(samples) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for sample in samples:
        frame = pl.read_parquet(sample.path, columns=["RID", "POS"])
        mapping.update(dict(zip(frame.get_column("RID").to_list(), frame.get_column("POS").to_list())))
    return mapping


def test_row_group_order_is_reproducible_across_instances(samples):
    a = build(samples)
    b = build(samples)
    a.set_epoch(3)
    b.set_epoch(3)
    assert np.array_equal(collect_rids(a), collect_rids(b))
    a.close()
    b.close()


# --- encoder ----------------------------------------------------------------

def test_encoder_matches_preprocess_helpers():
    """RowEncoder is a vectorised rewrite; it must stay bit-identical.

    ``preprocess`` imports torch, so this is the one test here that needs it.
    """
    pytest.importorskip("torch")
    from uv_vae.preprocess import encode_categorical_column, encode_numeric_column

    rng = np.random.default_rng(0)
    frame = pl.DataFrame(
        {
            "REF": rng.choice(["A", "C", "G", "T", None], 500).tolist(),
            "ALT": rng.choice(["A", "C", "G", "T", "N"], 500).tolist(),
            "RID": rng.integers(0, 10_000, 500).tolist(),
            "QUAL": [None if v < 0.1 else v for v in rng.random(500)],
        }
    )
    means = {"RID": 3.5, "QUAL": 0.5}
    stds = {"RID": 2.0, "QUAL": 0.25}
    categorical, numeric, mask = RowEncoder(
        CATEGORICAL_SPECS, NUMERIC_SPECS, means, stds
    ).encode(frame)

    expected_cat = np.stack(
        [encode_categorical_column(frame, spec) for spec in CATEGORICAL_SPECS], axis=1
    )
    assert np.array_equal(categorical, expected_cat)

    values, masks = zip(*[encode_numeric_column(frame, s) for s in NUMERIC_SPECS])
    mean_arr = np.array([means[s.name] for s in NUMERIC_SPECS], dtype=np.float32)
    std_arr = np.array([stds[s.name] for s in NUMERIC_SPECS], dtype=np.float32)
    expected_mask = np.stack(masks, axis=1)
    expected_num = np.stack(values, axis=1)
    expected_num = np.where(expected_mask > 0, expected_num, mean_arr)
    expected_num = (expected_num - mean_arr) / std_arr

    assert np.array_equal(mask, expected_mask)
    assert np.array_equal(numeric, expected_num)


def test_encoder_handles_unknown_categories_as_null():
    encoder = RowEncoder(
        CATEGORICAL_SPECS, NUMERIC_SPECS, {"RID": 0.0, "QUAL": 0.0}, {"RID": 1.0, "QUAL": 1.0}
    )
    frame = pl.DataFrame(
        {"REF": ["A", "Z", None], "ALT": ["C", "C", "C"], "RID": [1, 2, 3], "QUAL": [1.0, 2.0, 3.0]}
    )
    categorical, _, _ = encoder.encode(frame)
    # "Z" is not in the map and None fills with the null token; both take index 0.
    assert categorical[:, 0].tolist() == [1, 0, 0]


def test_missing_feature_column_is_reported(tmp_path, samples):
    path = tmp_path / "narrow.parquet"
    pq.write_table(pa.table({"POS": pa.array([1, 2, 3])}), path)
    with pytest.raises(RuntimeError, match="missing feature columns"):
        build([SampleSource("narrow", str(path), rows=3)], batch_size=2)


def test_row_hash_strategy_needs_no_site_columns(samples):
    source = build(
        samples,
        split_config=SplitConfig(strategy=PER_SAMPLE_ROW_HASH, val_fraction=0.1, seed=42),
    )
    assert len(collect_rids(source)) > 0
    source.close()
