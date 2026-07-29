"""Proportionally interleaved reading across many per-sample parquet featuremaps.

The problem this solves has two halves, and they are independent.

**Composition.**  With 95 per-sample files of unequal size, a batch must contain
every sample in its true proportion, or the VAE's implied prior over samples is
whatever the reader happened to visit.  Concatenating the files, or globbing
them, gives 95 contiguous blocks: with a 500 000-row shuffle window against a
~50M-row block, every batch for the first ~50M rows is 100% sample 1, then 100%
sample 2, and so on.  That is not a mildly unbalanced batch -- it is sequential
fine-tuning on 95 datasets in turn, and the final latent geometry would
disproportionately reflect whichever samples were read last.

:class:`InterleavedRowSource` instead keeps one reader per file and draws
``k_i = w_i * batch_size`` rows from file *i*, where ``w_i`` is that file's share
of the total post-filter row count.  Every batch holds all 95 samples in their
true proportions, and because the draw rate is proportional to size all readers
exhaust together -- so one pass uses every row exactly once, with no truncation
and no repetition.

**Physical layout.**  Measured on the production featuremaps: the files are
perfectly sorted by ``(CHROM, POS)`` -- zero out-of-order steps in a 3M-row probe
-- and **88% of consecutive rows sit at the same locus**, at ~8-9 reads per site.
All structural correlation in the data is local and is an artefact of read order.

A sequential scan with a shuffle buffer cannot fix that at any affordable size,
because a contiguous 500 000-row window *is* a narrow genomic window: shuffling
inside it still leaves every row in the batch drawn from the same small region.
95 readers x 500 000 rows would also be ~10 GB of host RAM.

So the reader shuffles **row-group order** instead.  Each file holds ~1 570
independently addressable row groups of ~123 000 rows, and
``ParquetFile.read_row_group`` can visit them in any order.  Shuffling that order
means consecutive reads come from unrelated genomic locations, and a second
shuffle *within* the decoded buffer breaks the residual within-group ordering.
Neither costs extra I/O.

Design constraints worth keeping
--------------------------------

*Decode sequentially, buffer concurrently.*  The GPU decode path (cuDF) allocates
from a 4 GB RMM pool inside a 16 GB per-process budget; 95 concurrent decodes at
the streaming default chunk size would want ~5.7 GB and spill.  The batch loop
pulls from one reader at a time, so peak decode memory is one row group
regardless of how many files are open.

*One DuckDB instance, not 95.*  ``connect_duckdb`` opens a fresh in-memory
*database* per call, each with its own buffer manager, memory limit near 80% of
system RAM, and thread pool.  This module does not open any: the row filter is
evaluated by polars (measured ~5x faster than DuckDB over an in-memory Arrow
table, and 1.6x faster than pyarrow.compute), through
``polars.SQLContext`` so the ``--row-filter`` SQL string keeps working unchanged.

This module deliberately does **not** import torch, so the interleave logic is
testable without it.  The ``IterableDataset`` wrapper lives in
``multi_streaming.py``.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from uv_vae.features import FeatureSpec
from uv_vae.splitting import SplitConfig, split_mask, stable_seed

# Post-filter rows each reader holds decoded and shuffled at once.  At ~320 B/row
# (11 int64 categorical + 29 float32 numeric + 29 float32 mask) this is ~10 MB per
# reader, so 95 readers cost ~1 GB of host RAM.
DEFAULT_SHUFFLE_BUFFER_ROWS = 32_768

# Below this many row groups per sample per epoch, the readers' exhaustion points
# scatter enough to leave a noticeable tail of short batches. At 1 570 row groups
# per file, epoch_shards=20 gives ~78 and is comfortably clear of it.
MIN_ROW_GROUPS_PER_EPOCH = 20

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def columns_referenced(sql_fragment: str | None, available: set[str]) -> set[str]:
    """Column names a SQL fragment mentions.

    Used to work out which columns the row filter needs on top of the feature
    columns -- ``FILT`` is in the default filter but is not a feature.  Any
    identifier that is not a real column (``MIXED`` inside a string literal, the
    keyword ``AND``) drops out through the intersection, so over-matching is
    harmless.  Under-matching cannot pass silently: the filter would raise on a
    missing column.
    """
    if not sql_fragment:
        return set()
    return {token for token in _IDENTIFIER.findall(sql_fragment) if token in available}


class RowEncoder:
    """Categorical/numeric encoding for one decoded chunk.

    Produces output bit-identical to ``StreamingParquetDataset._encode_chunk``,
    which is pinned by ``test_encoder_matches_preprocess_helpers``.  The
    categorical path is vectorised through ``Series.replace_strict`` rather than
    ``np.fromiter`` over a Python generator: measured 1.7x faster on real data
    and, over 4.75 billion rows, that is the difference between a tractable pass
    and an intractable one.  ``preprocess.encode_categorical_column`` is left
    alone so single-file runs stay byte-for-byte comparable.
    """

    def __init__(
        self,
        categorical_specs: list[FeatureSpec],
        numeric_specs: list[FeatureSpec],
        numeric_means: dict[str, float],
        numeric_stds: dict[str, float],
    ) -> None:
        self.categorical_specs = categorical_specs
        self.numeric_specs = numeric_specs
        self.means = np.array(
            [numeric_means[spec.name] for spec in numeric_specs], dtype=np.float32
        )
        self.stds = np.array(
            [numeric_stds[spec.name] for spec in numeric_specs], dtype=np.float32
        )
        # null_token / null_index come straight off FeatureSpec rather than being
        # re-derived here, so this cannot drift from
        # preprocess.encode_categorical_column.
        self._cat_lookup = []
        for spec in categorical_specs:
            if spec.values is None:
                raise ValueError(f"Categorical feature {spec.name} has no category map")
            self._cat_lookup.append(
                (spec.name, dict(spec.values), spec.null_token, spec.values.get(spec.null_token, 0))
            )

    @property
    def feature_names(self) -> list[str]:
        return [s.name for s in self.categorical_specs] + [s.name for s in self.numeric_specs]

    def encode(self, frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        height = frame.height

        if self._cat_lookup:
            columns = []
            for name, lookup, null_token, null_index in self._cat_lookup:
                columns.append(
                    frame.get_column(name)
                    .cast(pl.String)
                    .fill_null(null_token)
                    .replace_strict(lookup, default=null_index, return_dtype=pl.Int64)
                    .to_numpy()
                )
            categorical = np.stack(columns, axis=1).astype(np.int64, copy=False)
        else:
            categorical = np.zeros((height, 0), dtype=np.int64)

        if self.numeric_specs:
            values = []
            masks = []
            for spec in self.numeric_specs:
                column = (
                    frame.get_column(spec.name)
                    .cast(pl.Float32)
                    .fill_null(float("nan"))
                    .to_numpy()
                    .astype(np.float32, copy=False)
                )
                values.append(column)
                masks.append((~np.isnan(column)).astype(np.float32, copy=False))
            numeric = np.stack(values, axis=1)
            mask = np.stack(masks, axis=1)
            numeric = np.where(mask > 0, numeric, self.means)
            numeric = (numeric - self.means) / self.stds
        else:
            numeric = np.zeros((height, 0), dtype=np.float32)
            mask = np.zeros((height, 0), dtype=np.float32)

        return categorical, numeric.astype(np.float32, copy=False), mask


def allocate_draws(weights: np.ndarray, batch_size: int) -> np.ndarray:
    """Split ``batch_size`` across files by ``weights``, using largest remainder.

    Plain truncation floors any sample below ``1/batch_size`` of the pool to zero
    draws, so a sample at 0.2% of 95 unequal files would never appear in a batch
    at all.  Largest-remainder (Hamilton) apportionment gives every file its floor
    and hands the leftover seats to the largest fractional parts, so the draws sum
    to exactly ``batch_size`` and small samples still get their share.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("weights must be a non-empty 1-D array")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    total = weights.sum()
    if total <= 0:
        raise ValueError("weights must not sum to zero")

    exact = weights / total * batch_size
    floors = np.floor(exact).astype(np.int64)
    remainder = int(batch_size - floors.sum())
    if remainder > 0:
        # Ties broken by index so the allocation is deterministic.
        order = np.lexsort((np.arange(weights.size), -(exact - floors)))
        floors[order[:remainder]] += 1
    return floors


@dataclass
class SampleSource:
    """One per-sample parquet file and everything the reader needs about it."""

    sample_id: str
    path: str
    rows: int
    """Post-filter row count for the WHOLE file (both splits)."""


class ParquetSampleReader:
    """Streams one sample's rows in shuffled row-group order.

    Two levels of shuffling, because the files are sorted by genomic position:

    1. **row-group order** -- reshuffled every epoch (or every *cycle*, when epoch
       sharding is on), so consecutive reads come from unrelated loci;
    2. **within the decode buffer** -- the whole buffer is permuted on every
       refill, so a batch's contribution from this file is a random sample of the
       window rather than a run of adjacent reads at the same site.

    Without (2), taking ``k_i`` rows sequentially out of one decoded row group
    would hand the model ~86 reads spanning only ~10 adjacent loci.
    """

    def __init__(
        self,
        source: SampleSource,
        encoder: RowEncoder,
        split_config: SplitConfig,
        split: str,
        row_filter: str | None,
        seed: int,
        shuffle_buffer_rows: int = DEFAULT_SHUFFLE_BUFFER_ROWS,
        epoch_varying: bool = True,
        row_group_limit: int | None = None,
    ) -> None:
        self.source = source
        self.encoder = encoder
        self.split_config = split_config
        self.split = split
        self.row_filter = row_filter
        self.seed = seed
        self.shuffle_buffer_rows = max(1, int(shuffle_buffer_rows))
        # Training reshuffles row groups every epoch and permutes each buffer.
        # Validation does neither: its row-group order is shuffled ONCE with a
        # fixed seed, so when row_group_limit caps it to a subset that subset is a
        # random spread across the genome (not the first K groups, which would be
        # one contiguous region) and is the SAME subset every epoch. Early
        # stopping compares val loss across epochs, so it has to be the same rows.
        self.epoch_varying = epoch_varying
        self.row_group_limit = row_group_limit

        self._handle = pq.ParquetFile(source.path, pre_buffer=False, memory_map=False)
        metadata = self._handle.metadata
        self.num_row_groups = metadata.num_row_groups
        group_rows = [metadata.row_group(i).num_rows for i in range(self.num_row_groups)]
        self._group_rows = np.asarray(group_rows, dtype=np.int64)
        self.total_raw_rows = int(self._group_rows.sum())
        # Physical start offset of each row group, so per_sample_row_hash can key
        # on a row's address in the file rather than its position in this scan.
        self._group_start = np.concatenate([[0], np.cumsum(group_rows)]).astype(np.int64)

        available = set(self._handle.schema_arrow.names)
        needed = set(encoder.feature_names)
        needed |= columns_referenced(row_filter, available)
        needed |= {c for c in split_config.extra_columns if c in available}
        missing_features = sorted(set(encoder.feature_names) - available)
        if missing_features:
            raise RuntimeError(
                f"{source.path} is missing feature columns {missing_features}"
            )
        missing_split = sorted(set(split_config.extra_columns) - available)
        if missing_split:
            raise RuntimeError(
                f"{source.path} is missing {missing_split}, required by split strategy "
                f"{split_config.strategy!r}"
            )
        self._read_columns = sorted(needed)

        self._order: np.ndarray = np.arange(self.num_row_groups)
        self._cursor_group = 0
        self._buffer: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._buffer_cursor = 0
        self._rng = np.random.default_rng(seed)
        self.rows_served = 0
        self.set_epoch(0, epoch_shards=1)

    # -- planning ---------------------------------------------------------

    def set_epoch(self, epoch: int, epoch_shards: int = 1) -> None:
        """Choose this epoch's row groups and reset the read cursor.

        With ``epoch_shards = E`` the row-group order is shuffled once per *cycle*
        of E epochs and epoch *e* takes the stride ``order[e % E :: E]``.  The E
        strides are disjoint and their union is every row group, so a complete
        pass over the file costs E epochs instead of one -- and each epoch is
        still a proportional mix of every sample.  Seeding on the cycle rather
        than the epoch is what makes the strides disjoint; reshuffling every
        epoch would resample with replacement across the cycle.
        """
        shards = max(1, int(epoch_shards)) if self.epoch_varying else 1
        cycle = int(epoch) // shards
        order = np.arange(self.num_row_groups)
        order_key = cycle if self.epoch_varying else "fixed"
        np.random.default_rng(
            stable_seed(self.source.sample_id, self.seed, order_key)
        ).shuffle(order)
        if shards > 1:
            order = order[int(epoch) % shards :: shards]
        if self.row_group_limit is not None:
            order = order[: self.row_group_limit]
        self._order = order
        self._cursor_group = 0
        self._buffer = None
        self._buffer_cursor = 0
        self._rng = np.random.default_rng(stable_seed(self.source.sample_id, self.seed, epoch, "buf"))
        self.rows_served = 0

    @property
    def planned_row_groups(self) -> int:
        return int(self._order.size)

    @property
    def planned_rows(self) -> float:
        """Estimated post-filter rows this reader will yield *this epoch*.

        Scales the file's known post-filter total by the share of raw rows that
        this epoch's row groups hold.  Exact post-filter counts per row group are
        not known without reading them, but filter yield is close to uniform
        across groups, so this is accurate enough to be worth far more than
        assuming every epoch holds exactly ``1/E`` of the file.
        """
        if self.total_raw_rows == 0:
            return 0.0
        planned_raw = float(self._group_rows[self._order].sum())
        return self.source.rows * planned_raw / self.total_raw_rows

    # -- reading ----------------------------------------------------------

    def _decode_group(self, group_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        table = self._handle.read_row_group(group_index, columns=self._read_columns)
        if table.num_rows == 0:
            return None
        frame = pl.from_arrow(table)

        start = int(self._group_start[group_index])
        positions = np.arange(start, start + table.num_rows, dtype=np.int64)

        if self.row_filter:
            frame = frame.with_columns(pl.Series("__row_position", positions))
            frame = pl.SQLContext(rows=frame).execute(
                f"SELECT * FROM rows WHERE {self.row_filter}", eager=True
            )
            if frame.height == 0:
                return None
            positions = frame.get_column("__row_position").to_numpy()

        keep = split_mask(
            self.split_config,
            self.source.sample_id,
            self.split,
            frame=frame,
            row_positions=positions,
        )
        if not keep.any():
            return None
        frame = frame.filter(pl.Series(keep))

        return self.encoder.encode(frame)

    def _refill(self) -> bool:
        """Decode row groups until the buffer is full or the epoch's groups run out.

        Returns False only when nothing is left.  An empty result from one row
        group is routine rather than terminal -- a validation reader keeps ~10% of
        rows and the row filter removes ~74% -- so this loops instead of stopping.
        """
        leftover_cat, leftover_num, leftover_mask = None, None, None
        if self._buffer is not None and self._buffer_cursor < self._buffer[0].shape[0]:
            leftover_cat = self._buffer[0][self._buffer_cursor :]
            leftover_num = self._buffer[1][self._buffer_cursor :]
            leftover_mask = self._buffer[2][self._buffer_cursor :]

        chunks_cat = [leftover_cat] if leftover_cat is not None else []
        chunks_num = [leftover_num] if leftover_num is not None else []
        chunks_mask = [leftover_mask] if leftover_mask is not None else []
        buffered = chunks_cat[0].shape[0] if chunks_cat else 0

        while buffered < self.shuffle_buffer_rows and self._cursor_group < self._order.size:
            group_index = int(self._order[self._cursor_group])
            self._cursor_group += 1
            decoded = self._decode_group(group_index)
            if decoded is None:
                continue
            chunks_cat.append(decoded[0])
            chunks_num.append(decoded[1])
            chunks_mask.append(decoded[2])
            buffered += decoded[0].shape[0]

        if buffered == 0:
            self._buffer = None
            self._buffer_cursor = 0
            return False

        cat = np.concatenate(chunks_cat) if len(chunks_cat) > 1 else chunks_cat[0]
        num = np.concatenate(chunks_num) if len(chunks_num) > 1 else chunks_num[0]
        mask = np.concatenate(chunks_mask) if len(chunks_mask) > 1 else chunks_mask[0]

        if self.epoch_varying:
            perm = self._rng.permutation(cat.shape[0])
            cat, num, mask = cat[perm], num[perm], mask[perm]

        self._buffer = (cat, num, mask)
        self._buffer_cursor = 0
        return True

    def take(self, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Up to ``count`` rows, or None when this reader is exhausted."""
        if count <= 0:
            return None
        parts_cat, parts_num, parts_mask = [], [], []
        remaining = count

        while remaining > 0:
            if self._buffer is None or self._buffer_cursor >= self._buffer[0].shape[0]:
                if not self._refill():
                    break
            cat, num, mask = self._buffer
            available = cat.shape[0] - self._buffer_cursor
            take = min(remaining, available)
            stop = self._buffer_cursor + take
            parts_cat.append(cat[self._buffer_cursor : stop])
            parts_num.append(num[self._buffer_cursor : stop])
            parts_mask.append(mask[self._buffer_cursor : stop])
            self._buffer_cursor = stop
            remaining -= take

        if not parts_cat:
            return None
        served = sum(part.shape[0] for part in parts_cat)
        self.rows_served += served
        if len(parts_cat) == 1:
            return parts_cat[0], parts_num[0], parts_mask[0]
        return (
            np.concatenate(parts_cat),
            np.concatenate(parts_num),
            np.concatenate(parts_mask),
        )

    def close(self) -> None:
        self._handle.close()


@dataclass
class InterleaveReport:
    """What the reader actually did -- copied into the run's training_report.json."""

    sample_ids: list[str] = field(default_factory=list)
    rows_per_sample: dict[str, int] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    """Each sample's share of the whole cohort -- the target composition."""
    draws_per_batch: dict[str, int] = field(default_factory=dict)
    epoch_weights: dict[str, float] = field(default_factory=dict)
    """Share of the rows available in THIS epoch's row groups.

    Equals ``weights`` when ``epoch_shards == 1``; with sharding it differs
    slightly because a stride of row groups does not hold exactly 1/E of each
    file. Draws follow this, not ``weights``, so readers exhaust together.
    """
    row_groups_this_epoch: dict[str, int] = field(default_factory=dict)
    rows_served: dict[str, int] = field(default_factory=dict)
    batches: int = 0
    full_batches: int = 0
    ragged_batches: int = 0
    rows_emitted: int = 0
    ragged_rows: int = 0
    """Rows emitted in short batches at the end of an epoch.

    Readers are drawn from in proportion to their size, so they exhaust at
    roughly the same time -- but not exactly, because per-row-group filter yield
    varies and files hold different numbers of row groups. Once the first reader
    runs dry the remaining batches are short and over-represent whichever samples
    still have rows.

    The alternative would be to end the epoch when the first reader exhausts,
    which keeps every batch exactly proportional but drops the tail -- and "every
    row passes through the model" is the requirement this design exists to meet.
    So the tail is kept and measured instead. It shrinks as row groups per file
    per epoch grows: negligible at 1 570 groups across 95 files, but visible on
    small inputs, which is why it is reported rather than assumed away.
    """

    def as_dict(self) -> dict:
        return {
            "sample_ids": list(self.sample_ids),
            "rows_per_sample": dict(self.rows_per_sample),
            "weights": dict(self.weights),
            "epoch_weights": dict(self.epoch_weights),
            "draws_per_batch": dict(self.draws_per_batch),
            "row_groups_this_epoch": dict(self.row_groups_this_epoch),
            "rows_served": dict(self.rows_served),
            "batches": self.batches,
            "full_batches": self.full_batches,
            "ragged_batches": self.ragged_batches,
            "rows_emitted": self.rows_emitted,
            "ragged_rows": self.ragged_rows,
            "ragged_row_fraction": (
                self.ragged_rows / self.rows_emitted if self.rows_emitted else 0.0
            ),
        }


class InterleavedRowSource:
    """Builds proportionally-composed batches from many per-sample readers.

    Yields ``(categorical, numeric, mask)`` numpy arrays.  The torch wrapper is in
    ``multi_streaming.py`` so this stays importable and testable without torch.
    """

    def __init__(
        self,
        sources: list[SampleSource],
        encoder: RowEncoder,
        split_config: SplitConfig,
        split: str,
        row_filter: str | None,
        batch_size: int,
        seed: int = 42,
        epoch_shards: int = 1,
        shuffle_buffer_rows: int = DEFAULT_SHUFFLE_BUFFER_ROWS,
        epoch_varying: bool = True,
        row_group_limits: dict[str, int] | None = None,
    ) -> None:
        if not sources:
            raise ValueError("InterleavedRowSource needs at least one sample source")
        if batch_size < len(sources):
            raise ValueError(
                f"batch_size={batch_size} is smaller than the {len(sources)} samples being "
                f"interleaved, so some samples could not appear in a batch at all. "
                f"Use batch_size >= {len(sources)}."
            )
        self.sources = sources
        self.batch_size = int(batch_size)
        self.split = split
        self.epoch_shards = max(1, int(epoch_shards))
        self._epoch = 0

        limits = row_group_limits or {}
        self.readers = [
            ParquetSampleReader(
                source=source,
                encoder=encoder,
                split_config=split_config,
                split=split,
                row_filter=row_filter,
                seed=seed,
                shuffle_buffer_rows=shuffle_buffer_rows,
                epoch_varying=epoch_varying,
                row_group_limit=limits.get(source.sample_id),
            )
            for source in sources
        ]

        counts = np.array([source.rows for source in sources], dtype=np.float64)
        self.weights = counts / counts.sum()
        self.draws = allocate_draws(self.weights, self.batch_size)
        self.epoch_weights = self.weights.copy()

        starved = [
            source.sample_id
            for source, draw in zip(sources, self.draws, strict=True)
            if draw == 0
        ]
        if starved:
            raise ValueError(
                f"batch_size={batch_size} allocates zero rows to {len(starved)} sample(s) "
                f"({starved[:3]}{'...' if len(starved) > 3 else ''}). Raise batch_size so "
                f"every sample contributes at least one row per batch."
            )

        # Readers are constructed with epoch_shards=1; apply the real sharding
        # before anyone iterates, so epoch 0 reads its stride rather than the
        # whole file.
        self.set_epoch(0)
        self.last_report = self.describe()

        # Sharding too aggressively leaves each file a handful of row groups per
        # epoch. Composition is still corrected by _rebalance, but with few groups
        # the readers' exhaustion points scatter (the spread falls like
        # 1/sqrt(groups per file per epoch)), so the epoch ends with a long tail of
        # short batches. Warn rather than refuse: it is a quality knob, not a
        # correctness one, and a deliberately tiny smoke run is legitimate.
        fewest = min(reader.planned_row_groups for reader in self.readers)
        if self.epoch_shards > 1 and fewest < MIN_ROW_GROUPS_PER_EPOCH:
            print(
                f"[multi_parquet] epoch_shards={self.epoch_shards} leaves as few as "
                f"{fewest} row groups per sample per epoch (< {MIN_ROW_GROUPS_PER_EPOCH}); "
                f"expect a long tail of short batches. Lower epoch_shards for a "
                f"cleaner epoch.",
                file=sys.stderr,
                flush=True,
            )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        for reader in self.readers:
            reader.set_epoch(self._epoch, epoch_shards=self.epoch_shards)
        self._rebalance()

    def _rebalance(self) -> None:
        """Set this epoch's per-file draws from the rows actually available now.

        Weighting by each file's *whole-file* row count is wrong for a sharded
        epoch: epoch e holds only the stride ``order[e::E]`` of a file's row
        groups, and those strides do not hold exactly ``1/E`` of every file.  The
        mismatch makes readers exhaust at different times, which leaves a tail of
        short batches over-representing whoever still has rows -- measured at 15%
        of the epoch with only ~4 row groups per file per epoch.

        Recomputing the draws from what each reader actually holds this epoch
        keeps batches proportional to the available data and lets the readers run
        out together.  Cumulative composition across a full cycle of E epochs is
        still exactly the true proportion, because a cycle covers every row group
        exactly once.
        """
        available = np.array([reader.planned_rows for reader in self.readers], dtype=np.float64)
        if available.sum() <= 0:
            return
        self.epoch_weights = available / available.sum()
        self.draws = allocate_draws(self.epoch_weights, self.batch_size)
        if int(self.draws.min()) < 1:
            starved = [
                reader.source.sample_id
                for reader, draw in zip(self.readers, self.draws, strict=True)
                if draw < 1
            ]
            raise ValueError(
                f"epoch {self._epoch} allocates zero rows to {starved}; those samples "
                f"would not be read at all this epoch. Raise batch_size or lower "
                f"epoch_shards."
            )

    def cap_rows(self, max_rows: int, keep_fraction: float) -> dict[str, int]:
        """Restrict each reader to enough row groups to yield ~``max_rows`` in total.

        This exists for the validation split.  A site-keyed split scatters
        validation rows across every row group, so reading "the validation set"
        means touching the whole dataset -- at 10% of 4.75 billion rows that is
        475 million rows per epoch, which would cost *more* than a sharded
        training epoch.  Capping to a fixed, randomly-spread subset of row groups
        keeps validation proportional across samples and identical from epoch to
        epoch, which is what early stopping needs, at a fraction of the I/O.

        Returns the per-sample row-group limits it applied.
        """
        if max_rows <= 0:
            return {}
        limits: dict[str, int] = {}
        for reader, weight in zip(self.readers, self.weights, strict=True):
            rows_in_split = reader.source.rows * keep_fraction
            per_group = rows_in_split / max(1, reader.num_row_groups)
            if per_group <= 0:
                continue
            wanted = int(np.ceil(max_rows * weight / per_group))
            limits[reader.source.sample_id] = int(
                np.clip(wanted, 1, reader.num_row_groups)
            )
            reader.row_group_limit = limits[reader.source.sample_id]
        self.set_epoch(self._epoch)
        return limits

    def describe(self) -> InterleaveReport:
        return InterleaveReport(
            sample_ids=[s.sample_id for s in self.sources],
            rows_per_sample={s.sample_id: s.rows for s in self.sources},
            weights={
                s.sample_id: float(w) for s, w in zip(self.sources, self.weights, strict=True)
            },
            draws_per_batch={
                s.sample_id: int(d) for s, d in zip(self.sources, self.draws, strict=True)
            },
            epoch_weights={
                s.sample_id: float(w)
                for s, w in zip(self.sources, self.epoch_weights, strict=True)
            },
            row_groups_this_epoch={
                r.source.sample_id: r.planned_row_groups for r in self.readers
            },
        )

    def __iter__(self):
        # Restart every reader, so the source can be iterated more than once.
        # The validation loader depends on this: it is iterated once per epoch but
        # its plan never changes, and without the reset the second epoch would
        # find every reader exhausted and raise "No batches were produced".
        for reader in self.readers:
            reader.set_epoch(self._epoch, epoch_shards=self.epoch_shards)

        report = self.describe()
        while True:
            parts_cat, parts_num, parts_mask = [], [], []
            rows = 0
            for reader, draw in zip(self.readers, self.draws, strict=True):
                taken = reader.take(int(draw))
                if taken is None:
                    continue
                parts_cat.append(taken[0])
                parts_num.append(taken[1])
                parts_mask.append(taken[2])
                rows += taken[0].shape[0]

            if rows == 0:
                break

            cat = np.concatenate(parts_cat)
            num = np.concatenate(parts_num)
            mask = np.concatenate(parts_mask)

            # The batch is assembled sample-block by sample-block, so without this
            # the rows arrive grouped by sample. Nothing downstream depends on row
            # order within a batch, but a grouped batch makes any per-sample effect
            # look like a within-batch trend to anyone inspecting one.
            perm = np.random.default_rng(
                stable_seed(self.split, self._epoch, report.batches)
            ).permutation(rows)

            report.batches += 1
            report.rows_emitted += rows
            if rows == self.batch_size:
                report.full_batches += 1
            else:
                report.ragged_batches += 1
                report.ragged_rows += rows

            yield cat[perm], num[perm], mask[perm]

        for reader in self.readers:
            report.rows_served[reader.source.sample_id] = reader.rows_served
        self.last_report = report

    def close(self) -> None:
        for reader in self.readers:
            reader.close()
