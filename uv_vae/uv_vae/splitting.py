"""Content-hash train/val split predicates for multi-file interleaved training.

``streaming.py`` splits train/val by a row's POSITION in the scan
(``row_index % val_denominator == seed % val_denominator``).  That works for a
single sequential scan and nothing else.  Interleaving 95 files and reading each
one's row groups in shuffled order destroys any meaningful global scan position,
so the positional split cannot survive it.

Three further properties of the positional split are worth stating, because they
motivate the replacement rather than just an adaptation of it:

1. **It admits only ``val_denominator`` distinct splits** -- 10 at
   ``train_fraction=0.9``, because the remainder is ``seed % 10``.  Seeds 42, 52
   and 62 select the byte-identical validation set, and every script in this
   repository pins ``SEED=42``.  Split sensitivity is therefore unmeasurable.
2. **It requires ``preserve_insertion_order=true``** and two scans that agree,
   or a row lands in both splits.
3. ``train_fraction`` quantises to ``1/round(1/(1-f))``.

The replacement hashes a row *key* instead.  The result depends only on the
contents of the key, so it is invariant to read order, it survives interleaving
and row-group shuffling unchanged, it offers 2**64 distinct splits, and
``val_fraction`` is continuous.

Strategies
----------

``per_sample_row_hash``
    Row-level, independent per row, salted per sample.  The key is the row's
    *physical* address in the file -- ``(row_group_index, offset_within_group)``
    -- which is a stable row identifier that costs nothing to read and does not
    depend on the order row groups are visited.  This is the cheapest option and
    it is what the positional split was reaching for.

    **It leaks at the locus level.**  These featuremaps hold single-read SNVs at
    ~8-9 reads per site, and reads at one site are physically adjacent, so an
    independent per-row draw puts most sites on both sides of the split.  A site
    with 30 reads puts ~27 in train and ~3 in val, and those rows share ``REF``,
    ``ALT`` and ``X_PREV1..3``/``X_NEXT1..3`` -- 8 of the 11 categorical
    features.  Validation loss then reads optimistically low, and because early
    stopping watches validation ELBO it stops *later* than it should.  That is a
    directional bias, not noise.

``per_sample_site_hash``
    Site-level: the key is ``(CHROM, POS, REF, ALT)``, salted per sample.  Every
    read at a locus lands on the same side *within* a sample.

    **It still leaks across samples.**  All 95 featuremaps are human genomes and
    share loci, so a per-sample salt puts a given site in validation for sample 3
    and in training for sample 60 — and the model has then already seen that
    locus's ``REF``/``ALT``/``X_PREV1..3``/``X_NEXT1..3``.  Use this only when the
    question really is per-sample.

``global_site_hash`` (default)
    Same key, *unsalted*, so a locus lands on the same side in **every** sample.
    This is the only clean locus holdout across the cohort, which is why it is the
    default: early stopping watches validation ELBO, and any leak biases that
    signal in one direction — it stops later than it should, contaminating the
    epoch count, which is itself one of the axes this project measures.

None of these hold out whole *samples*.  Holding out samples answers a different
scientific question ("does this generalise to an unseen individual?") and needs a
file-level split, not a row predicate -- see SAMPLING_STRATEGY.md section 9.

Hash stability
--------------

SplitMix64 (Steele, Lea & Flood 2014, OOPSLA) is implemented here with every
constant pinned in this file, rather than calling DuckDB's ``hash()`` or polars'
``hash_rows()``.  Neither of those guarantees stability across versions, and a
train/val split that silently changes when a dependency is upgraded would
invalidate every comparison across runs.

Salting folds into the hash *input*, never onto its output.  ``hash(k, A) XOR
hash(k, B)`` is a constant offset independent of ``k``, so XOR-salting produces
draws that are structured across salts rather than independent -- two salts can
even give mutually exclusive splits.  ``test_salts_produce_independent_draws``
pins the correct behaviour.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import polars as pl

MASK64 = (1 << 64) - 1

# SplitMix64 constants (Steele et al. 2014).
_GOLDEN_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_MIX_A = np.uint64(0xBF58476D1CE4E5B9)
_MIX_B = np.uint64(0x94D049BB133111EB)
_SHIFT_30 = np.uint64(30)
_SHIFT_27 = np.uint64(27)
_SHIFT_31 = np.uint64(31)

# FNV-1a 64-bit, for turning short strings (CHROM, REF, ALT) into integers.
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3

# Arbitrary non-zero start state so an all-zero key does not hash to a fixed point.
_HASH_INIT = np.uint64(0x243F6A8885A308D3)

PER_SAMPLE_ROW_HASH = "per_sample_row_hash"
PER_SAMPLE_SITE_HASH = "per_sample_site_hash"
GLOBAL_SITE_HASH = "global_site_hash"
STRATEGIES = (PER_SAMPLE_ROW_HASH, PER_SAMPLE_SITE_HASH, GLOBAL_SITE_HASH)

# Columns a site-keyed split must read on top of the feature columns.
SITE_KEY_COLUMNS = ("CHROM", "POS", "REF", "ALT")


def stable_seed(*parts: object) -> int:
    """A 32-bit seed from arbitrary parts, reproducible ACROSS PROCESSES.

    Python's builtin ``hash()`` is randomised per process unless ``PYTHONHASHSEED``
    is set *before* the interpreter starts.  ``training.seed_everything`` sets it
    with ``os.environ.setdefault``, which is far too late to affect the process
    already running -- so seeding a shuffle with ``hash(filename)`` silently
    produces a different order on every run, and the run is not reproducible.
    CRC32 has no such behaviour.
    """
    digest = 0
    for part in parts:
        digest = zlib.crc32(str(part).encode("utf-8"), digest)
    return digest & 0xFFFFFFFF


def fnv1a64(text: str) -> int:
    """64-bit FNV-1a of a string. Called once per DISTINCT value, not per row."""
    digest = _FNV_OFFSET
    for byte in text.encode("utf-8"):
        digest = ((digest ^ byte) * _FNV_PRIME) & MASK64
    return digest


def splitmix64(values: np.ndarray) -> np.ndarray:
    """Vectorised SplitMix64 finaliser over a uint64 array."""
    state = np.asarray(values, dtype=np.uint64).copy()
    # uint64 arithmetic wraps in numpy, which is exactly the modular behaviour
    # SplitMix64 is defined in; errstate keeps it from reporting that as a fault.
    with np.errstate(over="ignore"):
        state = state + _GOLDEN_GAMMA
        state = (state ^ (state >> _SHIFT_30)) * _MIX_A
        state = (state ^ (state >> _SHIFT_27)) * _MIX_B
        state = state ^ (state >> _SHIFT_31)
    return state


def mix_keys(*columns: np.ndarray) -> np.ndarray:
    """Fold several uint64 key columns into one uint64 hash per row.

    Each column is XORed into the running state and the result re-mixed, so the
    output is a non-linear function of every input including the salt.  That is
    what makes two salts give *independent* splits rather than two splits offset
    by a constant.
    """
    if not columns:
        raise ValueError("mix_keys needs at least one column")
    length = len(columns[0])
    state = np.full(length, _HASH_INIT, dtype=np.uint64)
    for column in columns:
        state = splitmix64(state ^ np.asarray(column, dtype=np.uint64))
    return state


def string_column_codes(series: pl.Series) -> np.ndarray:
    """Map a string column to uint64 codes by hashing each DISTINCT value once.

    ``CHROM`` has ~25 distinct values and ``REF``/``ALT`` have ~5, so the Python
    loop in :func:`fnv1a64` runs a couple of dozen times per chunk, not once per
    row.
    """
    text = series.cast(pl.String).fill_null("")
    lookup = {value: fnv1a64(value) for value in text.unique().to_list()}
    return (
        text.replace_strict(lookup, default=0, return_dtype=pl.UInt64)
        .to_numpy()
        .astype(np.uint64, copy=False)
    )


@dataclass(frozen=True)
class SplitConfig:
    """How to divide rows between train and validation.

    ``val_fraction`` is normalised to 10 decimal places on construction.  Callers
    derive it in different ways -- a shell runner may compute ``1 - 0.9`` while
    the CLI passes a literal ``0.1`` -- and in floating point those are *not* the
    same number (``1.0 - 0.9 == 0.09999999999999998``).  Left alone they produce
    different thresholds and therefore genuinely different partitions of the
    data, so a statistics stage and a training stage that derive it differently
    would silently disagree about which rows are validation rows.
    """

    strategy: str = GLOBAL_SITE_HASH
    val_fraction: float = 0.1
    seed: int = 42

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(
                f"Unknown split strategy {self.strategy!r}; expected one of {STRATEGIES}"
            )
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError(
                f"val_fraction must be strictly between 0 and 1, got {self.val_fraction}"
            )
        object.__setattr__(self, "val_fraction", round(float(self.val_fraction), 10))

    @property
    def threshold(self) -> int:
        """Hash values strictly below this go to validation.

        Computed through :class:`~fractions.Fraction` on the decimal literal
        because ``int(0.1 * 2**64)`` rounds the float product and lands on
        1844674407370955264 instead of the exact 1844674407370955161.  The
        difference does not matter to the split fraction; it matters that the
        number is defined by the decimal the user wrote and not by a platform's
        float rounding.
        """
        return int(Fraction(str(self.val_fraction)) * (1 << 64))

    @property
    def needs_site_columns(self) -> bool:
        return self.strategy in {PER_SAMPLE_SITE_HASH, GLOBAL_SITE_HASH}

    @property
    def extra_columns(self) -> tuple[str, ...]:
        """Columns to read beyond the feature columns for this strategy."""
        return SITE_KEY_COLUMNS if self.needs_site_columns else ()

    def salt(self, sample_id: str) -> int:
        """Per-sample salt, or a cohort-wide one for ``global_site_hash``."""
        if self.strategy == GLOBAL_SITE_HASH:
            return stable_seed("global", self.seed)
        return stable_seed(sample_id, self.seed)

    def as_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "val_fraction": self.val_fraction,
            "seed": self.seed,
            "threshold": str(self.threshold),
        }


def row_hash(
    config: SplitConfig,
    sample_id: str,
    frame: pl.DataFrame | None = None,
    row_positions: np.ndarray | None = None,
) -> np.ndarray:
    """The uint64 hash for each row under ``config``.

    ``row_positions`` are the rows' physical addresses in the file, needed only by
    ``per_sample_row_hash``.  ``frame`` must carry :data:`SITE_KEY_COLUMNS` for
    the site strategies.
    """
    salt = np.uint64(config.salt(sample_id))

    if config.strategy == PER_SAMPLE_ROW_HASH:
        if row_positions is None:
            raise ValueError("per_sample_row_hash needs row_positions")
        positions = np.asarray(row_positions, dtype=np.uint64)
        return mix_keys(np.full(len(positions), salt, dtype=np.uint64), positions)

    if frame is None:
        raise ValueError(f"{config.strategy} needs a frame carrying {SITE_KEY_COLUMNS}")
    missing = [name for name in SITE_KEY_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(
            f"{config.strategy} needs columns {missing} which are not in the chunk"
        )

    height = frame.height
    return mix_keys(
        np.full(height, salt, dtype=np.uint64),
        string_column_codes(frame.get_column("CHROM")),
        frame.get_column("POS").cast(pl.Int64).fill_null(0).to_numpy().astype(np.uint64, copy=False),
        string_column_codes(frame.get_column("REF")),
        string_column_codes(frame.get_column("ALT")),
    )


def val_mask(
    config: SplitConfig,
    sample_id: str,
    frame: pl.DataFrame | None = None,
    row_positions: np.ndarray | None = None,
) -> np.ndarray:
    """Boolean mask, True where the row belongs to validation."""
    hashes = row_hash(config, sample_id, frame=frame, row_positions=row_positions)
    return hashes < np.uint64(config.threshold)


def split_mask(
    config: SplitConfig,
    sample_id: str,
    split: str,
    frame: pl.DataFrame | None = None,
    row_positions: np.ndarray | None = None,
) -> np.ndarray:
    """Boolean mask selecting the rows of ``split`` ("train" or "val")."""
    if split not in {"train", "val"}:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    mask = val_mask(config, sample_id, frame=frame, row_positions=row_positions)
    return mask if split == "val" else ~mask
