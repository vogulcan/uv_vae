"""Optional GPU decode path for the interleaved multi-parquet reader.

``multi_parquet.ParquetSampleReader`` decodes a row group with pyarrow, filters
it with polars, hashes the split key with numpy and encodes with polars -- all on
the CPU.  On the 95-file cohort that leaves the GPU at ~28% utilisation: it
finishes a batch and waits for the next one to be decoded.  This module is the
same four steps against cuDF/cuPy so the work lands on the GPU instead.

What moves and what does not
----------------------------

*Parquet read* moves.  ``cudf.read_parquet(row_groups=[i])`` decompresses on the
GPU through nvCOMP, which is the part of a row group's cost that CPU threads
cannot avoid.  Row-group addressing is preserved exactly, so the shuffled visit
order that breaks the genomic clustering is unchanged.

*Row filter* moves only when it fits a deliberately small grammar -- see
:func:`compile_filter`.  ``--row-filter`` takes arbitrary SQL, and cuDF has no
SQL evaluator; a hand-rolled translator that silently mis-read a predicate would
change which rows train, which is far worse than decoding slowly.  So anything
outside the grammar raises :class:`UnsupportedFilter` and the caller falls back
to the CPU path for that reader.

*Encode and split hash* move.  Both must be **bit-identical** to the CPU path,
for different reasons: the encoder's output is pinned against
``preprocess.encode_categorical_column`` by
``test_encoder_matches_preprocess_helpers``, and the split hash decides which
rows are validation rows -- a one-bit difference silently re-partitions the data
and makes the run incomparable to every earlier one.  ``test_gpu_decode.py``
asserts exact equality of all three arrays and of the split mask, and is the
gate on enabling this path.

Bit-exactness notes
-------------------

*Integer hashing* is exact by construction: SplitMix64 is uint64 add/xor/shift/
multiply, all of which wrap mod 2**64 identically in cuPy and numpy.

*String hashing stays on the CPU.*  :func:`splitting.fnv1a64` runs over the
DISTINCT values of CHROM/REF/ALT -- ~25 and ~5 of them -- so it is a couple of
dozen Python calls per row group either way.  Running it on the host and
gathering the codes on the device means the code assigned to a given string
cannot differ between the two paths, rather than merely being expected not to.

*Float standardisation* is elementwise subtract then divide in float32, issued as
separate kernels, so no fused-multiply-add contracts the two into one
differently-rounded operation.
"""

from __future__ import annotations

import re

import numpy as np

from uv_vae.features import FeatureSpec
from uv_vae.splitting import (
    PER_SAMPLE_ROW_HASH,
    SITE_KEY_COLUMNS,
    SplitConfig,
    fnv1a64,
)

_GOLDEN_GAMMA = 0x9E3779B97F4A7C15
_MIX_A = 0xBF58476D1CE4E5B9
_MIX_B = 0x94D049BB133111EB
_HASH_INIT = 0x243F6A8885A308D3


class UnsupportedFilter(ValueError):
    """A row filter this module refuses to translate; caller falls back to CPU."""


def gpu_modules():
    """Return ``(cudf, cupy)`` or ``(None, None)`` when either is unavailable.

    Imported lazily and never at module scope: ``multi_parquet`` is importable
    (and tested) on machines with no GPU at all, and this module is imported from
    it.
    """
    try:
        import cudf
        import cupy
    except Exception:  # pragma: no cover - depends on the host
        return None, None
    return cudf, cupy


def gpu_decode_available() -> bool:
    return gpu_modules()[0] is not None


# ---------------------------------------------------------------------------
# row filter
# ---------------------------------------------------------------------------

# Deliberately tiny: AND-separated comparisons against a literal, nothing else.
# The production default -- "st = 'MIXED' AND et = 'MIXED' AND FILT = 1" -- is
# inside it. OR, parentheses, arithmetic, IN and function calls are all outside,
# and land on the CPU path rather than being approximated.
_TERM = re.compile(
    r"""^\s*
    (?P<column>[A-Za-z_][A-Za-z0-9_]*)\s*
    (?:
        (?P<isnull>IS\s+(?P<not>NOT\s+)?NULL)
      | (?P<op><=|>=|!=|<>|=|<|>)\s*
        (?:'(?P<text>[^']*)'|(?P<number>-?\d+(?:\.\d+)?))
    )
    \s*$""",
    re.IGNORECASE | re.VERBOSE,
)


def _split_conjunction(sql: str) -> list[str]:
    """Split on top-level AND, refusing anything that could nest or disjoin."""
    if "(" in sql or ")" in sql:
        raise UnsupportedFilter("parenthesised filters are not translated")
    if re.search(r"\bOR\b", sql, re.IGNORECASE):
        raise UnsupportedFilter("OR is not translated")
    parts = re.split(r"\bAND\b", sql, flags=re.IGNORECASE)
    if not parts:
        raise UnsupportedFilter("empty filter")
    return parts


def compile_filter(sql: str | None, available: set[str]):
    """Build ``mask(cudf_frame) -> boolean cudf.Series`` for ``sql``.

    Raises :class:`UnsupportedFilter` for anything outside the grammar, which the
    caller must treat as "decode this reader on the CPU" -- never as "no filter".
    """
    if not sql or not sql.strip():
        return None

    terms = []
    for raw in _split_conjunction(sql):
        match = _TERM.match(raw)
        if match is None:
            raise UnsupportedFilter(f"cannot translate predicate {raw.strip()!r}")
        column = match.group("column")
        if column not in available:
            raise UnsupportedFilter(f"filter references unknown column {column!r}")
        if match.group("isnull"):
            terms.append((column, "notnull" if match.group("not") else "isnull", None))
            continue
        op = match.group("op")
        if match.group("text") is not None:
            value: object = match.group("text")
        else:
            literal = match.group("number")
            value = float(literal) if "." in literal else int(literal)
        terms.append((column, "<>" if op == "!=" else op, value))

    def mask(frame):
        result = None
        for column, op, value in terms:
            series = frame[column]
            if op == "isnull":
                current = series.isnull()
            elif op == "notnull":
                current = series.notnull()
            elif op == "=":
                current = series == value
            elif op == "<>":
                current = series != value
            elif op == "<":
                current = series < value
            elif op == "<=":
                current = series <= value
            elif op == ">":
                current = series > value
            else:
                current = series >= value
            # A null compares false, matching SQL three-valued logic collapsing
            # to "not selected" -- and matching what polars' SQL filter does.
            current = current.fillna(False)
            result = current if result is None else (result & current)
        return result

    return mask


# ---------------------------------------------------------------------------
# hashing
# ---------------------------------------------------------------------------


def _splitmix64_gpu(state, cupy):
    """SplitMix64 finaliser over a cuPy uint64 array.

    Mirrors :func:`splitting.splitmix64` operation for operation. uint64
    arithmetic wraps mod 2**64 in cuPy exactly as it does in numpy, which is the
    modular arithmetic SplitMix64 is defined in.
    """
    u64 = cupy.uint64
    state = state + u64(_GOLDEN_GAMMA)
    state = (state ^ (state >> u64(30))) * u64(_MIX_A)
    state = (state ^ (state >> u64(27))) * u64(_MIX_B)
    return state ^ (state >> u64(31))


def _mix_keys_gpu(columns, cupy):
    length = columns[0].size
    state = cupy.full(length, cupy.uint64(_HASH_INIT), dtype=cupy.uint64)
    for column in columns:
        state = _splitmix64_gpu(state ^ column.astype(cupy.uint64), cupy)
    return state


def _string_codes_gpu(series, cupy, default: int = 0):
    """uint64 code per row, hashing each DISTINCT string once on the HOST.

    Mirrors :func:`splitting.string_column_codes`. The categorical round-trip
    gives a dense integer code per row plus the category list in one pass; the
    gather then maps those to FNV-1a values computed by the same host function
    the CPU path uses, so identical strings cannot get different codes.
    """
    text = series.astype("str").fillna("")
    categorical = text.astype("category")
    categories = categorical.cat.categories.to_arrow().to_pylist()
    table = cupy.asarray(
        [fnv1a64(value) if value is not None else default for value in categories],
        dtype=cupy.uint64,
    )
    codes = categorical.cat.codes.values.astype(cupy.int64)
    if table.size == 0:
        return cupy.zeros(text.size, dtype=cupy.uint64)
    # cuDF uses -1 for "not in categories"; there is none after fillna, but clip
    # rather than trust that, so a stray -1 cannot wrap into a huge gather index.
    return table[cupy.clip(codes, 0, table.size - 1)]


def gpu_split_mask(
    config: SplitConfig,
    sample_id: str,
    split: str,
    frame,
    row_positions=None,
):
    """Boolean cuPy mask selecting ``split``'s rows -- bit-identical to CPU.

    ``frame`` is a cuDF DataFrame carrying :data:`splitting.SITE_KEY_COLUMNS` for
    the site strategies. Returns a cuPy bool array, not a cuDF Series, so the
    caller can index numpy-style.
    """
    cudf, cupy = gpu_modules()
    if cudf is None:
        raise RuntimeError("gpu_split_mask needs cudf and cupy")
    if split not in {"train", "val"}:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    salt = cupy.uint64(config.salt(sample_id))
    height = len(frame)

    if config.strategy == PER_SAMPLE_ROW_HASH:
        if row_positions is None:
            raise ValueError("per_sample_row_hash needs row_positions")
        positions = cupy.asarray(np.asarray(row_positions, dtype=np.uint64))
        hashes = _mix_keys_gpu(
            [cupy.full(positions.size, salt, dtype=cupy.uint64), positions], cupy
        )
    else:
        missing = [c for c in SITE_KEY_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"{config.strategy} needs columns {missing}")
        pos = frame["POS"].astype("int64").fillna(0).values.astype(cupy.uint64)
        hashes = _mix_keys_gpu(
            [
                cupy.full(height, salt, dtype=cupy.uint64),
                _string_codes_gpu(frame["CHROM"], cupy),
                pos,
                _string_codes_gpu(frame["REF"], cupy),
                _string_codes_gpu(frame["ALT"], cupy),
            ],
            cupy,
        )

    is_val = hashes < cupy.uint64(config.threshold)
    return is_val if split == "val" else ~is_val


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------


class GpuRowEncoder:
    """cuDF counterpart of :class:`multi_parquet.RowEncoder`.

    Produces arrays bit-identical to the CPU encoder. Returns cuPy arrays; the
    caller decides whether to keep them on the device or copy them back.
    """

    def __init__(
        self,
        categorical_specs: list[FeatureSpec],
        numeric_specs: list[FeatureSpec],
        numeric_means: dict[str, float],
        numeric_stds: dict[str, float],
    ) -> None:
        cudf, cupy = gpu_modules()
        if cudf is None:
            raise RuntimeError("GpuRowEncoder needs cudf and cupy")
        self._cupy = cupy
        self.categorical_specs = categorical_specs
        self.numeric_specs = numeric_specs
        self.means = cupy.asarray(
            np.array([numeric_means[s.name] for s in numeric_specs], dtype=np.float32)
        )
        self.stds = cupy.asarray(
            np.array([numeric_stds[s.name] for s in numeric_specs], dtype=np.float32)
        )
        self._cat_lookup = []
        for spec in categorical_specs:
            if spec.values is None:
                raise ValueError(f"Categorical feature {spec.name} has no category map")
            self._cat_lookup.append(
                (
                    spec.name,
                    dict(spec.values),
                    spec.null_token,
                    spec.values.get(spec.null_token, 0),
                )
            )

    @property
    def feature_names(self) -> list[str]:
        return [s.name for s in self.categorical_specs] + [
            s.name for s in self.numeric_specs
        ]

    def encode(self, frame):
        cupy = self._cupy
        height = len(frame)

        if self._cat_lookup:
            columns = []
            for name, lookup, null_token, null_index in self._cat_lookup:
                text = frame[name].astype("str").fillna(null_token)
                categorical = text.astype("category")
                categories = categorical.cat.categories.to_arrow().to_pylist()
                table = cupy.asarray(
                    np.array(
                        [lookup.get(value, null_index) for value in categories],
                        dtype=np.int64,
                    )
                )
                codes = categorical.cat.codes.values.astype(cupy.int64)
                if table.size == 0:
                    columns.append(cupy.full(height, null_index, dtype=cupy.int64))
                else:
                    columns.append(table[cupy.clip(codes, 0, table.size - 1)])
            categorical_out = cupy.stack(columns, axis=1).astype(cupy.int64, copy=False)
        else:
            categorical_out = cupy.zeros((height, 0), dtype=cupy.int64)

        if self.numeric_specs:
            values = []
            masks = []
            for spec in self.numeric_specs:
                column = (
                    frame[spec.name]
                    .astype("float32")
                    .fillna(float("nan"))
                    .values.astype(cupy.float32)
                )
                values.append(column)
                masks.append((~cupy.isnan(column)).astype(cupy.float32))
            numeric = cupy.stack(values, axis=1)
            mask = cupy.stack(masks, axis=1)
            numeric = cupy.where(mask > 0, numeric, self.means)
            numeric = (numeric - self.means) / self.stds
        else:
            numeric = cupy.zeros((height, 0), dtype=cupy.float32)
            mask = cupy.zeros((height, 0), dtype=cupy.float32)

        return categorical_out, numeric.astype(cupy.float32, copy=False), mask
