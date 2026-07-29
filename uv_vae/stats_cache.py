"""Per-file, cached normalisation statistics that combine analytically.

Two problems, one module.

**Repeated scans.**  ``streaming.py`` makes three complete passes over the data
before the first gradient step -- ``get_row_count``, ``get_non_null_counts`` and
``compute_streaming_stats`` are the same scan with different aggregates.  Over
4.75 billion rows that is three passes across roughly a terabyte of parquet.
They fit in one query.

**Rescanning on every run.**  ``compute_streaming_stats`` writes its output into a
freshly timestamped run directory (``mkdir(exist_ok=False)``), so the numbers are
persisted but never read back except by checkpoints, post-training.  Every run
recomputes them.

Statistics here are stored **per file** and combined analytically, so adding a
96th sample costs one scan instead of 96, and a re-run costs none.

Combination
-----------

Files are combined with the Chan-Golub-LeVeque pairwise update rather than by
accumulating ``sum(x^2)``, which loses precision catastrophically when the mean
is large relative to the spread::

    delta = mu_B - mu_A
    n     = n_A + n_B
    mu    = mu_A + delta * n_B / n
    M2    = M2_A + M2_B + delta^2 * n_A * n_B / n

(Chan, Golub & LeVeque 1983, *The American Statistician* 37(3):242-247; see also
Welford 1962 and Pebay, SAND2008-6212.)

Two details matter as much as the formula:

* **Each column carries its own n** -- its non-null count, not the file's row
  count, because SQL aggregates skip NULLs.  ``ml_features.json`` lists columns
  that are 100% null on the production featuremap, so weighting by row count
  would misweight every column that has any nulls at all.
* **Files combine as a balanced binary tree**, not a left fold.  Chan et al. show
  the update is least accurate when the two partitions differ greatly in size,
  which is exactly what a fold produces once it has accumulated a few files.

The all-null drop uses counts summed across **all** files, so a feature that is
null in one sample but populated in another is kept.  Deriving the model shape
from a single file would build the wrong network.

Scope note
----------

Statistics are computed over all filtered rows, **not** over training rows only
-- matching ``streaming.compute_streaming_stats`` exactly, so a multi-file run
stays numerically comparable to a single-file one.  It also keeps the cache
independent of the train/val split, so sweeping ``val_fraction`` does not
invalidate it.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from uv_vae.data import connect_duckdb, quote_ident, split_specs
from uv_vae.features import FeatureSpec

CACHE_VERSION = 1


@dataclass(frozen=True)
class ColumnMoments:
    """Count, mean and sum-of-squared-deviations for one numeric column."""

    n: int
    mean: float
    m2: float

    @property
    def variance(self) -> float:
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)


def combine_moments(a: ColumnMoments, b: ColumnMoments) -> ColumnMoments:
    """Chan-Golub-LeVeque pairwise combination of two partitions."""
    if a.n == 0:
        return b
    if b.n == 0:
        return a
    n = a.n + b.n
    delta = b.mean - a.mean
    mean = a.mean + delta * (b.n / n)
    m2 = a.m2 + b.m2 + delta * delta * (a.n * b.n / n)
    return ColumnMoments(n=n, mean=mean, m2=m2)


def combine_moment_tree(parts: list[ColumnMoments]) -> ColumnMoments:
    """Combine many partitions as a balanced binary tree.

    A left fold would repeatedly merge a large running accumulator with one small
    file, which is precisely the regime where the pairwise update is least
    accurate.  The 95 featuremaps have very unequal row counts, so this is not
    hypothetical.
    """
    if not parts:
        return ColumnMoments(n=0, mean=0.0, m2=0.0)
    level = list(parts)
    while len(level) > 1:
        level = [
            combine_moments(level[i], level[i + 1]) if i + 1 < len(level) else level[i]
            for i in range(0, len(level), 2)
        ]
    return level[0]


@dataclass(frozen=True)
class FileStats:
    """Everything one parquet file contributes, cached on disk."""

    sample_id: str
    path: str
    rows: int
    non_null_counts: dict[str, int]
    moments: dict[str, ColumnMoments]

    def to_json(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "path": self.path,
            "rows": self.rows,
            "non_null_counts": self.non_null_counts,
            "moments": {k: asdict(v) for k, v in self.moments.items()},
        }

    @classmethod
    def from_json(cls, payload: dict) -> "FileStats":
        return cls(
            sample_id=payload["sample_id"],
            path=payload["path"],
            rows=int(payload["rows"]),
            non_null_counts={k: int(v) for k, v in payload["non_null_counts"].items()},
            moments={
                k: ColumnMoments(n=int(v["n"]), mean=float(v["mean"]), m2=float(v["m2"]))
                for k, v in payload["moments"].items()
            },
        )


@dataclass
class MultiFileStats:
    """Combined statistics and the derived model shape."""

    per_file: list[FileStats]
    total_rows: int
    non_null_counts: dict[str, int]
    numeric_means: dict[str, float]
    numeric_stds: dict[str, float]
    categorical_specs: list[FeatureSpec]
    numeric_specs: list[FeatureSpec]
    dropped_all_null_features: list[str] = field(default_factory=list)

    @property
    def rows_by_sample(self) -> dict[str, int]:
        return {stats.sample_id: stats.rows for stats in self.per_file}


def spec_fingerprint(feature_specs: list[FeatureSpec]) -> str:
    payload = json.dumps(
        [[s.name, s.kind, sorted((s.values or {}).items())] for s in feature_specs],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cache_key(path: str | Path, row_filter: str | None, fingerprint: str) -> str:
    """Identity of one file's statistics.

    Includes size and mtime so a regenerated or truncated parquet is not served
    from a stale entry, and the row filter and feature fingerprint because both
    change the answer.
    """
    resolved = Path(path).resolve()
    stat = resolved.stat()
    material = "|".join(
        [
            str(CACHE_VERSION),
            str(resolved),
            str(stat.st_size),
            str(stat.st_mtime_ns),
            row_filter or "",
            fingerprint,
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def compute_file_stats(
    path: str | Path,
    sample_id: str,
    feature_specs: list[FeatureSpec],
    row_filter: str | None = None,
    threads: int | None = None,
) -> FileStats:
    """One DuckDB pass yielding row count, non-null counts and numeric moments."""
    numeric = [spec for spec in feature_specs if spec.is_numeric]

    selects = ["count(*) AS __rows"]
    for spec in feature_specs:
        selects.append(f"count({quote_ident(spec.name)}) AS {quote_ident('nn_' + spec.name)}")
    for spec in numeric:
        column = f"CAST({quote_ident(spec.name)} AS DOUBLE)"
        selects.append(f"avg({column}) AS {quote_ident('avg_' + spec.name)}")
        selects.append(f"var_samp({column}) AS {quote_ident('var_' + spec.name)}")

    sql = f"SELECT {', '.join(selects)} FROM read_parquet(?)"
    if row_filter:
        sql += f" WHERE {row_filter}"

    with connect_duckdb(threads=threads) as conn:
        row = conn.execute(sql, [str(path)]).fetchone()
    if row is None:
        raise RuntimeError(f"DuckDB returned no statistics row for {path}")

    cursor = 0
    rows = int(row[cursor])
    cursor += 1

    non_null_counts: dict[str, int] = {}
    for spec in feature_specs:
        non_null_counts[spec.name] = int(row[cursor] or 0)
        cursor += 1

    moments: dict[str, ColumnMoments] = {}
    for spec in numeric:
        mean_value = row[cursor]
        var_value = row[cursor + 1]
        cursor += 2
        n = non_null_counts[spec.name]
        if n == 0 or mean_value is None or not math.isfinite(float(mean_value)):
            moments[spec.name] = ColumnMoments(n=0, mean=0.0, m2=0.0)
            continue
        # var_samp is NULL for n == 1 and for constant columns it is 0.
        variance = float(var_value) if var_value is not None else 0.0
        if not math.isfinite(variance) or variance < 0.0:
            variance = 0.0
        moments[spec.name] = ColumnMoments(
            n=n, mean=float(mean_value), m2=variance * max(0, n - 1)
        )

    return FileStats(
        sample_id=sample_id,
        path=str(Path(path).resolve()),
        rows=rows,
        non_null_counts=non_null_counts,
        moments=moments,
    )


def _read_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    try:
        payload = json.loads(cache_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if payload.get("cache_version") != CACHE_VERSION:
        return {}
    return payload.get("entries", {})


def _write_cache(cache_path: Path, entries: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"cache_version": CACHE_VERSION, "entries": entries}, indent=2)
    )
    temporary.replace(cache_path)


def sample_id_for(path: str | Path) -> str:
    """Sample identifier derived from the filename.

    It is the salt for the per-sample split, so it must be stable across runs and
    machines -- hence the filename rather than an enumeration index, which would
    shift if a file were added or the glob order changed.
    """
    return Path(path).name.split(".")[0]


def load_or_compute_stats(
    paths: list[str | Path],
    feature_specs: list[FeatureSpec],
    row_filter: str | None = None,
    cache_path: str | Path | None = None,
    threads: int | None = None,
    verbose: bool = True,
) -> MultiFileStats:
    """Per-file statistics for every path, from cache where possible."""
    if not paths:
        raise ValueError("load_or_compute_stats needs at least one parquet path")

    fingerprint = spec_fingerprint(feature_specs)
    cache_file = Path(cache_path) if cache_path else None
    entries = _read_cache(cache_file) if cache_file else {}
    dirty = False

    per_file: list[FileStats] = []
    for index, path in enumerate(paths, start=1):
        sample_id = sample_id_for(path)
        key = cache_key(path, row_filter, fingerprint)
        cached = entries.get(key)
        if cached is not None:
            per_file.append(FileStats.from_json(cached))
            if verbose:
                print(
                    f"[stats] {index}/{len(paths)} {sample_id}: cached "
                    f"({per_file[-1].rows:,} rows)",
                    file=sys.stderr,
                    flush=True,
                )
            continue

        if verbose:
            print(
                f"[stats] {index}/{len(paths)} {sample_id}: scanning ...",
                file=sys.stderr,
                flush=True,
            )
        stats = compute_file_stats(
            path=path,
            sample_id=sample_id,
            feature_specs=feature_specs,
            row_filter=row_filter,
            threads=threads,
        )
        if stats.rows == 0:
            raise RuntimeError(
                f"No rows in {path} matched the row filter {row_filter!r}. A sample "
                f"contributing zero rows cannot be interleaved."
            )
        per_file.append(stats)
        entries[key] = stats.to_json()
        dirty = True
        if verbose:
            print(
                f"[stats] {index}/{len(paths)} {sample_id}: {stats.rows:,} rows",
                file=sys.stderr,
                flush=True,
            )

    if cache_file and dirty:
        _write_cache(cache_file, entries)

    return combine_file_stats(per_file, feature_specs)


def combine_file_stats(
    per_file: list[FileStats], feature_specs: list[FeatureSpec]
) -> MultiFileStats:
    """Fold per-file statistics into the cohort-wide numbers and model shape."""
    if not per_file:
        raise ValueError("combine_file_stats needs at least one FileStats")

    non_null_counts: dict[str, int] = {}
    for spec in feature_specs:
        non_null_counts[spec.name] = sum(
            stats.non_null_counts.get(spec.name, 0) for stats in per_file
        )

    # split_specs drops columns whose SUMMED non-null count is zero, so a feature
    # populated in even one sample survives.
    categorical_specs, numeric_specs, dropped = split_specs(feature_specs, non_null_counts)

    numeric_means: dict[str, float] = {}
    numeric_stds: dict[str, float] = {}
    for spec in numeric_specs:
        combined = combine_moment_tree(
            [
                stats.moments.get(spec.name, ColumnMoments(0, 0.0, 0.0))
                for stats in per_file
            ]
        )
        mean = combined.mean if math.isfinite(combined.mean) else 0.0
        std = combined.std
        # Matches streaming.compute_streaming_stats: a degenerate spread becomes 1.0
        # so normalisation is a no-op rather than a division by ~0.
        if not math.isfinite(std) or std < 1e-6:
            std = 1.0
        numeric_means[spec.name] = float(mean)
        numeric_stds[spec.name] = float(std)

    return MultiFileStats(
        per_file=per_file,
        total_rows=sum(stats.rows for stats in per_file),
        non_null_counts=non_null_counts,
        numeric_means=numeric_means,
        numeric_stds=numeric_stds,
        categorical_specs=categorical_specs,
        numeric_specs=numeric_specs,
        dropped_all_null_features=sorted(dropped),
    )
