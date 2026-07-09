from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_FEATURE_SPEC_PATH = "ml_features.json"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: str
    values: dict[str, int] | None = None

    @property
    def is_categorical(self) -> bool:
        return self.kind == "c"

    @property
    def is_numeric(self) -> bool:
        return self.kind in {"int", "float"}

    @property
    def null_token(self) -> str:
        if not self.values:
            return ""
        return "" if "" in self.values else next(iter(self.values))

    @property
    def cardinality(self) -> int:
        return len(self.values or {})


def resolve_feature_spec_path(path: str | Path) -> Path:
    requested = Path(path)
    if requested.is_absolute():
        candidate = requested
    else:
        cwd_candidate = Path.cwd() / requested
        candidate = cwd_candidate if cwd_candidate.exists() else requested
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"Unable to resolve feature spec path {path!r}")


def load_feature_specs(path: str | Path) -> list[FeatureSpec]:
    payload = json.loads(resolve_feature_spec_path(path).read_text())
    features = payload["features"]
    return [
        FeatureSpec(
            name=item["name"],
            kind=item["type"],
            values=item.get("values"),
        )
        for item in features
    ]
