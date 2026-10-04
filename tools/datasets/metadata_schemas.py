"""Validate catalogue, campaign, and experiment metadata with shared schemas."""

from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
from typing import Iterable

import jsonschema
import yaml


ROOT = Path(__file__).resolve().parents[2]
DATASET_SCHEMA_ROOT = ROOT / "tools" / "datasets" / "schemas"
EXPERIMENT_SCHEMA_ROOT = ROOT / "tools" / "experiments" / "schemas"
SCHEMA_PATHS = {
    "recipe": DATASET_SCHEMA_ROOT / "recipe.schema.yaml",
    "generation": DATASET_SCHEMA_ROOT / "generation.schema.yaml",
    "validation": DATASET_SCHEMA_ROOT / "validation.schema.yaml",
    "campaign": DATASET_SCHEMA_ROOT / "campaign.schema.yaml",
    "experiment": EXPERIMENT_SCHEMA_ROOT / "experiment.schema.yaml",
}


@lru_cache(maxsize=None)
def schema_validator(kind: str) -> jsonschema.Draft202012Validator:
    """Load, check, and cache the validator for one metadata responsibility."""

    try:
        path = SCHEMA_PATHS[kind]
    except KeyError as error:
        raise ValueError(f"Unknown metadata schema: {kind}") from error
    schema = yaml.safe_load(path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def validate_document(document: object, kind: str, source: object = "document") -> None:
    """Reject one document with a stable, path-oriented schema diagnostic."""

    errors = sorted(
        schema_validator(kind).iter_errors(document),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if not errors:
        return
    error = errors[0]
    location = ".".join(str(part) for part in error.absolute_path) or "<root>"
    raise ValueError(
        f"{source}: invalid {kind} metadata at {location}: {error.message}"
    )


def load_document(path: Path) -> object:
    """Read JSON or YAML without assigning defaults or coercing field values."""

    source = path.read_text(encoding="utf-8")
    return json.loads(source) if path.suffix == ".json" else yaml.safe_load(source)


def validate_path(path: Path, kind: str) -> None:
    """Load and validate one metadata file against its declared owner schema."""

    validate_document(load_document(path), kind, path)


def catalogue_documents(root: Path) -> Iterable[tuple[str, Path]]:
    """Enumerate every versioned catalogue document governed by a schema."""

    catalog = root / "catalog"
    for kind in ("recipe", "generation", "validation"):
        for branch in ("prod", "other"):
            yield from ((kind, path) for path in sorted((catalog / branch).rglob(f"{kind}.yaml")))


def validate_repository(
    root: Path,
    *,
    campaigns: Iterable[Path] = (),
    experiments: Iterable[Path] = (),
) -> Counter[str]:
    """Validate the catalogue plus explicitly selected ignored local documents."""

    counts: Counter[str] = Counter()
    documents = list(catalogue_documents(root))
    documents.extend(("campaign", path) for path in campaigns)
    documents.extend(("experiment", path) for path in experiments)
    for kind, path in documents:
        validate_path(path, kind)
        counts[kind] += 1
    return counts


def main() -> int:
    """Validate tracked catalogue metadata and optional explicit local inputs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--campaign", type=Path, action="append", default=[])
    parser.add_argument("--experiment", type=Path, action="append", default=[])
    arguments = parser.parse_args()
    counts = validate_repository(
        arguments.root.resolve(),
        campaigns=arguments.campaign,
        experiments=arguments.experiment,
    )
    summary = ", ".join(f"{kind}={counts[kind]}" for kind in sorted(counts))
    print(f"Metadata schemas valid: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
