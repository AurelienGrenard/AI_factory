"""Build temporary, paired scaling inputs without changing catalogue datasets.

A 100-row tile samples 90 core and 10 stress rows at the midpoints of ten-row
strata. Repeating the same aligned tile keeps workload composition identical
at 100, 1,000 and 10,000 prices. These are benchmark fixtures, not new independent
financial samples or replacements for the ordered production catalogue.
The separate catalogue profile copies all 1,000 original rows unchanged for
dataset-runtime measurements; its provenance cannot be mixed with tile scaling.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

TILE_INDICES = tuple(range(5, 1000, 10))
INPUT_PROFILE = "repeated_stratified_90_core_10_stress_v1"
CATALOGUE_INPUT_PROFILE = "ordered_catalogue_900_core_100_stress_v1"
MAX_PRICE_COUNT = 10000


def source_indices(rows: int) -> list[int]:
    if not 1 <= rows <= MAX_PRICE_COUNT:
        raise ValueError("Scaling input count must be between 1 and 10,000")
    return [TILE_INDICES[i % len(TILE_INDICES)] for i in range(rows)]


def expand_document(document: dict, rows: int, label: str, directory: Path,
                    profile: str = INPUT_PROFILE) -> dict:
    row_field, = [field for field in ("models", "products", "curves") if field in document]
    if document["row_count"] != 1000 or len(document[row_field]) != 1000:
        raise ValueError("Scaling fixtures require the canonical 1,000-row source")
    if profile not in (INPUT_PROFILE, CATALOGUE_INPUT_PROFILE):
        raise ValueError("Unknown pricing input profile")
    if profile == CATALOGUE_INPUT_PROFILE and rows != 1000:
        raise ValueError("The catalogue profile requires all 1,000 ordered source rows")
    result = copy.deepcopy(document)
    result["database_id"] = f"scaling_{label}_r{rows}"
    result["catalog"] = str(directory)
    result["url"] = "https://datasets.ai-factory.example/performance/" + result["database_id"] + ".json"
    result["row_count"] = rows
    result["benchmark_only"] = True
    result["source_database_id"] = document["database_id"]
    result["input_profile"] = profile
    if profile == CATALOGUE_INPUT_PROFILE:
        # Keep every original row, identifier, parameter and calendar unchanged.
        return result
    result[row_field] = []
    for i, index in enumerate(source_indices(rows)):
        row = copy.deepcopy(document[row_field][index])
        row["source_row_id"] = row["id"]
        row["id"] = f"{i + 1:06d}"
        result[row_field].append(row)
    return result


def prepare_inputs(sources: dict[str, Path], directory: Path, counts: list[int],
                   profile: str = INPUT_PROFILE) -> dict:
    """Write fresh fixtures and their source provenance; no GPU or generator runs."""
    if profile not in (INPUT_PROFILE, CATALOGUE_INPUT_PROFILE):
        raise ValueError("Unknown pricing input profile")
    if profile == CATALOGUE_INPUT_PROFILE and set(counts) != {1000}:
        raise ValueError("The catalogue profile requires all 1,000 ordered source rows")
    directory.mkdir(parents=True, exist_ok=False)
    documents = {name: json.loads(path.read_text()) for name, path in sources.items()}
    source_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for name, path in sources.items()}
    identity = {"profile": profile, "source_sha256": source_hashes}
    if profile == INPUT_PROFILE:
        identity["tile_source_indices_zero_based"] = TILE_INDICES
    else:
        identity["source_indices_zero_based"] = list(range(1000))
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    by_count = {}
    hashes = {}
    for count in sorted(set(counts)):
        paths = {}
        for name, document in documents.items():
            path = directory / f"{name}_r{count}.json"
            path.write_text(json.dumps(expand_document(document, count, name, directory, profile), indent=2) + "\n")
            paths[name] = str(path.resolve())
            hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        by_count[str(count)] = paths
    result = {**identity, "fixture_signature": signature, "by_count": by_count,
              "paths_sha256": hashes, "sources": {name: str(path) for name, path in sources.items()},
              "seed_mapping": "canonical base seed + global benchmark row index; repeated inputs use distinct row keys"
                              if profile == INPUT_PROFILE else "canonical base seed + original catalogue row index"}
    (directory / "provenance.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
