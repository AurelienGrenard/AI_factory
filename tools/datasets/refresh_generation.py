#!/usr/bin/env python3
"""Refresh editable publication metadata after a coherent recipe/JSON curation.

The command never reruns pricing and never invents historical tool versions.
It checks the resulting contract before replacing generation.yaml.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.datasets.artifact_publication import contained_path, digest
from tools.datasets.dataset_provenance import assess, hardware_from_observation
from tools.datasets.metadata_schemas import validate_document


def refresh(recipe: dict, previous: dict, dataset: Path, root: Path,
            removed_gradients: list[str]) -> dict:
    if not isinstance(recipe, dict) or not isinstance(previous, dict):
        raise ValueError("Recipe and generation metadata must be mappings")
    artifact = previous.get("artifact", {})
    former_sha = artifact.get("sha256")
    current_sha = digest(dataset)
    if not former_sha or not current_sha:
        raise ValueError("Missing original or current dataset SHA256")
    if len(set(removed_gradients)) != len(removed_gradients):
        raise ValueError("Duplicate removed gradient")
    if removed_gradients:
        if recipe.get("kind") != "price_gradients":
            raise ValueError("Removed gradients require a price-gradient recipe")
        current = {item["parameter"] for item in recipe["sensitivity"]["parameters"]}
        if current.intersection(removed_gradients):
            raise ValueError("Removed gradient is still selected by the recipe")
    input_paths = recipe.get("inputs", {})
    if not isinstance(input_paths, dict):
        raise ValueError("Recipe inputs must be a mapping")
    inputs = {}
    for role, relative in input_paths.items():
        path = contained_path(root, relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        inputs[role] = {"path": relative, "sha256": digest(path)}
    if previous.get("schema_version") == 2:
        environment = previous["environment"]
    elif previous.get("schema_version") == 1:
        old = previous.get("provenance", {})
        options = old.get("build_options", {})
        observation = old.get("runtime_observation")
        environment = {
            "hardware": old.get("hardware_used") or hardware_from_observation(
                observation, cpu=recipe.get("kind") in {
                    "model_parameters", "product_parameters", "curve_parameters"}),
            "software": {
                "cmake": options.get("cmake_version", "unavailable"),
                "cxx_compiler": options.get("cxx_compiler_version", "unavailable"),
                "cuda_compiler": options.get("cuda_compiler_version", "unavailable"),
            },
        }
    else:
        raise ValueError("Unsupported generation metadata version")
    relative_dataset = recipe.get("output", {}).get("path")
    if not relative_dataset:
        raise ValueError("Recipe has no output path")
    record = {
        "schema_version": 2,
        "status": "complete",
        "recipe": {"path": previous.get("recipe", {}).get("path")},
        "artifact": {"path": relative_dataset, "row_count": recipe["row_count"],
                     "sha256": current_sha},
        "inputs": inputs,
        "environment": environment,
    }
    if previous.get("timing") is not None:
        record["timing"] = previous["timing"]
    if previous.get("curation") is not None:
        record["curation"] = previous["curation"]
    if removed_gradients:
        record["curation"] = {
            "source_sha256": former_sha,
            "removed_gradients": removed_gradients,
        }
    validate_document(record, "generation", dataset)
    result = assess(recipe, record, dataset, root)
    if result["status"] != "compatible":
        raise ValueError("Current recipe/JSON mismatch: " + "; ".join(result["reasons"]))
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--removed-gradient", action="append", default=[],
                        help="optional curation note for a removed coordinate; repeat as needed")
    args = parser.parse_args()
    root = args.root.resolve()
    recipe = yaml.safe_load(args.recipe.read_text())
    previous = yaml.safe_load(args.generation.read_text())
    record = refresh(recipe, previous, args.dataset, root, args.removed_gradient)
    with tempfile.NamedTemporaryFile("w", dir=args.generation.parent,
                                     prefix=".generation-", suffix=".yaml",
                                     delete=False) as stream:
        yaml.safe_dump(record, stream, sort_keys=False)
        temporary = Path(stream.name)
    temporary.replace(args.generation)
    print(f"Updated {args.generation}: {record['artifact']['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
