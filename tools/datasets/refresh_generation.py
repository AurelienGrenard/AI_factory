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
from tools.datasets.dataset_provenance import assess, hardware_from_observation, known_fields
from tools.datasets.metadata_schemas import validate_document


def _historical_inputs(recipe: dict, previous: dict) -> dict[str, dict[str, str]]:
    """Keep hashes observed at execution time, even if input files changed."""
    declared = recipe.get("inputs", {})
    if not isinstance(declared, dict):
        raise ValueError("Recipe inputs must be a mapping")
    version = previous.get("schema_version")
    if version in (2, 4):
        inputs = previous.get("inputs", {})
        if {role: item.get("path") for role, item in inputs.items()} != declared:
            raise ValueError("Historical input paths contradict recipe")
        hashes = {role: item["sha256"] for role, item in inputs.items()}
    elif version == 3:
        hashes = previous.get("inputs", {})
    elif version == 1:
        recorded = previous.get("provenance", {}).get("input_files", {})
        hashes = {role: recorded.get(relative) for role, relative in declared.items()}
    else:
        raise ValueError("Unsupported generation metadata version")
    if set(hashes) != set(declared) or any(
        not isinstance(value, str) or len(value) != 64 for value in hashes.values()
    ):
        raise ValueError("Historical dataset SHA256 is missing")
    return {role: {"path": relative, "sha256": hashes[role]}
            for role, relative in declared.items()}

def _historical_environment(recipe: dict, previous: dict) -> tuple[dict, dict]:
    version = previous["schema_version"]
    if version in (3, 4):
        hardware, software = previous.get("hardware", {}), previous.get("software", {})
    elif version == 2:
        environment = previous["environment"]
        hardware, software = environment["hardware"], environment["software"]
    else:
        old = previous.get("provenance", {})
        options = old.get("build_options", {})
        observation = old.get("runtime_observation")
        cpu = recipe.get("kind") in {
            "model_parameters", "product_parameters", "curve_parameters",
            "correlation_parameters",
        }
        hardware = old.get("hardware_used") or hardware_from_observation(
            observation, cpu=cpu
        )
        software = {
            "cmake": options.get("cmake_version"),
            "cxx_compiler": options.get("cxx_compiler_version"),
            "cuda_compiler": options.get("cuda_compiler_version"),
        }
    hardware = known_fields({key: value for key, value in hardware.items()
                             if key not in {"host", "index", "cuda_visible_devices"}})
    if "model" in hardware:
        hardware["cpu_model"] = hardware.pop("model")
    software = known_fields(software)
    return hardware, software

def refresh(recipe: dict, previous: dict, dataset: Path, root: Path,
            removed_gradients: list[str]) -> dict:
    if not isinstance(recipe, dict) or not isinstance(previous, dict):
        raise ValueError("Recipe and generation metadata must be mappings")
    former_output = previous.get(
        "output" if previous.get("schema_version") in (3, 4) else "artifact", {}
    )
    if not former_output.get("sha256"):
        raise ValueError("Missing original dataset SHA256")
    if len(set(removed_gradients)) != len(removed_gradients):
        raise ValueError("Duplicate removed gradient")
    if removed_gradients:
        if recipe.get("kind") != "price_gradients":
            raise ValueError("Removed gradients require a price-gradient recipe")
        current = {item["parameter"] for item in recipe["sensitivity"]["parameters"]}
        if current.intersection(removed_gradients):
            raise ValueError("Removed gradient is still selected by the recipe")
    relative_dataset = recipe.get("output", {}).get("path")
    if not relative_dataset or contained_path(root, relative_dataset).resolve() != dataset.resolve():
        raise ValueError("Dataset path contradicts recipe")
    hardware, software = _historical_environment(recipe, previous)
    record = {
        "schema_version": 4,
        "status": "complete",
        "inputs": _historical_inputs(recipe, previous),
        "output": {"path": relative_dataset, "sha256": digest(dataset)},
    }
    if hardware:
        record["hardware"] = hardware
    if software:
        record["software"] = software
    if (recipe.get("kind") in {"prices", "price_gradients", "price_delta"}
            and previous.get("timing") and previous.get("schema_version") == 4):
        record["timing"] = previous["timing"]
    validate_document(record, "generation", dataset)
    result = assess(recipe, record, dataset)
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
    print(f"Updated {args.generation}: {record['output']['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
