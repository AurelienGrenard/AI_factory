"""Inspect reuse of an existing price/sample dataset without running or publishing it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.datasets.artifact_publication import contained_path, digest
from tools.datasets.dataset_provenance import assess
from tools.datasets.generate_catalog import describe_job, inventory, require_current_build


def candidate_descriptor(root: Path, build: Path, target: str) -> dict:
    jobs = inventory(root, {"prices", "price_delta", "price_gradients", "samples"}, set(), {target})
    require_current_build(root, build, jobs)
    job = jobs[0]
    job.update(describe_job(root, build, job))
    result = {
        "recipe_sha256": digest(contained_path(root, job["recipe"])),
        "inputs": job["semantic_inputs"],
        "execution": {"binary_sha256": digest(build / job["target"]),
                      "generator_sha256": digest(contained_path(root, job["generator"])),
                      "declared_method": job["declared_method"],
                      "launch_plan": job.get("launch_plan"),
                      "build_hashes": {name: digest(build / name) for name in ("CMakeCache.txt", "build.ninja")}},
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True, help="Canonical recipe YAML")
    parser.add_argument("--generation", type=Path, required=True, help="Published generation receipt YAML")
    parser.add_argument("--dataset", type=Path, required=True, help="Existing JSON, even if relocated")
    parser.add_argument("--target", required=True, help="Current native generator target")
    parser.add_argument("--build", type=Path, default=ROOT / "build")
    args = parser.parse_args()
    try:
        recipe = yaml.safe_load(args.recipe.read_text())
        generation = yaml.safe_load(args.generation.read_text())
        if not isinstance(recipe, dict) or not isinstance(generation, dict):
            raise ValueError("Recipe and generation receipt must be mappings")
        # Integrity and legacy checks do not depend on having a current build.
        result = assess(
            recipe, generation, args.dataset, None, digest(args.recipe)
        )
        if result.pop("candidate_required", False):
            candidate = candidate_descriptor(ROOT, args.build.resolve(), args.target)
            result = assess(
                recipe, generation, args.dataset, candidate, digest(args.recipe)
            )
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError, yaml.YAMLError) as error:
        result = {"status": "review_required", "reasons": [str(error)], "certification": "not_assessed"}
    print(json.dumps(result, indent=2, allow_nan=False))
    return {"compatible": 0, "review_required": 2, "regenerate": 3, "invalid": 4}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
