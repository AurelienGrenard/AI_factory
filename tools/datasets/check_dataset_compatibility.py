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
from tools.datasets.dataset_provenance import assess, specification
from tools.datasets.generate_catalog import describe_job, inventory, require_current_build


def candidate_descriptor(root: Path, build: Path, target: str) -> dict:
    jobs = inventory(root, {"prices", "price_delta", "samples"}, set(), {target})
    require_current_build(root, build, jobs)
    job = jobs[0]
    job.update(describe_job(root, build, job))
    result = {
        "specification": specification(job),
        "inputs": job["semantic_inputs"],
        "execution": {"binary_sha256": digest(build / job["target"]),
                      "recipe_sha256": digest(contained_path(root, job["recipe"])),
                      "declared_method": job["declared_method"],
                      "launch_plan": job.get("launch_plan"),
                      "build_hashes": {name: digest(build / name) for name in ("CMakeCache.txt", "build.ninja")}},
    }
    if job.get("recipe_metadata"):
        result["execution"]["recipe_metadata_sha256"] = digest(contained_path(root, job["recipe_metadata"]))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True, help="YAML published with the existing dataset")
    parser.add_argument("--dataset", type=Path, required=True, help="Existing JSON, even if relocated")
    parser.add_argument("--target", required=True, help="Current native generator target")
    parser.add_argument("--build", type=Path, default=ROOT / "build-dev")
    args = parser.parse_args()
    try:
        catalog = yaml.safe_load(args.catalog.read_text())
        if not isinstance(catalog, dict):
            raise ValueError("Catalogue must be a mapping")
        # Integrity and legacy checks do not depend on having a current build.
        result = assess(catalog, args.dataset, None)
        if result.pop("candidate_required", False):
            candidate = candidate_descriptor(ROOT, args.build.resolve(), args.target)
            result = assess(catalog, args.dataset, candidate)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError, yaml.YAMLError) as error:
        result = {"status": "review_required", "reasons": [str(error)], "certification": "not_assessed"}
    print(json.dumps(result, indent=2, allow_nan=False))
    return {"compatible": 0, "review_required": 2, "regenerate": 3, "invalid": 4}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
