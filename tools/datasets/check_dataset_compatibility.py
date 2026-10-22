"""Check the currently published JSON against its recipe and technical receipt.

The command does not require a native build or compare historical generator bytes.
It checks metadata and artifact integrity, not numerical reproducibility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.datasets.dataset_provenance import assess


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="root used to verify available parameter inputs")
    # Accepted for older scripts, but neither an executable nor a build is a
    # condition for editing published metadata.
    parser.add_argument("--target", help=argparse.SUPPRESS)
    parser.add_argument("--build", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        recipe = yaml.safe_load(args.recipe.read_text())
        generation = yaml.safe_load(args.generation.read_text())
        root = args.root.resolve()
        relative_recipe = args.recipe.resolve().relative_to(root).as_posix()
        relative_generation = args.generation.resolve().relative_to(root).as_posix()
        if generation.get("schema_version") in (1, 2) and generation.get("recipe", {}).get("path") != relative_recipe:
            raise ValueError("generation.yaml names a different recipe")
        if relative_generation != (args.recipe.resolve().with_name("generation.yaml")
                                   .relative_to(root).as_posix()):
            raise ValueError("generation.yaml must share the recipe directory")
        relative_generator = (
            args.recipe.resolve().with_name("generator.cpp")
            .relative_to(root).as_posix()
        )
        if recipe.get("generator") != relative_generator:
            raise ValueError("recipe.yaml names a different generator.cpp")
        result = assess(recipe, generation, args.dataset, root)
    except (OSError, ValueError, TypeError, yaml.YAMLError) as error:
        result = {"status": "review_required", "reasons": [str(error)],
                  "certification": "not_assessed"}
    print(json.dumps(result, indent=2, allow_nan=False))
    return {"compatible": 0, "review_required": 2, "invalid": 4}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
