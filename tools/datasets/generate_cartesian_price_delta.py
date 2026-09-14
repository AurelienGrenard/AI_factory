"""Build and run Cartesian equity price-delta catalogue recipes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
CODEGEN = ROOT / "tools" / "codegen" / "pricing_bindings"
sys.path.insert(0, str(CODEGEN))

from capability_manifest import PRICE_DELTA_DATASET_SPECS  # noqa: E402


def cartesian_targets(models: set[str], requested: set[str]) -> list[str]:
    available = {
        spec.cmake_target: spec
        for spec in PRICE_DELTA_DATASET_SPECS
        if spec.construction == "cartesian"
    }
    unknown = requested - available.keys()
    if unknown:
        raise ValueError("Unknown Cartesian price-delta targets: " + ", ".join(sorted(unknown)))
    selected = [
        target for target, spec in available.items()
        if (not models or spec.model in models) and (not requested or target in requested)
    ]
    if not selected:
        raise ValueError("No Cartesian price-delta generator matches the selection")
    return sorted(selected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--build-jobs", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    arguments = parser.parse_args()

    controller = ROOT / "tools" / "datasets" / "generate_catalog.py"
    if arguments.resume:
        if not arguments.execute or arguments.run_dir is None:
            parser.error("--resume requires --execute and --run-dir")
        return subprocess.call([
            sys.executable, str(controller), "--run-dir", str(arguments.run_dir),
            "--execute", "--resume",
        ], cwd=ROOT)

    targets = cartesian_targets(set(arguments.model), set(arguments.target))
    if arguments.build_jobs <= 0:
        parser.error("--build-jobs must be positive")
    if not arguments.skip_build:
        environment = {**os.environ, "CCACHE_DISABLE": "1"}
        subprocess.run([
            "cmake", "--build", str(arguments.build), "--target",
            *targets, "inspect_pricing_launch_plan", f"-j{arguments.build_jobs}",
        ], cwd=ROOT, env=environment, check=True)

    command = [
        sys.executable, str(controller), "--build", str(arguments.build),
        "--kind", "price_delta",
    ]
    for target in targets:
        command.extend(("--target", target))
    if arguments.execute:
        if arguments.run_dir is None:
            parser.error("--execute requires --run-dir")
        command.extend(("--run-dir", str(arguments.run_dir), "--execute"))
        if arguments.publish:
            command.append("--publish")
    elif arguments.publish:
        parser.error("--publish requires --execute")
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
