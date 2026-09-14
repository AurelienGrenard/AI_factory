"""Select, build and run pricing recipes from the typed capability manifest."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/codegen/pricing_bindings"))

from capability_manifest import AVAILABLE_DATASET_SPECS  # noqa: E402


def select_specs(
    asset_class: str, construction: str, kind: str,
    model_family: str | None, models: set[str], requested: set[str],
) -> list:
    eligible = [
        spec for spec in AVAILABLE_DATASET_SPECS
        if spec.asset_class == asset_class
        and spec.construction == construction
        and spec.dataset_kind == kind
        and (model_family is None or
             spec.source_prefix.startswith(f"model/{asset_class}/{model_family}/"))
    ]
    available = {spec.cmake_target: spec for spec in eligible}
    unknown = requested - available.keys()
    if unknown:
        raise ValueError("Targets outside the selection: " + ", ".join(sorted(unknown)))
    selected = [
        spec for spec in eligible
        if (not models or spec.model in models)
        and (not requested or spec.cmake_target in requested)
    ]
    if not selected:
        raise ValueError("No pricing generator matches the selection")
    return sorted(selected, key=lambda spec: spec.cmake_target)


def another_campaign_running() -> bool:
    lock_path = ROOT / "datasets/.generation-campaign.lock"
    if not lock_path.is_file():
        return False
    with lock_path.open() as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(lock, fcntl.LOCK_UN)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-class", choices=("equity", "fixed_income"), required=True)
    parser.add_argument("--construction", choices=("aligned", "cartesian"), required=True)
    parser.add_argument("--kind", choices=("prices", "price_delta"), default="prices")
    parser.add_argument("--model-family", help="model family below the asset class, e.g. markovian")
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--build", type=Path, default=ROOT / "build")
    parser.add_argument("--build-jobs", type=int, default=1)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--list", action="store_true", help="show selection without building")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if args.build_jobs <= 0:
        parser.error("--build-jobs must be positive")
    if args.publish and not args.execute:
        parser.error("--publish requires --execute")
    if args.execute and args.run_dir is None:
        parser.error("--execute requires --run-dir")
    if args.list and (args.execute or args.publish or args.skip_build):
        parser.error("--list cannot be combined with execution or build options")
    if args.execute and another_campaign_running():
        parser.error("Another dataset campaign is running; wait for it to finish or stop it first")

    try:
        specs = select_specs(
            args.asset_class, args.construction, args.kind,
            args.model_family, set(args.model), set(args.target),
        )
    except ValueError as error:
        parser.error(str(error))
    targets = [spec.cmake_target for spec in specs]
    if args.list:
        print(json.dumps({
            "count": len(specs),
            "asset_class": args.asset_class,
            "construction": args.construction,
            "kind": args.kind,
            "model_family": args.model_family,
            "targets": [
                {"target": spec.cmake_target, "model": spec.model,
                 "curve": spec.curve, "product": spec.product,
                 "dataset": spec.dataset_path}
                for spec in specs
            ],
        }, indent=2))
        return 0

    if not args.skip_build:
        subprocess.run([
            "cmake", "--build", str(args.build), "--target",
            *targets, "inspect_pricing_launch_plan", f"-j{args.build_jobs}",
        ], cwd=ROOT, env={**os.environ, "CCACHE_DISABLE": "1"}, check=True)

    command = [
        sys.executable, str(ROOT / "tools/datasets/generate_catalog.py"),
        "--build", str(args.build), "--kind", args.kind,
    ]
    for target in targets:
        command.extend(("--target", target))
    if args.execute:
        command.extend(("--run-dir", str(args.run_dir), "--execute"))
        if args.publish:
            command.append("--publish")
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
