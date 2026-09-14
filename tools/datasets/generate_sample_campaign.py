"""Select, build, and run model-terminal sample recipes without replacing existing data."""

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
    *, asset_class: str = "all", model_family: str | None = None,
    models: set[str] | None = None, targets: set[str] | None = None,
    include_published: bool = False, root: Path = ROOT,
) -> tuple[list, list]:
    """Return (selected, already published) from the authoritative manifest."""
    models = models or set()
    targets = targets or set()
    eligible = [
        spec for spec in AVAILABLE_DATASET_SPECS
        if spec.dataset_kind == "samples"
        and (asset_class == "all" or spec.asset_class == asset_class)
        and (model_family is None or spec.source_prefix.startswith(
            f"model/{spec.asset_class}/{model_family}/"
        ))
    ]
    available = {spec.cmake_target for spec in eligible}
    unknown = targets - available
    if unknown:
        raise ValueError("Targets outside the sample selection: " + ", ".join(sorted(unknown)))
    selected = []
    published = []
    for spec in sorted(eligible, key=lambda item: item.cmake_target):
        if (models and spec.model not in models) or (targets and spec.cmake_target not in targets):
            continue
        dataset = root / spec.dataset_path
        catalog = root / spec.catalog_yaml_path
        if dataset.exists() != catalog.exists():
            raise ValueError(
                f"Partial published pair for {spec.cmake_target}: {dataset}, {catalog}"
            )
        if dataset.exists():
            published.append(spec)
            if not include_published:
                continue
        selected.append(spec)
    if not selected and not published:
        raise ValueError("No sample generator matches the selection")
    return selected, published


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
    parser.add_argument("--asset-class", choices=("all", "equity", "fixed_income"), default="all")
    parser.add_argument("--model-family", help="equity family, such as markovian or rough")
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--include-published", action="store_true",
                        help="explicitly select datasets that already have JSON and YAML")
    parser.add_argument("--build", type=Path, default=ROOT / "build")
    parser.add_argument("--build-jobs", type=int, default=2)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--list", action="store_true", help="inspect selection without building")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--resume", action="store_true", help="resume frozen campaign without rebuilding")
    args = parser.parse_args()
    if args.build_jobs <= 0:
        parser.error("--build-jobs must be positive")
    if args.publish and not args.execute:
        parser.error("--publish requires --execute")
    if args.execute and args.run_dir is None:
        parser.error("--execute requires --run-dir")
    if args.list and (args.execute or args.publish or args.resume):
        parser.error("--list cannot be combined with execution")
    if args.resume:
        if (not args.execute or args.run_dir is None or args.include_published
                or args.publish or args.model or args.target or args.model_family
                or args.asset_class != "all"):
            parser.error("--resume uses only the frozen --run-dir and --execute")
        return subprocess.call([
            sys.executable, str(ROOT / "tools/datasets/generate_catalog.py"),
            "--run-dir", str(args.run_dir), "--execute", "--resume",
        ], cwd=ROOT)
    try:
        selected, published = select_specs(
            asset_class=args.asset_class, model_family=args.model_family,
            models=set(args.model), targets=set(args.target),
            include_published=args.include_published,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.list:
        print(json.dumps({
            "available": len({spec.cmake_target for spec in (*selected, *published)}),
            "selected": len(selected),
            "already_published": len(published),
            "rows_per_dataset": 3_000_000,
            "targets": [
                {"target": spec.cmake_target, "model": spec.model,
                 "asset_class": spec.asset_class, "dataset": spec.dataset_path}
                for spec in selected
            ],
        }, indent=2))
        return 0
    if not selected:
        print("All matching sample datasets already have JSON and YAML; nothing to build or run.")
        return 0
    if args.execute and another_campaign_running():
        parser.error("Another dataset campaign is running")
    targets = [spec.cmake_target for spec in selected]
    subprocess.run([
        "cmake", "--build", str(args.build), "--target", *targets,
        f"-j{args.build_jobs}",
    ], cwd=ROOT, env={**os.environ, "CCACHE_DISABLE": "1"}, check=True)
    command = [sys.executable, str(ROOT / "tools/datasets/generate_catalog.py"),
               "--build", str(args.build), "--kind", "samples"]
    for target in targets:
        command.extend(("--target", target))
    if args.execute:
        command.extend(("--run-dir", str(args.run_dir), "--execute"))
        if args.publish:
            command.append("--publish")
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
