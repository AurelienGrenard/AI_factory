#!/usr/bin/env python3
"""Build and run one catalog generator in an isolated, provenance-recorded directory."""

from __future__ import annotations

import argparse
import fcntl
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from runpy import run_path
import shutil
import subprocess
import sys
import time
from uuid import uuid4

try:
    import yaml
    import jsonschema  # needed by the shared metadata validator
except ModuleNotFoundError as error:
    raise SystemExit(
        f"Missing Python package {error.name}; install PyYAML and jsonschema "
        "in the Python environment used for this command."
    ) from error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/codegen/pricing_bindings"))
from capability_manifest import AVAILABLE_DATASET_SPECS  # noqa: E402
from tools.datasets.artifact_publication import contained_path, digest, publish_pair  # noqa: E402
from tools.datasets.dataset_provenance import cpu_model, generation_record, software_versions  # noqa: E402
from tools.datasets.metadata_schemas import validate_document  # noqa: E402
from tools.datasets.multi_asset_specs import REFERENCE_DATASET_SPECS  # noqa: E402

PARAMETER_KINDS = {"model_parameters", "curve_parameters", "product_parameters",
                   "correlation_parameters"}

LOCAL_SPECS_PATH = ROOT / "work/tools/multi_asset_specs.py"
LOCAL_DATASET_SPECS = (
    tuple(run_path(str(LOCAL_SPECS_PATH))["LOCAL_DATASET_SPECS"])
    if LOCAL_SPECS_PATH.is_file() else ()
)

SPECS = {
    spec.cmake_target: spec for spec in
    (*AVAILABLE_DATASET_SPECS, *REFERENCE_DATASET_SPECS,
     *LOCAL_DATASET_SPECS)
    if contained_path(ROOT, spec.generator_path).is_file()
}
OUTPUT_SPECS = {spec.dataset_path: spec for spec in SPECS.values()}


def build_directory(preset: str) -> Path:
    document = json.loads((ROOT / "CMakePresets.json").read_text())
    for item in document["configurePresets"]:
        if item["name"] == preset and "binaryDir" in item:
            return Path(item["binaryDir"].replace("${sourceDir}", str(ROOT)))
    raise ValueError(f"Unknown configured preset: {preset}")


def recipe_for(spec) -> dict:
    path = contained_path(ROOT, spec.recipe_yaml_path)
    recipe = yaml.safe_load(path.read_text())
    validate_document(recipe, "recipe", path)
    if (recipe["output"]["path"] != spec.dataset_path
            or recipe["generator"] != spec.generator_path):
        raise ValueError(f"Recipe does not match the manifest: {path}")
    return recipe


def copy_input(relative: str, stage: Path) -> None:
    source = contained_path(ROOT, relative)
    destination = stage / relative
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def cache_values(build: Path) -> dict[str, str]:
    values = {}
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line or ":" not in line.split("=", 1)[0]:
            continue
        key, value = line.split("=", 1)
        values[key.split(":", 1)[0]] = value
    return values


def hardware(kind: str, gpu: str | None) -> dict:
    if kind in PARAMETER_KINDS:
        return {"device": "cpu", "cpu_model": cpu_model()}
    selector = gpu if gpu is not None else os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    if not selector:
        raise ValueError("No GPU is visible; set --gpu or CUDA_VISIBLE_DEVICES")
    result = subprocess.run([
        "nvidia-smi", "-i", selector,
        "--query-gpu=name,compute_cap,uuid,driver_version",
        "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, check=True)
    parts = [part.strip() for part in result.stdout.strip().split(",")]
    if len(parts) != 4:
        raise ValueError(f"Cannot identify GPU {selector}: {result.stdout!r}")
    return dict(zip(("name", "compute_capability", "uuid", "driver_version"), parts)) | {"device": "gpu", "cpu_model": cpu_model()}


def enrich_receipt(spec, recipe: dict, stage: Path, build: Path, gpu: str | None,
                   wall_seconds: float) -> Path:
    generation_path = stage / spec.generation_yaml_path
    dataset_path = stage / spec.dataset_path
    if not generation_path.is_file() or not dataset_path.is_file():
        raise FileNotFoundError(f"Generator did not create {dataset_path} and {generation_path}")
    native = yaml.safe_load(generation_path.read_text())
    record = generation_record(
        native, spec.recipe_yaml_path, spec.dataset_path, recipe.get("inputs", {}),
        stage, hardware(spec.dataset_kind, gpu), software_versions(build),
        wall_seconds=wall_seconds if spec.dataset_kind in {"prices", "price_gradients", "price_delta"} else None,
    )
    generation_path.write_text(yaml.safe_dump(record, sort_keys=False))
    return generation_path


def build_target(build: Path, target: str) -> None:
    print(f"Building {target}", flush=True)
    subprocess.run(["cmake", "--build", str(build), "--target", target, "--parallel", "1"], cwd=ROOT, check=True)


def run_binary(build: Path, target: str, stage: Path, gpu: str | None, smoke: bool = False) -> float:
    environment = os.environ.copy()
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = gpu
    command = [str(build / target)] + (["--smoke-test"] if smoke else [])
    print(f"Running {target} in {stage}", flush=True)
    started = time.perf_counter()
    subprocess.run(command, cwd=stage, env=environment, check=True)
    return time.perf_counter() - started


def ensure_input(relative: str, stage: Path, build: Path, gpu: str | None) -> None:
    destination = stage / relative
    if destination.is_file():
        return
    source = contained_path(ROOT, relative)
    if source.is_file():
        copy_input(relative, stage)
        return
    dependency = OUTPUT_SPECS.get(relative)
    if dependency is None or dependency.dataset_kind not in PARAMETER_KINDS:
        raise FileNotFoundError(f"Missing input {relative}; no parameter generator is available")
    dependency_recipe = recipe_for(dependency)
    copy_input(dependency.recipe_yaml_path, stage)
    build_target(build, dependency.cmake_target)
    wall_seconds = run_binary(build, dependency.cmake_target, stage, gpu)
    enrich_receipt(dependency, dependency_recipe, stage, build, gpu, wall_seconds)
    if not destination.is_file():
        raise FileNotFoundError(f"Dependency generator did not create {relative}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", choices=("local-sm89", "ppti", "ppti-gpu1", "ppti-gpu3"))
    parser.add_argument("target", nargs="?", help="CMake generator target or its generator.cpp/recipe.yaml path")
    parser.add_argument("--build-dir", type=Path, help="override the preset build directory")
    parser.add_argument("--gpu", help="GPU index to expose to the generator (for example 0, 1, 2, 3)")
    parser.add_argument("--list", action="store_true", help="list the available generator targets")
    parser.add_argument("--smoke-test", action="store_true", help="run the sample generator's quick check")
    parser.add_argument("--publish", action="store_true", help="publish a new immutable dataset and receipt")
    parser.add_argument("--run-dir", type=Path, help="unique output directory under work/; place run.log here too")
    arguments = parser.parse_args()
    if arguments.list:
        for name in sorted(SPECS):
            print(name)
        return 0
    if not arguments.target:
        parser.error("specify a generator target or use --list")
    target = arguments.target
    if target not in SPECS:
        matches = [spec.cmake_target for spec in SPECS.values()
                   if target in (spec.generator_path, spec.recipe_yaml_path)]
        if len(matches) != 1:
            parser.error(f"unknown generator {target!r}; use --list")
        target = matches[0]
    spec = SPECS[target]
    if arguments.smoke_test and spec.dataset_kind != "samples":
        parser.error("--smoke-test is supported by sample generators only")
    if arguments.smoke_test and arguments.publish:
        parser.error("--smoke-test does not publish a dataset")
    if arguments.publish and arguments.run_dir:
        parser.error("--publish and --run-dir cannot be combined")
    build = arguments.build_dir.resolve() if arguments.build_dir else build_directory(arguments.preset)
    if not (build / "CMakeCache.txt").is_file():
        parser.error(f"configure {arguments.preset} first; missing {build / 'CMakeCache.txt'}")
    recipe = recipe_for(spec)
    if spec.dataset_kind not in PARAMETER_KINDS:
        observed_architecture = hardware(spec.dataset_kind, arguments.gpu)["compute_capability"].replace(".", "")
        compiled_architectures = cache_values(build).get("CUDA_WORKBENCH_ARCHITECTURES", "").split(";")
        if observed_architecture not in compiled_architectures:
            parser.error(
                f"selected GPU is SM{observed_architecture}, but {build} targets "
                f"{';'.join(compiled_architectures)}"
            )
    run_dir = arguments.run_dir.resolve() if arguments.run_dir else (
        ROOT / "work/generation/runs" / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{target}-{uuid4().hex[:8]}"
    )
    work_root = (ROOT / "work").resolve()
    if not run_dir.is_relative_to(work_root) or run_dir == work_root:
        parser.error(f"--run-dir must be inside {work_root}")
    if run_dir.exists():
        if not arguments.run_dir or not run_dir.is_dir() or any(
            entry.name not in {"run.log", "progress.json"} for entry in run_dir.iterdir()
        ):
            parser.error(f"run directory is not empty or cannot be reused: {run_dir}")
    else:
        run_dir.mkdir(parents=True)
    stage = run_dir
    print(f"Staging directory: {stage}", flush=True)
    copy_input(spec.recipe_yaml_path, stage)
    for input_path in recipe.get("inputs", {}).values():
        ensure_input(input_path, stage, build, arguments.gpu)
    build_target(build, target)
    wall_seconds = run_binary(build, target, stage, arguments.gpu, arguments.smoke_test)
    if arguments.smoke_test:
        print("Smoke test passed; no dataset was generated.")
        return 0
    receipt = enrich_receipt(spec, recipe, stage, build, arguments.gpu, wall_seconds)
    dataset = stage / spec.dataset_path
    if arguments.publish:
        missing_published_inputs = [name for name in recipe.get("inputs", {}).values()
                                    if not contained_path(ROOT, name).is_file()]
        if missing_published_inputs:
            raise ValueError(
                "Cannot publish while parameter inputs exist only in the staged run: "
                + ", ".join(missing_published_inputs)
            )
        artifacts = [
            {"path": spec.dataset_path, "sha256": digest(dataset), "previous_sha256": None},
            {"path": spec.generation_yaml_path, "sha256": digest(receipt), "previous_sha256": None},
        ]
        lock_path = ROOT / "work/generation/.campaign.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            publish_pair(ROOT, stage, stage / "publication.json", artifacts)
        print(f"Published {contained_path(ROOT, spec.dataset_path)}")
    else:
        print(f"Dataset: {dataset}")
        print(f"Receipt: {receipt}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Generator stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
