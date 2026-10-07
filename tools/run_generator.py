#!/usr/bin/env python3
"""Build and run one catalog generator in an isolated, provenance-recorded directory."""

from __future__ import annotations

import argparse
import fcntl
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
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
from tools.datasets.dataset_provenance import fingerprint, input_fingerprints  # noqa: E402
from tools.datasets.metadata_schemas import validate_document  # noqa: E402

PARAMETER_KINDS = {"model_parameters", "curve_parameters", "product_parameters"}
SPECS = {
    spec.cmake_target: spec for spec in AVAILABLE_DATASET_SPECS
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
    if (recipe["output"]["path"] != spec.dataset_path or recipe["generation_output"] != spec.generation_yaml_path):
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


def compiler_version(path: str) -> str:
    if not path:
        return "unavailable"
    result = subprocess.run([path, "--version"], capture_output=True, text=True, check=True)
    return next((line for line in result.stdout.splitlines() if "release " in line), result.stdout.splitlines()[-1]) if Path(path).name == "nvcc" else result.stdout.splitlines()[0]


def build_evidence(build: Path, target: str) -> tuple[dict, dict]:
    cache = cache_values(build)
    graph_name = "build.ninja" if (build / "build.ninja").is_file() else "Makefile"
    if not (build / graph_name).is_file():
        raise FileNotFoundError(f"No CMake build graph in {build}")
    flag_file = build / "CMakeFiles" / f"{target}.dir" / "flags.make"
    target_flags = {}
    if flag_file.is_file():
        for line in flag_file.read_text().splitlines():
            if line.startswith(("CXX_FLAGS =", "CUDA_FLAGS =", "CXX_DEFINES =", "CUDA_DEFINES =")):
                key, value = line.split("=", 1)
                target_flags[key.strip()] = value.strip()
    hashes = {name: digest(build / name) for name in ("CMakeCache.txt", graph_name)}
    if flag_file.is_file():
        hashes[f"CMakeFiles/{target}.dir/flags.make"] = digest(flag_file)
    cuda_dependency_flags = {}
    link_file = build / "CMakeFiles" / f"{target}.dir" / "link.txt"
    if link_file.is_file():
        for dependency in sorted(set(re.findall(r"\blib([A-Za-z0-9_]+)\.a\b", link_file.read_text()))):
            dependency_flags = build / "CMakeFiles" / f"{dependency}.dir" / "flags.make"
            if not dependency_flags.is_file():
                continue
            lines = {}
            for line in dependency_flags.read_text().splitlines():
                if line.startswith(("CUDA_FLAGS =", "CUDA_DEFINES =")):
                    key, value = line.split("=", 1)
                    lines[key.strip()] = value.strip()
            if lines:
                cuda_dependency_flags[dependency] = lines
                hashes[f"CMakeFiles/{dependency}.dir/flags.make"] = digest(dependency_flags)
    options = {
        "cmake_generator": cache.get("CMAKE_GENERATOR"),
        "build_type": cache.get("CMAKE_BUILD_TYPE"),
        "cxx_standard": 20,
        "cuda_standard": 20,
        "cuda_architectures": cache.get("CUDA_WORKBENCH_ARCHITECTURES"),
        "tuning_profile": cache.get("AI_FACTORY_CUDA_TUNING_PROFILE_ID"),
        "mathdx_root": cache.get("AI_FACTORY_MATHDX_ROOT"),
        "cxx_compiler": cache.get("CMAKE_CXX_COMPILER"),
        "cxx_compiler_version": compiler_version(cache.get("CMAKE_CXX_COMPILER", "")),
        "cuda_compiler": cache.get("CMAKE_CUDA_COMPILER"),
        "cuda_compiler_version": compiler_version(cache.get("CMAKE_CUDA_COMPILER", "")),
        "cuda_host_compiler": cache.get("CMAKE_CUDA_HOST_COMPILER"),
        "cxx_release_flags": cache.get("CMAKE_CXX_FLAGS_RELEASE"),
        "cuda_release_flags": cache.get("CMAKE_CUDA_FLAGS_RELEASE"),
        "target_flags": target_flags,
        "cuda_dependency_flags": cuda_dependency_flags,
    }
    toolkit = Path(cache.get("CMAKE_CUDA_COMPILER", "")).parent.parent
    cuda_math_header = next((
        toolkit / relative for relative in (
            "targets/x86_64-linux/include/crt/math_functions.h",
            "include/crt/math_functions.h",
        ) if (toolkit / relative).is_file()
    ), None)
    if cuda_math_header is not None:
        options["cuda_math_header_sha256"] = digest(cuda_math_header)
    compatibility_marker = toolkit / ".ai-factory-cuda-glibc-compat.json"
    if compatibility_marker.is_file():
        options["cuda_glibc_compatibility"] = json.loads(compatibility_marker.read_text())
    return hashes, options


def hardware(kind: str, gpu: str | None) -> dict:
    if kind in PARAMETER_KINDS:
        model = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                      if line.startswith("model name")), platform.processor())
        return {"device": "cpu", "model": model, "host": platform.node()}
    selector = gpu if gpu is not None else os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    if not selector:
        raise ValueError("No GPU is visible; set --gpu or CUDA_VISIBLE_DEVICES")
    result = subprocess.run([
        "nvidia-smi", "-i", selector,
        "--query-gpu=index,name,compute_cap,uuid,driver_version",
        "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, check=True)
    parts = [part.strip() for part in result.stdout.strip().split(",")]
    if len(parts) != 5:
        raise ValueError(f"Cannot identify GPU {selector}: {result.stdout!r}")
    return dict(zip(("index", "name", "compute_capability", "uuid", "driver_version"), parts)) | {"device": "gpu"}


def enrich_receipt(spec, recipe: dict, stage: Path, build: Path, gpu: str | None) -> Path:
    generation_path = stage / spec.generation_yaml_path
    dataset_path = stage / spec.dataset_path
    if not generation_path.is_file() or not dataset_path.is_file():
        raise FileNotFoundError(f"Generator did not create {dataset_path} and {generation_path}")
    record = yaml.safe_load(generation_path.read_text())
    if not isinstance(record, dict) or record.get("status") != "complete":
        raise ValueError(f"Invalid native generation receipt: {generation_path}")
    inputs = list(recipe.get("inputs", {}).values())
    hashes, options = build_evidence(build, spec.cmake_target)
    record["recipe"] = {"path": spec.recipe_yaml_path, "sha256": digest(contained_path(ROOT, spec.recipe_yaml_path))}
    record["artifact"]["sha256"] = digest(dataset_path)
    record["inputs"] = input_fingerprints(stage, inputs)
    record["execution"].update({
        "binary_sha256": digest(build / spec.cmake_target),
        "generator_sha256": digest(contained_path(ROOT, spec.generator_path)),
        "declared_method": recipe.get("numerical_method", {"engine": spec.dataset_kind}),
        "build_hashes": hashes,
    })
    record["provenance"] = {
        "generator": spec.generator_path,
        "input_files": {name: digest(stage / name) for name in inputs},
        "hardware_used": hardware(spec.dataset_kind, gpu),
        "build_options": options,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    record["record_sha256"] = fingerprint(record)
    validate_document(record, "generation", generation_path)
    generation_path.write_text(yaml.safe_dump(record, sort_keys=False))
    return generation_path


def build_target(build: Path, target: str) -> None:
    print(f"Building {target}", flush=True)
    subprocess.run(["cmake", "--build", str(build), "--target", target, "--parallel", "1"], cwd=ROOT, check=True)


def run_binary(build: Path, target: str, stage: Path, gpu: str | None, smoke: bool = False) -> None:
    environment = os.environ.copy()
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = gpu
    command = [str(build / target)] + (["--smoke-test"] if smoke else [])
    print(f"Running {target} in {stage}", flush=True)
    subprocess.run(command, cwd=stage, env=environment, check=True)


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
    run_binary(build, dependency.cmake_target, stage, gpu)
    enrich_receipt(dependency, dependency_recipe, stage, build, gpu)
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
    run_binary(build, target, stage, arguments.gpu, arguments.smoke_test)
    if arguments.smoke_test:
        print("Smoke test passed; no dataset was generated.")
        return 0
    receipt = enrich_receipt(spec, recipe, stage, build, arguments.gpu)
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
