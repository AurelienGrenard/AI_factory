"""Record generation evidence and compare dataset reuse without claiming certification."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
import subprocess
import tarfile

import yaml

from tools.datasets.artifact_publication import contained_path, digest
from tools.datasets.metadata_schemas import validate_document


def fingerprint(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def parameter_fingerprint(path: Path) -> tuple[str, str]:
    """Keep every non-presentation field and row order; ignore file locations."""
    document = json.loads(path.read_text())
    roles = [key for key in ("models", "products", "curves") if key in document]
    if len(roles) != 1 or not isinstance(document[roles[0]], list):
        raise ValueError(f"Unknown parameter dataset schema: {path}")
    if (type(document.get("row_count")) is not int or document["row_count"] <= 0
            or document["row_count"] != len(document[roles[0]])):
        raise ValueError(f"Invalid parameter row count: {path}")
    semantic = {key: value for key, value in document.items()
                if key not in {"catalog", "url", "timing", "title"}}
    return roles[0], fingerprint(semantic)


def input_fingerprints(root: Path, paths: list[str]) -> dict:
    result = {}
    for relative in paths:
        role, checksum = parameter_fingerprint(contained_path(root, relative))
        if role in result:
            raise ValueError(f"Ambiguous parameter role: {role}")
        result[role] = checksum
    return result


def snapshot_sources(root: Path, destination: Path) -> str:
    """Archive the actual tracked/untracked maintained implementation, not only HEAD.

    This broad archive is provenance, never a semantic-compatibility fingerprint.
    SDKs, drivers and external libraries are not bundled.
    """
    names = subprocess.check_output([
        "git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--",
        "src", "tools", "cmake", "CMakeLists.txt", "CMakePresets.json",
    ], cwd=root).decode().split("\0")
    with tarfile.open(destination, "x:gz") as archive:
        for name in sorted(set(names) - {""}):
            path = contained_path(root, name)
            if path.is_file():
                before = digest(path)
                archive.add(path, arcname=name, recursive=False)
                if digest(path) != before:
                    raise ValueError(f"Source changed during snapshot: {name}")
    return digest(destination)


SCHEMA_VERSION = 4


def _version(command: list[str]) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=10, check=True)
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return next((line for line in lines if "release " in line), lines[0])
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unavailable"


def software_versions(build: Path) -> dict[str, str]:
    """Record the tools actually named by this build; no build hash is a reuse gate."""
    cache = {}
    cache_file = build / "CMakeCache.txt"
    if cache_file.is_file():
        for line in cache_file.read_text().splitlines():
            if "=" in line and ":" in line.split("=", 1)[0]:
                key, value = line.split("=", 1)
                cache[key.split(":", 1)[0]] = value
    cxx = cache.get("CMAKE_CXX_COMPILER", "")
    nvcc = cache.get("CMAKE_CUDA_COMPILER", "")
    versions = {
        "cmake": _version(["cmake", "--version"]),
        "cxx_compiler": _version([cxx, "--version"]) if cxx else "unavailable",
        "cuda_compiler": _version([nvcc, "--version"]) if nvcc else "unavailable",
    }
    json_config = Path(cache.get("nlohmann_json_DIR", "/nonexistent")) / "nlohmann_jsonConfigVersion.cmake"
    if json_config.is_file():
        match = re.search(r'PACKAGE_VERSION\s+"([^"]+)"', json_config.read_text())
        if match:
            versions["nlohmann_json"] = match.group(1)
    mathdx = cache.get("AI_FACTORY_MATHDX_ROOT", "")
    match = re.search(r'nvidia-mathdx-([0-9][^-]*)', mathdx)
    if match:
        versions["mathdx"] = match.group(1)
    return versions


def cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def known_fields(values: dict | None) -> dict:
    """Keep only technical facts that were actually recorded."""
    return {key: value for key, value in (values or {}).items()
            if value not in (None, "", "unavailable")}


def hardware_from_observation(observation: dict | None, *, cpu: bool = False) -> dict:
    observation = observation or {}
    if cpu:
        result = {"device": "cpu"}
    else:
        result = {}
    for source, target in (("hostname", "host"), ("cpu_model", "cpu_model")):
        if observation.get(source):
            result[target] = observation[source]
    if cpu:
        return result
    visible = observation.get("cuda_visible_devices")
    if visible is not None:
        result["cuda_visible_devices"] = visible
    rows = list(csv.DictReader((observation.get("stdout") or "").splitlines(), skipinitialspace=True))
    if rows:
        selected = (visible or "0").split(",")[0].strip()
        row = next((row for row in rows
                    if selected in {row.get("index", "").strip(),
                                    row.get("uuid", "").strip()}), rows[0])
        if row.get("name"):
            result["device"] = "gpu"
        for source, target in (("index", "index"), ("name", "name"),
                               ("uuid", "uuid"), ("compute_cap", "compute_capability"),
                               ("driver_version", "driver_version")):
            if row.get(source):
                result[target] = row[source].strip()
    return result

def generation_record(native: dict, recipe_path: str, dataset_path: str,
                      input_paths: dict[str, str], input_root: Path,
                      hardware: dict, software: dict) -> dict:
    """Keep a small editable receipt for the currently published JSON."""
    if not isinstance(native, dict) or native.get("status") != "complete":
        raise ValueError("Native generation receipt is incomplete")
    row_count = native.get("artifact", {}).get("row_count")
    if type(row_count) is not int or row_count < 1:
        raise ValueError("Native generation receipt lacks a valid row count")
    record = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "inputs": {role: {"path": relative,
                          "sha256": digest(contained_path(input_root, relative))}
                   for role, relative in input_paths.items()},
        "output": {"path": dataset_path,
                   "sha256": digest(contained_path(input_root, dataset_path))},
    }
    hardware = known_fields(hardware)
    software = known_fields(software)
    if hardware:
        record["hardware"] = hardware
    if software:
        record["software"] = software
    validate_document(record, "generation", recipe_path)
    return record


def attach_generation(work: Path, job: dict, state: dict) -> None:
    """Replace a checked native receipt with the compact publication record."""
    path = contained_path(work, job["generation"])
    native = yaml.safe_load(path.read_text())
    if not isinstance(native, dict) or native.get("schema_version") != 1:
        raise ValueError("Expected the native version-one generation receipt")
    if any(key in native for key in ("recipe", "inputs", "record_sha256")):
        raise ValueError("Native generation receipt unexpectedly contains provenance")
    recipe = yaml.safe_load(contained_path(work, job["recipe"]).read_text())
    if not isinstance(recipe, dict):
        raise ValueError("Invalid frozen recipe")
    record = generation_record(
        native, job["recipe"], job["dataset"], recipe.get("inputs", {}), work,
        hardware_from_observation(
            job.get("gpu_before"),
            cpu=job["kind"] in {"model_parameters", "product_parameters", "curve_parameters",
                                 "correlation_parameters"}),
        software_versions(Path(state["build"])),
    )
    path.write_text(yaml.safe_dump(record, sort_keys=False))


def _check_published_json(recipe: dict, generation: dict, document: dict) -> None:
    if not isinstance(document, dict):
        raise ValueError("Dataset JSON must be an object")
    rows = recipe.get("row_count")
    if (type(rows) is not int or rows < 1
            or document.get("row_count") != rows
            or (generation.get("schema_version") not in (3, 4)
                and generation["artifact"].get("row_count") != rows)
            or document.get("database_id") != recipe.get("dataset_id")):
        raise ValueError("Dataset identity or row count contradicts recipe")
    kind = recipe.get("kind")
    if kind == "price_gradients":
        from tools.datasets.price_gradients.contract import (
            _check_compact_outputs, _mixed_pairs, _selected_names)
        sensitivity = recipe["sensitivity"]
        names = [item["parameter"] for item in sensitivity["parameters"]]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate gradient parameter in recipe")
        orders = sensitivity.get("orders", ["first"])
        if any(order not in {"first", "diagonal_second", "mixed_second"}
               for order in orders) or len(set(orders)) != len(orders):
            raise ValueError("Invalid derivative orders in recipe")
        first = _selected_names(sensitivity, "first", names, "first" in orders)
        diagonal = _selected_names(sensitivity, "diagonal_second", names,
                                   "diagonal_second" in orders)
        mixed = ([a + "|" + b for a, b in _mixed_pairs(sensitivity, names)]
                 if "mixed_second" in orders else [])
        _check_compact_outputs(document, first, diagonal, mixed,
                               recipe.get("paths_per_price", 0),
                               "curve" in recipe.get("inputs", {}),
                               recipe.get("option_side"))
    elif kind == "prices":
        results = document.get("results")
        if not isinstance(results, list) or len(results) != rows:
            raise ValueError("Price result count contradicts recipe")
        for index, row in enumerate(results, 1):
            if not isinstance(row, dict) or row.get("id") != f"{index:06d}":
                raise ValueError(f"Invalid price result ID at row {index}")
            outputs = row.get("outputs", {})
            value = outputs.get("price") if isinstance(outputs, dict) else None
            error = outputs.get("standard_error") if isinstance(outputs, dict) else None
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or (recipe.get("paths_per_price", 0) > 0
                        and (type(error) not in (int, float)
                             or not math.isfinite(error) or error < 0))
                    or (error is not None and (type(error) not in (int, float)
                                               or not math.isfinite(error) or error < 0))):
                raise ValueError(f"Invalid price or standard error at row {index}")
    elif kind in {"model_parameters", "product_parameters", "curve_parameters",
                  "correlation_parameters"}:
        role = {"model_parameters": "models", "product_parameters": "products",
                "curve_parameters": "curves",
                "correlation_parameters": "correlations"}[kind]
        if not isinstance(document.get(role), list) or len(document[role]) != rows:
            raise ValueError("Parameter result count contradicts recipe")


def assess(recipe: dict, generation: dict, dataset: Path, root: Path | None = None) -> dict:
    """Check current metadata and JSON, without freezing historical code bytes."""
    def result(status: str, *reasons: str) -> dict:
        return {"status": status, "reasons": list(reasons),
                "certification": "not_assessed"}

    if not isinstance(recipe, dict) or not isinstance(generation, dict):
        return result("review_required", "Missing recipe or generation metadata")
    if generation.get("schema_version") not in (1, 2, 3, 4):
        return result("review_required", "Missing or unsupported generation metadata")
    try:
        validate_document(recipe, "recipe", dataset)
        validate_document(generation, "generation", dataset)
        version = generation["schema_version"]
        output = generation["output"] if version in (3, 4) else generation["artifact"]
        if output.get("sha256") != digest(dataset):
            return result("invalid", "Published JSON SHA256 differs from generation.yaml")
        if version in (1, 2) and generation["recipe"].get("path") is None:
            return result("invalid", "Generation receipt has no recipe path")
        if version in (2, 3, 4):
            if version in (2, 4) and output["path"] != recipe.get("output", {}).get("path"):
                return result("invalid", "Published JSON path contradicts recipe")
            declared_inputs = recipe.get("inputs", {})
            if set(generation["inputs"]) != set(declared_inputs):
                return result("invalid", "Input roles contradict recipe")
            if version in (2, 4) and {
                role: item["path"] for role, item in generation["inputs"].items()
            } != declared_inputs:
                return result("invalid", "Input paths contradict recipe")
            if root is not None:
                if contained_path(root, recipe["output"]["path"]).resolve() != dataset.resolve():
                    return result("invalid", "Published JSON path contradicts recipe")
                for role, relative in declared_inputs.items():
                    input_file = contained_path(root, relative)
                    checksum = (generation["inputs"][role] if version == 3
                                else generation["inputs"][role]["sha256"])
                    if input_file.is_file() and digest(input_file) != checksum:
                        return result("review_required", f"Current {role} input differs from generation.yaml")
        if recipe.get("kind") == "samples":
            from tools.datasets.generate_catalog import check_samples
            if version in (1, 2) and output.get("row_count") != recipe.get("row_count"):
                raise ValueError("Sample row count contradicts recipe")
            check_samples(dataset, {"generation": recipe["generation_details"],
                                    "rows": recipe["row_count"]}, recipe)
            document = None
        else:
            document = json.loads(dataset.read_text())
            _check_published_json(recipe, generation, document)
        if root is not None and recipe.get("kind") in {"prices", "price_gradients"}:
            for role, relative in recipe.get("inputs", {}).items():
                input_file = contained_path(root, relative)
                if not input_file.is_file():
                    continue
                source = json.loads(input_file.read_text())
                if role.startswith("model_") and role[6:].isdigit():
                    references = document.get("model_datasets", [])
                    index = int(role[6:])
                    reference = (references[index]
                                 if isinstance(references, list)
                                 and index < len(references) else None)
                else:
                    reference = document.get(role + "_dataset")
                if (not isinstance(reference, dict)
                        or reference.get("id") != source.get("database_id")
                        or reference.get("catalog") != source.get("catalog")):
                    raise ValueError(f"{role} dataset reference contradicts recipe input")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        return result("invalid", str(error))
    return result("compatible", "Current recipe and generation metadata describe the published JSON",
                  "This does not certify a numerical rerun or historical execution claims")
