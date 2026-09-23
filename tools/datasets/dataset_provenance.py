"""Record generation evidence and compare dataset reuse without claiming certification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

import yaml

from tools.datasets.artifact_publication import contained_path, digest


SCHEMA_VERSION = 1


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


def attach_generation(work: Path, job: dict, state: dict) -> None:
    """Complete a native receipt without copying the canonical recipe into it."""
    path = contained_path(work, job["generation"])
    record = yaml.safe_load(path.read_text())
    if not isinstance(record, dict) or record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Native output contains an invalid generation receipt")
    if any(key in record for key in ("recipe", "inputs", "provenance", "record_sha256")):
        raise ValueError("Native generation receipt unexpectedly contains provenance")
    record["recipe"] = {
        "path": job["recipe"],
        "sha256": job["recipe_sha256"],
    }
    record.setdefault("artifact", {})["sha256"] = digest(
        contained_path(work, job["dataset"])
    )
    record["inputs"] = job["semantic_inputs"]
    record["execution"].update({
        "binary_sha256": job["binary_sha256"],
        "generator_sha256": job["generator_sha256"],
        "declared_method": job["declared_method"],
        "launch_plan": job.get("launch_plan"),
        "build_hashes": state["build_hashes"],
    })
    record["provenance"] = {
        "revision": state["revision"],
        "source_archive_sha256": state["source_archive_sha256"],
        "runtime_observation": job.get(
            "gpu_before", {"unavailable": "not recorded"}
        ),
        "generator": job["generator"],
        "input_files": {
            name: state["input_hashes"][name] for name in job["inputs"]
        },
    }
    record["record_sha256"] = fingerprint(record)
    path.write_text(yaml.safe_dump(record, sort_keys=False))


def assess(
    recipe: dict,
    generation: dict,
    dataset: Path,
    candidate: dict | None,
    recipe_sha256: str,
) -> dict:
    """Conservative reuse decision; no modification, automatic acceptance or regeneration."""
    record = generation
    def result(status, *reasons):
        return {"status": status, "reasons": list(reasons), "certification": "not_assessed",
                "generation_record_sha256": record.get("record_sha256") if isinstance(record, dict) else None,
                "candidate_sha256": fingerprint(candidate) if candidate else None,
                "candidate": candidate}

    if (not isinstance(record, dict) or type(record.get("schema_version")) is not int
            or record["schema_version"] != SCHEMA_VERSION):
        return result("review_required", "Missing or unsupported generation provenance; do not backfill from today's code")
    for key, value in (
        ("artifact.sha256", record.get("artifact", {}).get("sha256")),
        ("recipe.sha256", record.get("recipe", {}).get("sha256")),
        ("record_sha256", record.get("record_sha256")),
    ):
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            return result("invalid", "Malformed generation checksum: " + key)
    if any(not isinstance(record.get(key), dict) for key in ("artifact", "recipe", "inputs", "execution", "provenance")):
        return result("invalid", "Incomplete generation record")
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    if record.get("record_sha256") != fingerprint(unsigned):
        return result("invalid", "Generation record checksum mismatch")
    if record["artifact"].get("sha256") != digest(dataset):
        return result("invalid", "Dataset content missing or changed")
    if record["recipe"].get("sha256") != recipe_sha256:
        return result("regenerate", "Canonical recipe bytes changed")
    if recipe.get("output", {}).get("path") is None:
        return result("invalid", "Canonical recipe has no output path")
    if candidate is None:
        return {**result("review_required", "Current generation has not been inspected"),
                "candidate_required": True}
    if record["recipe"].get("sha256") != candidate["recipe_sha256"]:
        return result("regenerate", "Dataset identity, shape, seeds or numerical recipe differs")
    if record.get("inputs") != candidate["inputs"]:
        return result("regenerate", "Ordered parameter content differs (paths and presentation are excluded)")
    execution = record["execution"]
    differences = [key for key in ("binary_sha256", "generator_sha256", "declared_method", "launch_plan", "build_hashes")
                   if execution.get(key) != candidate["execution"].get(key)]
    if differences:
        return result("review_required", "Changed generation evidence: " + ", ".join(differences),
                      "A refactor is not proof of changed mathematics; compare affected outputs before accepting reuse")
    return result("compatible", "Recorded inputs, recipe, executable and build configuration match",
                  "This is recipe compatibility, not bitwise portability or financial certification")
