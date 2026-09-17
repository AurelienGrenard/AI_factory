"""Recoverable immutable publication with a generation marker written last.

The data rename and marker rename are individually atomic. Consumers only
consider a dataset published when its matching ``generation.yaml`` exists.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


def digest(path: Path) -> str | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular artifact: {path}")
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def contained_path(root: Path, relative: str) -> Path:
    """Reject absolute paths, traversal, and symlink redirection before writes."""
    part = Path(relative)
    if part.is_absolute() or not part.parts or ".." in part.parts:
        raise ValueError(f"Expected a repository-relative path: {relative}")
    path = root.resolve() / part
    if path.resolve() != path:
        raise ValueError(f"Artifact path traverses a symlink: {path}")
    return path


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(document, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def publish_pair(root: Path, work: Path, journal: Path, artifacts: list[dict]) -> None:
    """Publish a new immutable dataset, writing ``generation.yaml`` last.

    An artifact contains path, sha256 and previous_sha256. Work and repository
    must share a filesystem so publication moves, rather than copies, big JSONs.
    The caller holds the campaign lock and preserves this journal for resume.
    """
    if len(artifacts) != 2 or len({item["path"] for item in artifacts}) != 2:
        raise ValueError("Publication requires exactly two distinct artifacts")
    if (not artifacts[0]["path"].endswith(".json")
            or not artifacts[1]["path"].endswith("/generation.yaml")):
        raise ValueError("Publication order must be dataset JSON then generation marker")
    for item in artifacts:
        checksum = item["sha256"]
        if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            raise ValueError("A staged artifact requires a SHA-256 checksum")
    if root.stat().st_dev != work.stat().st_dev:
        raise ValueError("Staging and publication must share a filesystem")
    existing = json.loads(journal.read_text()) if journal.exists() else None
    if existing is not None and existing["artifacts"] != artifacts:
        raise ValueError("Publication identity changed; start a new campaign")
    resuming = existing is not None and existing["state"] in ("publishing", "complete")

    # Verify the whole pair before touching either published file.
    for item in artifacts:
        destination = contained_path(root, item["path"])
        source = contained_path(work, item["path"])
        current = digest(destination)
        if current not in {None, item["sha256"]}:
            raise ValueError(
                f"Published datasets are immutable; choose a new dataset id: {destination}"
            )
        if item["previous_sha256"] not in {None, item["sha256"]}:
            raise ValueError(
                f"Campaign was frozen over a different published artifact: {destination}"
            )
        if not (resuming and current == item["sha256"]) and digest(source) != item["sha256"]:
            raise ValueError(f"Staged artifact missing or changed: {source}")

    if not resuming:
        save_json(journal, {"state": "publishing", "artifacts": artifacts})

    for item in artifacts:
        destination = contained_path(root, item["path"])
        current = digest(destination)
        if current == item["sha256"]:
            continue
        if current is not None:
            raise ValueError(f"Published artifact changed during publication: {destination}")
        source = contained_path(work, item["path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(source, destination)
        sync_directory(destination.parent)
        sync_directory(source.parent)
    save_json(journal, {"state": "complete", "artifacts": artifacts})
