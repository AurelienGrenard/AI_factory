"""Prepare line-streamed terminal samples as memory-mapped tensor arrays."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from learning.common.terminal_samples import TerminalSchema, iter_terminal_samples


_CACHE_SCHEMA_VERSION = 1
_SPLIT_CODES = {"train": np.uint8(0), "validation": np.uint8(1), "test": np.uint8(2)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_key(identity: dict[str, object]) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


@dataclass(frozen=True)
class TerminalTensorSchema:
    database_id: str
    model_family: str
    row_count: int
    parameter_count: int
    paths_per_parameter: int
    conditioning_names: tuple[str, ...]
    observable_names: tuple[str, ...]
    available_observable_names: tuple[str, ...]
    days_per_year: int

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        for name in (
            "conditioning_names",
            "observable_names",
            "available_observable_names",
        ):
            value[name] = list(value[name])
        return value

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "TerminalTensorSchema":
        copied = dict(value)
        for name in (
            "conditioning_names",
            "observable_names",
            "available_observable_names",
        ):
            copied[name] = tuple(str(item) for item in copied[name])
        return cls(**copied)


@dataclass
class PreparedTerminalDataset:
    directory: Path
    manifest: dict[str, object]
    schema: TerminalTensorSchema
    conditions: np.ndarray
    observations: np.ndarray
    maturity_days: np.ndarray
    parameter_groups: np.ndarray
    split_codes: np.ndarray

    @property
    def fingerprint(self) -> str:
        return str(self.manifest["cache_key"])

    @classmethod
    def open(cls, directory: str | Path) -> "PreparedTerminalDataset":
        directory = Path(directory)
        manifest = json.loads(
            (directory / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("schema_version") != _CACHE_SCHEMA_VERSION:
            raise ValueError("Unsupported prepared terminal cache schema")
        schema = TerminalTensorSchema.from_dict(manifest["tensor_schema"])

        def array(name: str) -> np.ndarray:
            path = directory / str(manifest["arrays"][name])
            if not path.is_file():
                raise FileNotFoundError(f"Missing prepared terminal array: {path}")
            return np.load(path, mmap_mode="r")

        prepared = cls(
            directory=directory,
            manifest=manifest,
            schema=schema,
            conditions=array("conditions"),
            observations=array("observations"),
            maturity_days=array("maturity_days"),
            parameter_groups=array("parameter_groups"),
            split_codes=array("split_codes"),
        )
        expected = {
            "conditions": (schema.row_count, len(schema.conditioning_names)),
            "observations": (schema.row_count, len(schema.observable_names)),
            "maturity_days": (schema.row_count,),
            "parameter_groups": (schema.row_count,),
            "split_codes": (schema.row_count,),
        }
        for name, shape in expected.items():
            if getattr(prepared, name).shape != shape:
                raise ValueError(
                    f"Prepared terminal {name} shape disagrees with manifest"
                )
        if not set(np.unique(prepared.split_codes)).issubset({0, 1, 2}):
            raise ValueError("Prepared terminal split codes are invalid")
        return prepared


def _peek_schema(path: Path, split_seed: int) -> TerminalSchema:
    iterator = iter_terminal_samples(path, seed=split_seed, limit=1)
    try:
        schema, _ = next(iterator)
    except StopIteration as error:
        raise ValueError(f"Terminal dataset is empty: {path}") from error
    finally:
        iterator.close()
    return schema


def _selected_observables(
    schema: TerminalSchema, observables: list[str] | tuple[str, ...] | None
) -> tuple[str, ...]:
    if observables is None:
        return schema.observable_names
    selected = tuple(str(name) for name in observables)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("observables must be a non-empty list of unique names")
    missing = sorted(set(selected) - set(schema.observable_names))
    if missing:
        raise ValueError(
            f"Unknown terminal observables {missing}; "
            f"available: {schema.observable_names}"
        )
    return selected


def prepare_terminal_dataset(
    dataset_path: str | Path,
    *,
    cache_root: str | Path | None = None,
    split_seed: int = 0,
    observables: list[str] | tuple[str, ...] | None = None,
    chunk_size: int = 65_536,
) -> PreparedTerminalDataset:
    """Validate and convert a terminal-sample JSON into an immutable mmap cache."""

    dataset_path = Path(dataset_path).resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    source_schema = _peek_schema(dataset_path, split_seed)
    selected = _selected_observables(source_schema, observables)
    identity: dict[str, object] = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "dataset_sha256": _sha256(dataset_path),
        "split_seed": int(split_seed),
        "observables": list(selected),
    }
    key = _identity_key(identity)
    root = Path(cache_root or "artifacts/learning/cache/terminal")
    root.mkdir(parents=True, exist_ok=True)
    destination = root / key
    if (destination / "manifest.json").is_file():
        return PreparedTerminalDataset.open(destination)
    if destination.exists():
        raise ValueError(f"Incomplete terminal cache already exists: {destination}")

    temporary = Path(tempfile.mkdtemp(prefix=f".{key}.tmp-", dir=root))
    arrays = {
        "conditions": "conditions.npy",
        "observations": "observations.npy",
        "maturity_days": "maturity_days.npy",
        "parameter_groups": "parameter_groups.npy",
        "split_codes": "split_codes.npy",
    }
    try:
        conditions = np.lib.format.open_memmap(
            temporary / arrays["conditions"],
            mode="w+",
            dtype=np.float32,
            shape=(
                source_schema.row_count,
                len(source_schema.conditioning_names),
            ),
        )
        values = np.lib.format.open_memmap(
            temporary / arrays["observations"],
            mode="w+",
            dtype=np.float32,
            shape=(source_schema.row_count, len(selected)),
        )
        maturity_days = np.lib.format.open_memmap(
            temporary / arrays["maturity_days"],
            mode="w+",
            dtype=np.int32,
            shape=(source_schema.row_count,),
        )
        groups = np.lib.format.open_memmap(
            temporary / arrays["parameter_groups"],
            mode="w+",
            dtype=np.int64,
            shape=(source_schema.row_count,),
        )
        split_codes = np.lib.format.open_memmap(
            temporary / arrays["split_codes"],
            mode="w+",
            dtype=np.uint8,
            shape=(source_schema.row_count,),
        )
        available_positions = {
            name: position
            for position, name in enumerate(source_schema.observable_names)
        }
        selected_positions = np.asarray(
            [available_positions[name] for name in selected], dtype=np.int64
        )
        iterator = iter_terminal_samples(dataset_path, seed=split_seed)
        first_schema, first_sample = next(iterator)
        if first_schema != source_schema:
            raise ValueError("Terminal dataset schema changed while preparing cache")
        first_pair = (first_schema, first_sample)
        written = 0
        for block_start in range(0, source_schema.row_count, chunk_size):
            source = (
                itertools.chain((first_pair,), iterator)
                if block_start == 0
                else iterator
            )
            pairs = list(itertools.islice(source, chunk_size))
            if not pairs:
                break
            if any(schema != source_schema for schema, _ in pairs):
                raise ValueError(
                    "Terminal dataset schema changed while preparing cache"
                )
            block = [sample for _, sample in pairs]
            block_end = block_start + len(block)
            conditions[block_start:block_end] = np.asarray(
                [sample.conditioning for sample in block], dtype=np.float32
            )
            all_values = np.asarray(
                [sample.values for sample in block], dtype=np.float32
            )
            values[block_start:block_end] = all_values[:, selected_positions]
            maturity_days[block_start:block_end] = np.asarray(
                [sample.maturity_days for sample in block], dtype=np.int32
            )
            groups[block_start:block_end] = np.asarray(
                [sample.parameter_group for sample in block], dtype=np.int64
            )
            split_codes[block_start:block_end] = np.asarray(
                [_SPLIT_CODES[sample.split] for sample in block],
                dtype=np.uint8,
            )
            written = block_end
        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            raise ValueError(
                "Terminal dataset contains more rows than declared"
            )
        if written != source_schema.row_count:
            raise ValueError(
                f"Prepared {written} terminal rows, "
                f"expected {source_schema.row_count}"
            )
        for array in (conditions, values, maturity_days, groups, split_codes):
            array.flush()
        tensor_schema = TerminalTensorSchema(
            database_id=source_schema.database_id,
            model_family=source_schema.model_family,
            row_count=source_schema.row_count,
            parameter_count=source_schema.parameter_count,
            paths_per_parameter=source_schema.paths_per_parameter,
            conditioning_names=source_schema.conditioning_names,
            observable_names=selected,
            available_observable_names=source_schema.observable_names,
            days_per_year=source_schema.days_per_year,
        )
        manifest: dict[str, object] = {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "cache_key": key,
            "identity": identity,
            "source": str(dataset_path),
            "tensor_schema": tensor_schema.to_dict(),
            "arrays": arrays,
        }
        _write_json(temporary / "manifest.json", manifest)
        try:
            os.replace(temporary, destination)
        except OSError:
            if not (destination / "manifest.json").is_file():
                raise
            shutil.rmtree(temporary)
        return PreparedTerminalDataset.open(destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
