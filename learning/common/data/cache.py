"""Prepare large pricing JSON files as compact memory-mapped tensor arrays."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Iterator

import numpy as np

from .contracts import GradientTarget, PricingTensorSchema
from .json_stream import open_object_array
from .references import EntityTable, find_repository_root, load_entity_tables


_CACHE_SCHEMA_VERSION = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _infer_gradients(
    header: dict, outputs: dict, feature_names: tuple[str, ...]
) -> tuple[GradientTarget, ...]:
    def resolve(parameter: str) -> tuple[int, str]:
        suffix = f".{parameter}"
        matches = [
            (index, name)
            for index, name in enumerate(feature_names)
            if name == parameter or name.endswith(suffix)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Sensitivity parameter {parameter!r} resolves to {len(matches)} features; "
                "publish an unambiguous role-qualified input"
            )
        return matches[0]

    if "gradients" in outputs:
        if "delta" in outputs:
            raise ValueError("A pricing row cannot mix delta and gradients contracts")
        values = outputs["gradients"]
        sensitivity = header.get("sensitivity")
        parameters = sensitivity.get("parameters") if isinstance(sensitivity, dict) else None
        if not isinstance(values, dict) or not isinstance(parameters, list) or not parameters:
            raise ValueError(
                "Gradient outputs require sensitivity.parameters metadata"
            )
        names = [
            str(item.get("parameter", "")) if isinstance(item, dict) else ""
            for item in parameters
        ]
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("Gradient sensitivity parameters must be non-empty and unique")
        if list(values) != names:
            raise ValueError("Gradient outputs disagree with sensitivity parameter order")
        errors = outputs.get("gradient_standard_errors")
        if errors is not None and (not isinstance(errors, dict) or list(errors) != names):
            raise ValueError("Gradient standard errors disagree with sensitivity parameter order")
        targets = []
        for parameter in names:
            index, name = resolve(parameter)
            targets.append(
                GradientTarget(
                    source_name=f"gradients:{parameter}",
                    value_name="price",
                    wrt_name=name,
                    wrt_index=index,
                    standard_error_name=(
                        f"gradient_standard_errors:{parameter}"
                        if errors is not None else None
                    ),
                )
            )
        return tuple(targets)

    if "delta" not in outputs:
        return ()
    sensitivity = header.get("sensitivity")
    if not isinstance(sensitivity, dict) or not sensitivity.get("parameter"):
        raise ValueError("A delta output requires sensitivity.parameter metadata")
    parameter = str(sensitivity["parameter"])
    index, name = resolve(parameter)
    return (
        GradientTarget(
            source_name="delta",
            value_name="price",
            wrt_name=name,
            wrt_index=index,
            standard_error_name=(
                "delta_standard_error" if "delta_standard_error" in outputs else None
            ),
        ),
    )


def _cache_identity(dataset_path: Path, tables: tuple[EntityTable, ...]) -> dict[str, object]:
    return {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "dataset_sha256": _sha256(dataset_path),
        "inputs": {
            table.role: _sha256(table.path)
            for table in tables
        },
    }


def _identity_key(identity: dict[str, object]) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_manifest(path: Path, value: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


@dataclass
class PreparedPricingDataset:
    """Read-only memory maps plus the schema and immutable cache identity."""

    directory: Path
    manifest: dict[str, object]
    schema: PricingTensorSchema
    features: np.ndarray
    values: np.ndarray
    gradients: np.ndarray
    value_standard_errors: np.ndarray
    gradient_standard_errors: np.ndarray
    entity_ordinals: np.ndarray

    @property
    def fingerprint(self) -> str:
        return str(self.manifest["cache_key"])

    @classmethod
    def open(cls, directory: str | Path) -> "PreparedPricingDataset":
        directory = Path(directory)
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != _CACHE_SCHEMA_VERSION:
            raise ValueError("Unsupported prepared pricing cache schema")
        schema = PricingTensorSchema.from_dict(manifest["tensor_schema"])

        def array(name: str) -> np.ndarray:
            path = directory / str(manifest["arrays"][name])
            if not path.is_file():
                raise FileNotFoundError(f"Missing prepared array: {path}")
            return np.load(path, mmap_mode="r")

        prepared = cls(
            directory=directory,
            manifest=manifest,
            schema=schema,
            features=array("features"),
            values=array("values"),
            gradients=array("gradients"),
            value_standard_errors=array("value_standard_errors"),
            gradient_standard_errors=array("gradient_standard_errors"),
            entity_ordinals=array("entity_ordinals"),
        )
        if prepared.features.shape != (schema.row_count, len(schema.feature_names)):
            raise ValueError("Prepared feature shape disagrees with its manifest")
        if prepared.values.shape != (schema.row_count, len(schema.value_names)):
            raise ValueError("Prepared value shape disagrees with its manifest")
        if prepared.gradients.shape != (schema.row_count, len(schema.gradients)):
            raise ValueError("Prepared gradient shape disagrees with its manifest")
        return prepared


def _feature_row(row: dict, tables: tuple[EntityTable, ...]) -> tuple[list[float], list[int]]:
    features: list[float] = []
    ordinals: list[int] = []
    for table in tables:
        field = f"{table.role}_id"
        entity_id = str(row.get(field, ""))
        entity = table.rows.get(entity_id)
        if entity is None:
            raise ValueError(f"Unknown {field} {entity_id!r}")
        features.extend(entity.values)
        ordinals.append(entity.ordinal)
    return features, ordinals


def _finite_output(outputs: dict, name: str, row_index: int) -> float:
    try:
        container, separator, field = name.partition(":")
        value = float(outputs[container][field] if separator else outputs[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Row {row_index}: missing numeric output {name!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"Row {row_index}: non-finite output {name!r}")
    return value


def _optional_standard_error(outputs: dict, name: str | None, row_index: int) -> float:
    if name is None or name.partition(":")[0] not in outputs:
        return math.nan
    value = _finite_output(outputs, name, row_index)
    if value < 0.0:
        raise ValueError(f"Row {row_index}: negative standard error {name!r}")
    return value


def prepare_pricing_dataset(
    dataset: str | Path,
    *,
    cache_root: str | Path | None = None,
    force: bool = False,
) -> PreparedPricingDataset:
    """Validate and convert one published price/price-delta JSON dataset."""

    dataset_path = Path(dataset).resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    with open_object_array(dataset_path, "results") as (header, rows):
        try:
            first = next(rows)
        except StopIteration as error:
            raise ValueError("Price dataset contains no results") from error
    row_count = _positive_int(header.get("row_count"), "row_count")
    tables = load_entity_tables(dataset_path, header)
    feature_names = tuple(name for table in tables for name in table.feature_names)
    outputs = first.get("outputs")
    if not isinstance(outputs, dict) or "price" not in outputs:
        raise ValueError("Price result rows must contain outputs.price")
    gradients = _infer_gradients(header, outputs, feature_names)
    schema = PricingTensorSchema(
        database_id=str(header.get("database_id", "")),
        row_count=row_count,
        feature_names=feature_names,
        value_names=("price",),
        gradients=gradients,
        entity_roles=tuple(table.role for table in tables),
    )
    identity = _cache_identity(dataset_path, tables)
    key = _identity_key(identity)
    if cache_root is None:
        cache_root = find_repository_root(dataset_path) / "artifacts" / "learning" / "cache"
    root = Path(cache_root)
    directory = root / key
    if directory.is_dir() and not force:
        prepared = PreparedPricingDataset.open(directory)
        if prepared.manifest.get("identity") != identity:
            raise ValueError("Prepared cache identity collision")
        return prepared

    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".{key}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        feature_array = np.lib.format.open_memmap(
            temporary / "features.npy", mode="w+", dtype=np.float32,
            shape=(row_count, len(feature_names)),
        )
        value_array = np.lib.format.open_memmap(
            temporary / "values.npy", mode="w+", dtype=np.float32,
            shape=(row_count, 1),
        )
        gradient_array = np.lib.format.open_memmap(
            temporary / "gradients.npy", mode="w+", dtype=np.float32,
            shape=(row_count, len(gradients)),
        )
        value_se_array = np.lib.format.open_memmap(
            temporary / "value_standard_errors.npy", mode="w+", dtype=np.float32,
            shape=(row_count, 1),
        )
        gradient_se_array = np.lib.format.open_memmap(
            temporary / "gradient_standard_errors.npy", mode="w+", dtype=np.float32,
            shape=(row_count, len(gradients)),
        )
        ordinal_array = np.lib.format.open_memmap(
            temporary / "entity_ordinals.npy", mode="w+", dtype=np.int32,
            shape=(row_count, len(tables)),
        )

        with open_object_array(dataset_path, "results") as (second_header, row_iterator):
            if second_header.get("database_id") != schema.database_id:
                raise ValueError("Dataset header changed while preparing its cache")
            count = 0
            for count, row in enumerate(row_iterator, start=1):
                index = count - 1
                if index >= row_count:
                    raise ValueError("Dataset contains more rows than row_count")
                row_outputs = row.get("outputs")
                if not isinstance(row_outputs, dict):
                    raise ValueError(f"Row {index}: missing outputs object")
                features, ordinals = _feature_row(row, tables)
                feature_array[index] = features
                ordinal_array[index] = ordinals
                value_array[index, 0] = _finite_output(row_outputs, "price", index)
                value_se_array[index, 0] = _optional_standard_error(
                    row_outputs, "standard_error", index
                )
                for gradient_index, gradient in enumerate(gradients):
                    gradient_array[index, gradient_index] = _finite_output(
                        row_outputs, gradient.source_name, index
                    )
                    gradient_se_array[index, gradient_index] = _optional_standard_error(
                        row_outputs, gradient.standard_error_name, index
                    )
            if count != row_count:
                raise ValueError(
                    f"Dataset contains {count} rows but declares row_count={row_count}"
                )

        for array in (
            feature_array,
            value_array,
            gradient_array,
            value_se_array,
            gradient_se_array,
            ordinal_array,
        ):
            array.flush()
        manifest = {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "cache_key": key,
            "identity": identity,
            "sources": {
                "dataset": str(dataset_path),
                "inputs": {table.role: str(table.path.resolve()) for table in tables},
            },
            "tensor_schema": schema.to_dict(),
            "arrays": {
                "features": "features.npy",
                "values": "values.npy",
                "gradients": "gradients.npy",
                "value_standard_errors": "value_standard_errors.npy",
                "gradient_standard_errors": "gradient_standard_errors.npy",
                "entity_ordinals": "entity_ordinals.npy",
            },
        }
        _write_manifest(temporary / "manifest.json", manifest)
        if directory.exists():
            if force:
                shutil.rmtree(directory)
            else:
                shutil.rmtree(temporary)
                return PreparedPricingDataset.open(directory)
        temporary.rename(directory)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return PreparedPricingDataset.open(directory)
