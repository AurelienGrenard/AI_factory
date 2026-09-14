"""Stream published model-terminal samples with leakage-safe group splits."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterator, Literal


Split = Literal["train", "validation", "test"]
_SAMPLES_START = re.compile(r'^\s*"samples"\s*:\s*\[\s*$')


@dataclass(frozen=True)
class TerminalSchema:
    database_id: str
    model_family: str
    row_count: int
    parameter_count: int
    paths_per_parameter: int
    parameter_names: tuple[str, ...]
    observable_names: tuple[str, ...]
    days_per_year: int

    @property
    def conditioning_names(self) -> tuple[str, ...]:
        return (*self.parameter_names, "T")


@dataclass(frozen=True)
class TerminalSample:
    conditioning: tuple[float, ...]
    values: tuple[float, ...]
    maturity_days: int
    parameter_group: int
    split: Split


def _read_header(stream) -> dict:
    lines: list[str] = []
    for line in stream:
        if _SAMPLES_START.match(line):
            lines.append(line)
            return json.loads("".join(lines) + "]}")
        lines.append(line)
    raise ValueError("Missing line-streamed samples array")


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _schema_from_header(header: dict, first: dict) -> TerminalSchema:
    construction = header["construction"]
    row_count = _positive_int(header["row_count"], "row_count")
    parameter_count = _positive_int(construction["parameter_count"], "parameter_count")
    paths_per_parameter = _positive_int(
        construction["paths_per_parameter"], "paths_per_parameter"
    )
    if row_count != parameter_count * paths_per_parameter:
        raise ValueError("row_count does not match construction shape")
    if construction.get("row_order") != "parameter-major, then path-major":
        raise ValueError("Unsupported sample row order")
    days_per_year = _positive_int(
        header["time_convention"]["days_per_year"], "days_per_year"
    )
    parameter_names = tuple(sorted(first["parameters"]))
    observable_names = tuple(sorted(first["values"]))
    if not parameter_names or not observable_names:
        raise ValueError("Empty parameters or terminal observables")
    return TerminalSchema(
        database_id=str(header["database_id"]),
        model_family=str(header["model_family"]),
        row_count=row_count,
        parameter_count=parameter_count,
        paths_per_parameter=paths_per_parameter,
        parameter_names=parameter_names,
        observable_names=observable_names,
        days_per_year=days_per_year,
    )


def _split_for_group(group: int, seed: int) -> Split:
    digest = hashlib.blake2b(
        f"{seed}:{group}".encode("ascii"), digest_size=8
    ).digest()
    bucket = int.from_bytes(digest, "big") % 10
    return "train" if bucket < 8 else "validation" if bucket == 8 else "test"


def _sample_from_row(row: dict, schema: TerminalSchema, index: int, seed: int) -> TerminalSample:
    if tuple(sorted(row["parameters"])) != schema.parameter_names:
        raise ValueError(f"Row {index}: parameter names differ from schema")
    if tuple(sorted(row["values"])) != schema.observable_names:
        raise ValueError(f"Row {index}: observable names differ from schema")
    days = row["maturity_days"]
    if type(days) is not int or days <= 0:
        raise ValueError(f"Row {index}: invalid maturity_days")
    theta = tuple(float(row["parameters"][name]) for name in schema.parameter_names)
    values = tuple(float(row["values"][name]) for name in schema.observable_names)
    maturity = float(row["T"])
    if not all(math.isfinite(x) for x in (*theta, maturity, *values)):
        raise ValueError(f"Row {index}: non-finite value")
    if not math.isclose(maturity, days / schema.days_per_year, rel_tol=1e-6, abs_tol=1e-7):
        raise ValueError(f"Row {index}: T differs from maturity_days / days_per_year")
    group = index // schema.paths_per_parameter
    return TerminalSample(
        conditioning=(*theta, maturity),
        values=values,
        maturity_days=days,
        parameter_group=group,
        split=_split_for_group(group, seed),
    )


def iter_terminal_samples(
    path: str | Path, *, seed: int = 0, limit: int | None = None
) -> Iterator[tuple[TerminalSchema, TerminalSample]]:
    """Yield schema and rows; validate the declared count on a complete scan."""
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")
    with Path(path).open(encoding="utf-8") as stream:
        header = _read_header(stream)
        schema: TerminalSchema | None = None
        count = 0
        for line in stream:
            stripped = line.strip()
            if stripped == "]" or stripped == "],":
                break
            if not stripped:
                continue
            if limit is not None and count >= limit:
                return
            row = json.loads(stripped.removesuffix(","))
            if schema is None:
                schema = _schema_from_header(header, row)
            yield schema, _sample_from_row(row, schema, count, seed)
            count += 1
        if count != _positive_int(header["row_count"], "row_count"):
            raise ValueError(f"Dataset has {count} rows, header declares {header['row_count']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    counts = {"train": 0, "validation": 0, "test": 0}
    schema = None
    for schema, sample in iter_terminal_samples(args.dataset, seed=args.seed, limit=args.limit):
        counts[sample.split] += 1
    if schema is None:
        raise SystemExit("No samples found")
    print(json.dumps({
        "database_id": schema.database_id,
        "model_family": schema.model_family,
        "conditioning": schema.conditioning_names,
        "observables": schema.observable_names,
        "rows_read": sum(counts.values()),
        "split_counts": counts,
    }, indent=2))


if __name__ == "__main__":
    main()
