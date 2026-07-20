"""Row grouping and output ordering shared by result executors."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tools.common.time_grid import step_count_for_maturity

IndexedRow = tuple[int, dict[str, Any]]
RowGroup = tuple[int, list[IndexedRow]]


def rows_grouped_by_step_count(
    rows: list[dict[str, Any]],
    product_by_id: dict[str, Any],
    target_dt: str | float | int,
) -> list[RowGroup]:
    grouped: dict[int, list[IndexedRow]] = {}
    for index, row in enumerate(rows):
        maturity = float(product_by_id[row["product_id"]]["maturity"])
        num_steps = step_count_for_maturity(maturity, target_dt)
        grouped.setdefault(num_steps, []).append((index, row))
    return list(grouped.items())


def ordered_outputs_from_groups(
    rows: list[dict[str, Any]],
    groups: list[RowGroup],
    run_group: Callable[
        [int, list[dict[str, Any]]],
        tuple[list[dict[str, float]], dict[str, float]],
    ],
) -> tuple[list[dict[str, float]], dict[str, float]]:
    outputs: list[dict[str, float] | None] = [None] * len(rows)
    timing: dict[str, float] = {"wall_seconds": 0.0}
    for num_steps, indexed_rows in groups:
        group_rows = [row for _, row in indexed_rows]
        group_outputs, group_timing = run_group(num_steps, group_rows)
        for (index, _), output in zip(indexed_rows, group_outputs, strict=True):
            outputs[index] = output
        for name, value in group_timing.items():
            timing[name] = timing.get(name, 0.0) + float(value)
    return [output for output in outputs if output is not None], timing
