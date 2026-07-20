"""Validation slices of production curve databases."""

from __future__ import annotations

from pathlib import Path

from tools.registry.common.slicing import (
    DEFAULT_VALIDATION_ROW_COUNT,
    write_database_slice,
)

__all__ = ["DEFAULT_VALIDATION_ROW_COUNT", "write_curve_slice"]


def write_curve_slice(
    *,
    project_root: Path,
    source_id: str,
    target_id: str,
    row_count: int = DEFAULT_VALIDATION_ROW_COUNT,
) -> Path:
    return write_database_slice(
        project_root=project_root,
        kind="curves",
        source_id=source_id,
        target_id=target_id,
        row_key="curves",
        row_count=row_count,
    )
