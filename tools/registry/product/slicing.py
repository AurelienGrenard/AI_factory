"""Validation slices of production product-parameter databases."""

from __future__ import annotations

from pathlib import Path

from tools.registry.common.slicing import (
    DEFAULT_VALIDATION_ROW_COUNT,
    write_database_slice,
)

__all__ = ["DEFAULT_VALIDATION_ROW_COUNT", "write_product_slice"]


def write_product_slice(
    *,
    project_root: Path,
    source_id: str,
    target_id: str,
    row_count: int = DEFAULT_VALIDATION_ROW_COUNT,
) -> Path:
    return write_database_slice(
        project_root=project_root,
        kind="products",
        source_id=source_id,
        target_id=target_id,
        row_key="products",
        row_count=row_count,
    )
