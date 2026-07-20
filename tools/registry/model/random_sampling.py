"""Random parameter sampling helpers for registry model databases."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from tools.registry.common.philox import philox_uniforms


def philox_uniform_columns(
    *,
    seed: int,
    row_count: int,
    bounds: Mapping[str, tuple[float, float]],
    decimals: int = 12,
) -> dict[str, np.ndarray]:
    """Draw independent uniform columns with the project Philox generator."""

    if row_count < 1:
        raise ValueError("row_count must be positive.")
    columns: dict[str, np.ndarray] = {}
    for stream, (name, (lower, upper)) in enumerate(bounds.items()):
        if not lower < upper:
            raise ValueError(f"Invalid bounds for {name}: [{lower}, {upper}].")
        uniforms = philox_uniforms(seed, row_count, stream=stream)
        values = lower + (upper - lower) * uniforms
        columns[name] = np.round(values, decimals=decimals)
    return columns


def uniform_between(
    *,
    seed: int,
    lower: np.ndarray,
    upper: np.ndarray,
    stream: int = 0,
    decimals: int = 12,
) -> np.ndarray:
    """Draw row-wise uniforms on [lower_i, upper_i]."""

    if np.any(lower > upper):
        raise ValueError("At least one lower bound is greater than its upper bound.")
    uniforms = philox_uniforms(seed, int(lower.size), stream=stream).reshape(lower.shape)
    values = lower + (upper - lower) * uniforms
    return np.round(values, decimals=decimals)
