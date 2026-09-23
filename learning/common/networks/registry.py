"""Small explicit registry for configurable network architectures."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import nn


NetworkBuilder = Callable[[int, int, dict[str, Any]], nn.Module]
_BUILDERS: dict[str, NetworkBuilder] = {}


def register_network(name: str) -> Callable[[NetworkBuilder], NetworkBuilder]:
    def decorator(builder: NetworkBuilder) -> NetworkBuilder:
        if name in _BUILDERS:
            raise ValueError(f"Network {name!r} is already registered")
        _BUILDERS[name] = builder
        return builder

    return decorator


def build_network(
    specification: dict[str, Any], *, input_dimension: int, output_dimension: int
) -> nn.Module:
    name = str(specification.get("name", "mlp"))
    try:
        builder = _BUILDERS[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown network {name!r}; registered networks: {sorted(_BUILDERS)}"
        ) from error
    return builder(input_dimension, output_dimension, specification)
