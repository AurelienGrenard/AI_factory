"""Configurable fully connected baseline for scalar pricing maps."""

from __future__ import annotations

from typing import Any

from torch import nn

from .registry import register_network


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


@register_network("mlp")
def build_mlp(
    input_dimension: int, output_dimension: int, specification: dict[str, Any]
) -> nn.Module:
    hidden_sizes = specification.get("hidden_sizes", [256, 256, 256])
    if not isinstance(hidden_sizes, list) or not hidden_sizes:
        raise ValueError("network.hidden_sizes must be a non-empty list")
    widths = [int(width) for width in hidden_sizes]
    if any(width <= 0 for width in widths):
        raise ValueError("All hidden layer widths must be positive")
    activation_name = str(specification.get("activation", "silu"))
    try:
        activation = _ACTIVATIONS[activation_name]
    except KeyError as error:
        raise ValueError(
            f"Unknown activation {activation_name!r}; choose from {sorted(_ACTIVATIONS)}"
        ) from error

    layers: list[nn.Module] = []
    previous = input_dimension
    for width in widths:
        layers.extend((nn.Linear(previous, width), activation()))
        previous = width
    layers.append(nn.Linear(previous, output_dimension))
    return nn.Sequential(*layers)
