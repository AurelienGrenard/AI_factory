"""Configurable latent-noise sampling for conditional generators."""

from __future__ import annotations

import math

import torch


SUPPORTED_NOISE_DISTRIBUTIONS = {"normal", "uniform"}


def sample_noise(
    count: int,
    latent_dimension: int,
    distribution: str,
    *,
    device: torch.device,
    random: torch.Generator | None = None,
    uniform_minimum: float = -1.0,
    uniform_maximum: float = 1.0,
) -> torch.Tensor:
    if count <= 0 or latent_dimension <= 0:
        raise ValueError(
            "Noise count and latent dimension must be positive"
        )
    if distribution not in SUPPORTED_NOISE_DISTRIBUTIONS:
        raise ValueError(
            f"Unknown noise distribution {distribution!r}; "
            f"choose from {sorted(SUPPORTED_NOISE_DISTRIBUTIONS)}"
        )
    if distribution == "uniform" and (
        not math.isfinite(uniform_minimum)
        or not math.isfinite(uniform_maximum)
        or uniform_minimum >= uniform_maximum
    ):
        raise ValueError(
            "Uniform noise bounds must be finite and increasing"
        )
    shape = (count, latent_dimension)
    if random is None:
        if distribution == "normal":
            return torch.randn(shape, device=device)
        unit = torch.rand(shape, device=device)
        return uniform_minimum + (
            uniform_maximum - uniform_minimum
        ) * unit

    if distribution == "normal":
        values = torch.randn(
            shape, generator=random, device="cpu"
        )
    else:
        unit = torch.rand(shape, generator=random, device="cpu")
        values = uniform_minimum + (
            uniform_maximum - uniform_minimum
        ) * unit
    return values.to(device)
