"""Device helpers shared by simulations."""

from __future__ import annotations

import torch


def resolve_device(device: str) -> torch.device:
    """Resolve 'cpu', 'cuda', or 'auto' into a torch device."""

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return resolved


def resolve_random_device(
    random_device: str,
    target_device: torch.device,
) -> torch.device:
    """Resolve the device used to generate random numbers."""

    if random_device == "target":
        return target_device
    return resolve_device(random_device)


def seeded_generator(seed: int, device: torch.device) -> torch.Generator:
    """Create a seeded torch generator on the requested device."""

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def warmup(
    device: torch.device,
    *,
    shape: tuple[int, ...] = (1,),
    dtype: torch.dtype = torch.float64,
) -> None:
    """Initialize CUDA allocation, random generation, and elementary kernels."""

    if device.type == "cuda":
        sample = torch.randn(shape, device=device, dtype=dtype)
        sample.exp_().sum()
        torch.cuda.synchronize(device)
        del sample
