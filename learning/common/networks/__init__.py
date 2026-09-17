"""Registered, dataset-independent PyTorch network builders."""

from .registry import build_network, register_network
from . import mlp as _mlp  # noqa: F401 -- register built-in network

__all__ = ["build_network", "register_network"]
