"""Load a small, explicit YAML/JSON experiment configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "method": "supervised",
    "extensions": [],
    "data": {
        "train_size": None,
        "split_seed": 20260915,
        "selection_seed": 1729,
        "split_role": "model",
        "validation_groups": 100,
        "test_groups": 100,
        "core_groups": 900,
        "validation_fraction": 0.1,
        "test_fraction": 0.1,
        "cache_root": None,
    },
    "network": {
        "name": "mlp",
        "hidden_sizes": [256, 256, 256],
        "activation": "silu",
    },
    "representation": {"name": "identity"},
    "training": {
        "seed": 1234,
        "epochs": 100,
        "batch_size": 4096,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "weight_decay": 0.0,
        "scheduler": "constant",
        "validation_interval": 10,
        "deterministic": True,
        "num_workers": 0,
        "device": "auto",
    },
}


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for name, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(name), dict):
            result[name] = _merge(result[name], value)
        else:
            result[name] = value
    return result


def default_loss(method: str) -> list[dict[str, Any]]:
    if method == "supervised":
        return [{"name": "value_mse", "weight": 1.0}]
    if method == "sobolev":
        return [
            {"name": "value_mse", "weight": 1.0},
            {"name": "gradient_mse", "weight": 0.1},
        ]
    if method == "custom":
        raise ValueError("method=custom requires loss.terms")
    raise ValueError("method must be supervised, sobolev or custom")


def load_config(path: str | Path | None, overrides: dict[str, Any]) -> dict[str, Any]:
    configured: dict[str, Any] = {}
    if path is not None:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Experiment configuration must be a mapping")
        configured = value
    config = _merge(DEFAULT_CONFIG, configured)
    config = _merge(config, overrides)
    method = str(config.get("method", "supervised"))
    if not isinstance(config.get("loss"), dict) or "terms" not in config["loss"]:
        config["loss"] = {"terms": default_loss(method)}
    terms = config["loss"]["terms"]
    if not isinstance(terms, list) or not all(isinstance(term, dict) for term in terms):
        raise ValueError("loss.terms must be a list of mappings")
    if not config.get("dataset"):
        raise ValueError("dataset is required")
    if not isinstance(config.get("extensions"), list) or not all(
        isinstance(name, str) for name in config["extensions"]
    ):
        raise ValueError("extensions must be a list of importable Python module names")
    config["dataset"] = str(config["dataset"])
    if config["data"].get("cache_root") is not None:
        config["data"]["cache_root"] = str(config["data"]["cache_root"])
    return config
