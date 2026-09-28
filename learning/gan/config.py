"""Validated configuration for conditional terminal GAN training."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "extensions": [],
    "data": {
        "observables": None,
        "observable_transforms": {},
        "train_size": None,
        "split_seed": 0,
        "selection_seed": 1729,
        "cache_root": None,
    },
    "generator": {
        "name": "conditional_residual_mlp",
        "latent_dimension": 16,
        "noise_distribution": "normal",
        "hidden_size": 256,
        "depth": 4,
        "activation": "silu",
        "layer_norm": True,
    },
    "critic": {
        "name": "conditional_projection_mlp",
        "hidden_size": 256,
        "depth": 4,
        "activation": "leaky_relu",
        "spectral_norm": False,
    },
    "objective": {
        "name": "wasserstein_gp",
        "gradient_penalty_weight": 10.0,
        "gradient_penalty_target": 1.0,
        "drift_weight": 0.001,
        "mode_seeking_weight": 0.0,
        "mode_seeking_epsilon": 1.0e-6,
        "moment_matching_weight": 0.0,
    },
    "training": {
        "seed": 1234,
        "epochs": 100,
        "batch_size": 2048,
        "critic_steps": 5,
        "generator_learning_rate": 0.0001,
        "critic_learning_rate": 0.0002,
        "adam_betas": [0.0, 0.9],
        "weight_decay": 0.0,
        "scheduler": "constant",
        "ema_decay": 0.999,
        "validation_interval": 5,
        "log_interval": 50,
        "deterministic": True,
        "num_workers": 0,
        "device": "auto",
    },
    "evaluation": {
        "max_samples": 8192,
        "mmd_samples": 2048,
        "projection_count": 64,
        "condition_projection_count": 16,
        "condition_bin_count": 8,
        "std_ratio_penalty_weight": 0.0,
        "seed": 941,
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


def _positive(value: object, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _positive_int(value: object, name: str) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return number


def load_config(
    path: str | Path | None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    configured: dict[str, Any] = {}
    if path is not None:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("GAN configuration must be a mapping")
        configured = value
    config = _merge(DEFAULT_CONFIG, configured)
    config = _merge(config, overrides or {})
    if not config.get("dataset"):
        raise ValueError("dataset is required")
    config["dataset"] = str(config["dataset"])

    extensions = config.get("extensions")
    if not isinstance(extensions, list) or not all(
        isinstance(name, str) for name in extensions
    ):
        raise ValueError("extensions must be a list of importable module names")

    data = config["data"]
    observables = data.get("observables")
    if observables is not None:
        if (
            not isinstance(observables, list)
            or not observables
            or not all(isinstance(name, str) and name for name in observables)
            or len(set(observables)) != len(observables)
        ):
            raise ValueError(
                "data.observables must be null or a non-empty unique name list"
            )
    if not isinstance(data.get("observable_transforms"), dict):
        raise ValueError("data.observable_transforms must be a mapping")
    train_size = data.get("train_size")
    if train_size is not None:
        data["train_size"] = _positive_int(train_size, "data.train_size")
    data["split_seed"] = int(data["split_seed"])
    data["selection_seed"] = int(data["selection_seed"])
    if data.get("cache_root") is not None:
        data["cache_root"] = str(data["cache_root"])

    generator = config["generator"]
    generator["latent_dimension"] = _positive_int(
        generator["latent_dimension"], "generator.latent_dimension"
    )
    generator["hidden_size"] = _positive_int(
        generator["hidden_size"], "generator.hidden_size"
    )
    generator["depth"] = _positive_int(
        generator["depth"], "generator.depth"
    )
    if generator.get("noise_distribution") not in {
        "normal",
        "uniform",
    }:
        raise ValueError(
            "generator.noise_distribution must be normal or uniform"
        )
    noise_minimum = float(generator.get("noise_minimum", -1.0))
    noise_maximum = float(generator.get("noise_maximum", 1.0))
    if "noise_minimum" in generator:
        generator["noise_minimum"] = noise_minimum
    if "noise_maximum" in generator:
        generator["noise_maximum"] = noise_maximum
    if generator["noise_distribution"] == "uniform" and (
        not math.isfinite(noise_minimum)
        or not math.isfinite(noise_maximum)
        or noise_minimum >= noise_maximum
    ):
        raise ValueError(
            "generator uniform noise bounds must be finite and increasing"
        )
    for name in ("parameter_embedding_size", "noise_embedding_size"):
        if name in generator:
            generator[name] = _positive_int(
                generator[name], f"generator.{name}"
            )
    critic = config["critic"]
    critic["hidden_size"] = _positive_int(
        critic["hidden_size"], "critic.hidden_size"
    )
    critic["depth"] = _positive_int(critic["depth"], "critic.depth")
    for name in (
        "parameter_embedding_size",
        "observation_embedding_size",
    ):
        if name in critic:
            critic[name] = _positive_int(
                critic[name], f"critic.{name}"
            )

    objective = config["objective"]
    if objective.get("name") not in {"wasserstein_gp", "logistic"}:
        raise ValueError(
            "objective.name must be wasserstein_gp or logistic"
        )
    objective["gradient_penalty_weight"] = _positive(
        objective["gradient_penalty_weight"],
        "objective.gradient_penalty_weight",
    )
    objective["gradient_penalty_target"] = _positive(
        objective["gradient_penalty_target"],
        "objective.gradient_penalty_target",
    )
    objective["drift_weight"] = float(objective["drift_weight"])
    if (
        not math.isfinite(objective["drift_weight"])
        or objective["drift_weight"] < 0.0
    ):
        raise ValueError(
            "objective.drift_weight must be finite and nonnegative"
        )
    for name in (
        "mode_seeking_weight",
        "moment_matching_weight",
    ):
        objective[name] = float(objective.get(name, 0.0))
        if (
            not math.isfinite(objective[name])
            or objective[name] < 0.0
        ):
            raise ValueError(
                f"objective.{name} must be finite and nonnegative"
            )
    objective["mode_seeking_epsilon"] = _positive(
        objective.get("mode_seeking_epsilon", 1.0e-6),
        "objective.mode_seeking_epsilon",
    )

    training = config["training"]
    for name in (
        "epochs",
        "batch_size",
        "critic_steps",
        "validation_interval",
        "log_interval",
    ):
        training[name] = _positive_int(
            training[name], f"training.{name}"
        )
    for name in ("generator_learning_rate", "critic_learning_rate"):
        training[name] = _positive(training[name], f"training.{name}")
    betas = training["adam_betas"]
    if (
        not isinstance(betas, list)
        or len(betas) != 2
        or not all(0.0 <= float(beta) < 1.0 for beta in betas)
    ):
        raise ValueError(
            "training.adam_betas must contain two values in [0, 1)"
        )
    training["adam_betas"] = [float(beta) for beta in betas]
    training["ema_decay"] = float(training["ema_decay"])
    if (
        not math.isfinite(training["ema_decay"])
        or not 0.0 <= training["ema_decay"] < 1.0
    ):
        raise ValueError(
            "training.ema_decay must be finite and lie in [0, 1)"
        )
    training["weight_decay"] = float(
        training.get("weight_decay", 0.0)
    )
    if (
        not math.isfinite(training["weight_decay"])
        or training["weight_decay"] < 0.0
    ):
        raise ValueError(
            "training.weight_decay must be finite and nonnegative"
        )
    if int(training.get("num_workers", 0)) != 0:
        raise ValueError(
            "training.num_workers must be 0 for the vectorized mmap loader"
        )
    if training.get("scheduler") not in {
        "constant",
        "cosine",
        "exponential",
    }:
        raise ValueError(
            "training.scheduler must be constant, cosine or exponential"
        )
    scheduler_gamma = float(training.get("scheduler_gamma", 0.99))
    if "scheduler_gamma" in training or (
        training["scheduler"] == "exponential"
    ):
        training["scheduler_gamma"] = scheduler_gamma
    if (
        not math.isfinite(scheduler_gamma)
        or not 0.0 < scheduler_gamma <= 1.0
    ):
        raise ValueError(
            "training.scheduler_gamma must lie in (0, 1]"
        )

    evaluation = config["evaluation"]
    for name in (
        "max_samples",
        "mmd_samples",
        "projection_count",
        "condition_projection_count",
        "condition_bin_count",
    ):
        evaluation[name] = _positive_int(
            evaluation[name], f"evaluation.{name}"
        )
    if evaluation["max_samples"] < 2 or evaluation["mmd_samples"] < 2:
        raise ValueError(
            "evaluation max_samples and mmd_samples must exceed one"
        )
    evaluation["std_ratio_penalty_weight"] = float(
        evaluation.get("std_ratio_penalty_weight", 0.0)
    )
    if (
        not math.isfinite(evaluation["std_ratio_penalty_weight"])
        or evaluation["std_ratio_penalty_weight"] < 0.0
    ):
        raise ValueError(
            "evaluation.std_ratio_penalty_weight must be finite and nonnegative"
        )
    evaluation["seed"] = int(evaluation["seed"])
    return config
