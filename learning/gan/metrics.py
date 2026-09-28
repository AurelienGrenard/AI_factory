"""Distributional metrics for conditional terminal generators."""

from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Any

import numpy as np
import torch

from learning.common.data.terminal_cache import TerminalTensorSchema
from learning.common.data.terminal_transforms import TerminalTransform

from .networks import ConditionalGenerator
from .noise import sample_noise


def _unit_projections(
    dimension: int, count: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    projections = rng.normal(size=(count, dimension))
    norms = np.linalg.norm(projections, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return projections / norms


def _sliced_wasserstein(
    real: np.ndarray,
    generated: np.ndarray,
    *,
    projection_count: int,
    seed: int,
) -> float:
    if real.shape != generated.shape or real.ndim != 2:
        raise ValueError(
            "Sliced Wasserstein inputs must be equally shaped matrices"
        )
    projections = _unit_projections(
        real.shape[1], projection_count, seed
    )
    real_projected = np.sort(real @ projections.T, axis=0)
    fake_projected = np.sort(generated @ projections.T, axis=0)
    return float(np.mean(np.abs(real_projected - fake_projected)))


def _squared_distances(
    left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    left_squared = np.sum(np.square(left), axis=1, keepdims=True)
    right_squared = np.sum(
        np.square(right), axis=1, keepdims=True
    ).T
    return np.maximum(
        left_squared + right_squared - 2.0 * left @ right.T,
        0.0,
    )


def _multiscale_mmd(
    real: np.ndarray, generated: np.ndarray, seed: int
) -> float:
    if real.shape != generated.shape or real.ndim != 2:
        raise ValueError("MMD inputs must be equally shaped matrices")
    if real.shape[0] < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    pooled = np.concatenate((real, generated), axis=0)
    pilot_size = min(512, pooled.shape[0])
    pilot = pooled[
        rng.choice(pooled.shape[0], size=pilot_size, replace=False)
    ]
    pilot_distances = _squared_distances(pilot, pilot)
    positive = pilot_distances[pilot_distances > 0.0]
    bandwidth_squared = (
        float(np.median(positive)) if positive.size else 1.0
    )
    bandwidth_squared = max(bandwidth_squared, 1e-8)
    scales = (0.5, 1.0, 2.0)
    xx = _squared_distances(real, real)
    yy = _squared_distances(generated, generated)
    xy = _squared_distances(real, generated)
    values = []
    for scale in scales:
        denominator = 2.0 * bandwidth_squared * scale * scale
        values.append(
            np.exp(-xx / denominator).mean()
            + np.exp(-yy / denominator).mean()
            - 2.0 * np.exp(-xy / denominator).mean()
        )
    return float(max(np.mean(values), 0.0))


def _conditional_wasserstein(
    conditions: np.ndarray,
    real: np.ndarray,
    generated: np.ndarray,
    *,
    condition_projection_count: int,
    bin_count: int,
    seed: int,
) -> float:
    projections = _unit_projections(
        conditions.shape[1], condition_projection_count, seed
    )
    values: list[float] = []
    for projection_index in range(condition_projection_count):
        ordering = np.argsort(
            conditions @ projections[projection_index],
            kind="stable",
        )
        for positions in np.array_split(ordering, bin_count):
            if positions.size < 2:
                continue
            for observable in range(real.shape[1]):
                real_sorted = np.sort(real[positions, observable])
                fake_sorted = np.sort(generated[positions, observable])
                values.append(
                    float(np.mean(np.abs(real_sorted - fake_sorted)))
                )
    return float(np.mean(values)) if values else 0.0


def _correlation(values: np.ndarray) -> np.ndarray:
    covariance = np.atleast_2d(np.cov(values, rowvar=False))
    standard_deviation = np.sqrt(
        np.maximum(np.diag(covariance), 0.0)
    )
    denominator = np.outer(
        standard_deviation, standard_deviation
    )
    correlation = np.zeros_like(covariance)
    np.divide(
        covariance,
        denominator,
        out=correlation,
        where=denominator > 1e-12,
    )
    return correlation


def evaluate_generator(
    generator: ConditionalGenerator,
    loader: Iterable[dict[str, torch.Tensor]],
    schema: TerminalTensorSchema,
    transform: TerminalTransform,
    device: torch.device,
    specification: dict[str, Any],
) -> dict[str, float | int]:
    """Compare real and generated held-out joint distributions."""

    max_samples = int(specification["max_samples"])
    projection_count = int(specification["projection_count"])
    condition_projection_count = int(
        specification["condition_projection_count"]
    )
    bin_count = int(specification["condition_bin_count"])
    seed = int(specification["seed"])
    random = torch.Generator(device="cpu")
    random.manual_seed(seed)
    was_training = generator.training
    generator.eval()
    condition_parts: list[np.ndarray] = []
    real_normalized_parts: list[np.ndarray] = []
    fake_normalized_parts: list[np.ndarray] = []
    fake_second_parts: list[np.ndarray] = []
    real_raw_parts: list[np.ndarray] = []
    fake_raw_parts: list[np.ndarray] = []
    seen = 0
    with torch.no_grad():
        for batch in loader:
            if seen >= max_samples:
                break
            take = min(
                int(batch["conditions"].shape[0]),
                max_samples - seen,
            )
            raw_conditions = batch["conditions"][:take].to(device)
            raw_observations = batch["observations"][:take].to(device)
            conditions = transform.normalize_conditions(raw_conditions)
            real_normalized = transform.normalize_observations(
                raw_observations
            )
            first_noise = sample_noise(
                take,
                generator.latent_dimension,
                generator.noise_distribution,
                random=random,
                device=device,
                uniform_minimum=generator.noise_minimum,
                uniform_maximum=generator.noise_maximum,
            )
            second_noise = sample_noise(
                take,
                generator.latent_dimension,
                generator.noise_distribution,
                random=random,
                device=device,
                uniform_minimum=generator.noise_minimum,
                uniform_maximum=generator.noise_maximum,
            )
            fake_normalized = generator(conditions, first_noise)
            fake_second = generator(conditions, second_noise)
            fake_raw = transform.denormalize_observations(
                fake_normalized
            )
            tensors = (
                conditions,
                real_normalized,
                fake_normalized,
                fake_second,
                raw_observations,
                fake_raw,
            )
            if not all(bool(torch.isfinite(item).all()) for item in tensors):
                raise FloatingPointError(
                    "Non-finite values encountered during GAN evaluation"
                )
            condition_parts.append(conditions.cpu().numpy())
            real_normalized_parts.append(
                real_normalized.cpu().numpy()
            )
            fake_normalized_parts.append(
                fake_normalized.cpu().numpy()
            )
            fake_second_parts.append(fake_second.cpu().numpy())
            real_raw_parts.append(raw_observations.cpu().numpy())
            fake_raw_parts.append(fake_raw.cpu().numpy())
            seen += take
    if was_training:
        generator.train()
    if seen < 2:
        raise ValueError(
            "GAN evaluation requires at least two held-out samples"
        )

    conditions = np.concatenate(condition_parts).astype(
        np.float64, copy=False
    )
    real_normalized = np.concatenate(
        real_normalized_parts
    ).astype(np.float64, copy=False)
    fake_normalized = np.concatenate(
        fake_normalized_parts
    ).astype(np.float64, copy=False)
    fake_second = np.concatenate(fake_second_parts).astype(
        np.float64, copy=False
    )
    real_raw = np.concatenate(real_raw_parts).astype(
        np.float64, copy=False
    )
    fake_raw = np.concatenate(fake_raw_parts).astype(
        np.float64, copy=False
    )

    metrics: dict[str, float | int] = {"sample_count": seen}
    standard_deviation_ratios: list[float] = []
    quantiles = (
        (0.01, "q01"),
        (0.05, "q05"),
        (0.5, "q50"),
        (0.95, "q95"),
        (0.99, "q99"),
    )
    for index, name in enumerate(schema.observable_names):
        real_column = real_raw[:, index]
        fake_column = fake_raw[:, index]
        prefix = name.replace(".", "_")
        real_mean = float(np.mean(real_column))
        fake_mean = float(np.mean(fake_column))
        real_std = float(np.std(real_column))
        fake_std = float(np.std(fake_column))
        metrics[f"{prefix}_real_mean"] = real_mean
        metrics[f"{prefix}_generated_mean"] = fake_mean
        metrics[f"{prefix}_mean_absolute_error"] = abs(
            fake_mean - real_mean
        )
        metrics[f"{prefix}_real_std"] = real_std
        metrics[f"{prefix}_generated_std"] = fake_std
        metrics[f"{prefix}_std_absolute_error"] = abs(
            fake_std - real_std
        )
        std_ratio = fake_std / max(real_std, 1.0e-8)
        metrics[f"{prefix}_std_ratio"] = std_ratio
        standard_deviation_ratios.append(std_ratio)
        metrics[f"{prefix}_generated_minimum"] = float(
            np.min(fake_column)
        )
        for quantile, label in quantiles:
            real_quantile = float(np.quantile(real_column, quantile))
            fake_quantile = float(np.quantile(fake_column, quantile))
            metrics[f"{prefix}_{label}_absolute_error"] = abs(
                fake_quantile - real_quantile
            )

    metrics["mean_absolute_log_std_ratio"] = float(
        np.mean(
            np.abs(
                np.log(
                    np.clip(
                        standard_deviation_ratios,
                        1.0e-8,
                        None,
                    )
                )
            )
        )
    )

    if real_normalized.shape[1] > 1:
        real_covariance = np.cov(real_normalized, rowvar=False)
        fake_covariance = np.cov(fake_normalized, rowvar=False)
        metrics["covariance_frobenius_error"] = float(
            np.linalg.norm(
                fake_covariance - real_covariance, ord="fro"
            )
        )
        real_correlation = _correlation(real_normalized)
        fake_correlation = _correlation(fake_normalized)
        metrics["correlation_frobenius_error"] = float(
            np.linalg.norm(
                fake_correlation - real_correlation, ord="fro"
            )
        )
    else:
        metrics["covariance_frobenius_error"] = abs(
            float(np.var(fake_normalized))
            - float(np.var(real_normalized))
        )
        metrics["correlation_frobenius_error"] = 0.0

    joint_real = np.concatenate(
        (conditions, real_normalized), axis=1
    )
    joint_fake = np.concatenate(
        (conditions, fake_normalized), axis=1
    )
    mmd_count = min(int(specification["mmd_samples"]), seen)
    metrics["observation_mmd"] = _multiscale_mmd(
        real_normalized[:mmd_count],
        fake_normalized[:mmd_count],
        seed + 1,
    )
    metrics["joint_mmd"] = _multiscale_mmd(
        joint_real[:mmd_count],
        joint_fake[:mmd_count],
        seed + 2,
    )
    metrics["observation_sliced_wasserstein"] = (
        _sliced_wasserstein(
            real_normalized,
            fake_normalized,
            projection_count=projection_count,
            seed=seed + 3,
        )
    )
    metrics["joint_sliced_wasserstein"] = _sliced_wasserstein(
        joint_real,
        joint_fake,
        projection_count=projection_count,
        seed=seed + 4,
    )
    metrics["conditional_sliced_wasserstein"] = (
        _conditional_wasserstein(
            conditions,
            real_normalized,
            fake_normalized,
            condition_projection_count=condition_projection_count,
            bin_count=min(bin_count, seen // 2),
            seed=seed + 5,
        )
    )
    metrics["generated_pair_distance_mean"] = float(
        np.mean(
            np.linalg.norm(
                fake_normalized - fake_second, axis=1
            )
        )
    )
    metrics["selection_score"] = float(
        metrics["joint_mmd"]
        + metrics["observation_sliced_wasserstein"]
        + metrics["conditional_sliced_wasserstein"]
        + float(specification.get("std_ratio_penalty_weight", 0.0))
        * metrics["mean_absolute_log_std_ratio"]
    )
    if not all(
        math.isfinite(float(value)) for value in metrics.values()
    ):
        raise FloatingPointError(
            "Non-finite distributional metric encountered"
        )
    return metrics
