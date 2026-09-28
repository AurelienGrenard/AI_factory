"""Train-only transforms for conditional terminal distributions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch


def _moments(
    array: np.ndarray,
    indices: np.ndarray,
    transform,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(array.shape[1], dtype=np.float64)
    squared = np.zeros(array.shape[1], dtype=np.float64)
    count = 0
    for start in range(0, indices.size, chunk_size):
        values = np.asarray(
            array[indices[start : start + chunk_size]], dtype=np.float64
        )
        values = transform(values)
        if not np.isfinite(values).all():
            raise ValueError(
                "Non-finite value produced while fitting terminal transforms"
            )
        total += values.sum(axis=0)
        squared += np.square(values).sum(axis=0)
        count += values.shape[0]
    if count == 0:
        raise ValueError(
            "Cannot fit terminal transforms on an empty training set"
        )
    mean = total / count
    variance = np.maximum(squared / count - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    scale[scale < 1e-12] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def _validated_specs(
    observable_names: tuple[str, ...], configured: dict[str, Any] | None
) -> tuple[dict[str, Any], ...]:
    configured = configured or {}
    unknown = sorted(set(configured) - set(observable_names))
    if unknown:
        raise ValueError(
            f"Transforms configured for unknown observables: {unknown}"
        )
    specs: list[dict[str, Any]] = []
    for observable in observable_names:
        raw = configured.get(observable, {"name": "identity"})
        if isinstance(raw, str):
            raw = {"name": raw}
        if not isinstance(raw, dict):
            raise ValueError(
                f"Transform for {observable!r} must be a mapping or name"
            )
        name = str(raw.get("name", "identity"))
        if name not in {"identity", "log_shift"}:
            raise ValueError(
                f"Unknown transform {name!r} for {observable!r}; "
                "choose identity or log_shift"
            )
        spec: dict[str, Any] = {"name": name}
        if name == "log_shift":
            offset = float(raw.get("offset", 1e-8))
            minimum = float(raw.get("minimum", 0.0))
            if not math.isfinite(offset) or offset <= 0.0:
                raise ValueError(
                    "log_shift offset must be finite and positive"
                )
            if not math.isfinite(minimum):
                raise ValueError("log_shift minimum must be finite")
            spec.update({"offset": offset, "minimum": minimum})
        specs.append(spec)
    return tuple(specs)


def _forward_numpy(
    values: np.ndarray, specs: tuple[dict[str, Any], ...]
) -> np.ndarray:
    transformed = np.array(values, dtype=np.float64, copy=True)
    for index, spec in enumerate(specs):
        if spec["name"] == "log_shift":
            shifted = transformed[:, index] + float(spec["offset"])
            if np.any(shifted <= 0.0):
                raise ValueError(
                    "log_shift received an observation at or below -offset"
                )
            transformed[:, index] = np.log(shifted)
    return transformed


@dataclass(frozen=True)
class TerminalTransform:
    observable_names: tuple[str, ...]
    observation_specs: tuple[dict[str, Any], ...]
    condition_mean: np.ndarray
    condition_scale: np.ndarray
    observation_mean: np.ndarray
    observation_scale: np.ndarray

    def to_dict(self) -> dict[str, object]:
        return {
            "observable_names": list(self.observable_names),
            "observation_specs": list(self.observation_specs),
            "condition_mean": self.condition_mean.tolist(),
            "condition_scale": self.condition_scale.tolist(),
            "observation_mean": self.observation_mean.tolist(),
            "observation_scale": self.observation_scale.tolist(),
        }

    @classmethod
    def from_dict(
        cls, value: dict[str, object]
    ) -> "TerminalTransform":
        return cls(
            observable_names=tuple(
                str(name) for name in value["observable_names"]
            ),
            observation_specs=tuple(
                dict(spec) for spec in value["observation_specs"]
            ),
            condition_mean=np.asarray(
                value["condition_mean"], dtype=np.float32
            ),
            condition_scale=np.asarray(
                value["condition_scale"], dtype=np.float32
            ),
            observation_mean=np.asarray(
                value["observation_mean"], dtype=np.float32
            ),
            observation_scale=np.asarray(
                value["observation_scale"], dtype=np.float32
            ),
        )

    def normalize_conditions(self, values: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(
            self.condition_mean, device=values.device
        )
        scale = torch.as_tensor(
            self.condition_scale, device=values.device
        )
        return (values - mean) / scale

    def normalize_observations(
        self, values: torch.Tensor
    ) -> torch.Tensor:
        columns = []
        for index, spec in enumerate(self.observation_specs):
            column = values[:, index]
            if spec["name"] == "log_shift":
                column = torch.log(column + float(spec["offset"]))
            columns.append(column)
        transformed = torch.stack(columns, dim=1)
        mean = torch.as_tensor(
            self.observation_mean, device=values.device
        )
        scale = torch.as_tensor(
            self.observation_scale, device=values.device
        )
        return (transformed - mean) / scale

    def denormalize_observations(
        self, values: torch.Tensor
    ) -> torch.Tensor:
        mean = torch.as_tensor(
            self.observation_mean, device=values.device
        )
        scale = torch.as_tensor(
            self.observation_scale, device=values.device
        )
        transformed = values * scale + mean
        columns = []
        for index, spec in enumerate(self.observation_specs):
            column = transformed[:, index]
            if spec["name"] == "log_shift":
                column = torch.exp(column) - float(spec["offset"])
                column = torch.clamp(
                    column, min=float(spec["minimum"])
                )
            columns.append(column)
        return torch.stack(columns, dim=1)


def fit_terminal_transform(
    conditions: np.ndarray,
    observations: np.ndarray,
    observable_names: tuple[str, ...],
    train_indices: np.ndarray,
    configured: dict[str, Any] | None = None,
    *,
    chunk_size: int = 65_536,
) -> TerminalTransform:
    specs = _validated_specs(observable_names, configured)
    condition_mean, condition_scale = _moments(
        conditions,
        train_indices,
        lambda values: values,
        chunk_size,
    )
    observation_mean, observation_scale = _moments(
        observations,
        train_indices,
        lambda values: _forward_numpy(values, specs),
        chunk_size,
    )
    return TerminalTransform(
        observable_names=observable_names,
        observation_specs=specs,
        condition_mean=condition_mean,
        condition_scale=condition_scale,
        observation_mean=observation_mean,
        observation_scale=observation_scale,
    )
