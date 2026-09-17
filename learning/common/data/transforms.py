"""Train-only standardization with derivative-aware scale conversion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Standardization:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    value_mean: np.ndarray
    value_scale: np.ndarray

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "value_mean": self.value_mean.tolist(),
            "value_scale": self.value_scale.tolist(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, list[float]]) -> "Standardization":
        return cls(**{name: np.asarray(numbers, dtype=np.float32) for name, numbers in value.items()})

    def normalized_gradient_scales(self, input_indices: np.ndarray) -> np.ndarray:
        """Return dx_raw/dx_norm divided by dy_raw/dy_norm."""

        if self.value_scale.size != 1:
            raise ValueError("Gradient scaling currently expects one scalar value target")
        return self.feature_scale[input_indices] / self.value_scale[0]


def _moments(array: np.ndarray, indices: np.ndarray, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(array.shape[1], dtype=np.float64)
    squared = np.zeros(array.shape[1], dtype=np.float64)
    count = 0
    for start in range(0, indices.size, chunk_size):
        values = np.asarray(array[indices[start : start + chunk_size]], dtype=np.float64)
        total += values.sum(axis=0)
        squared += np.square(values).sum(axis=0)
        count += values.shape[0]
    if count == 0:
        raise ValueError("Cannot fit transforms on an empty training set")
    mean = total / count
    variance = np.maximum(squared / count - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    scale[scale < 1e-12] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def fit_standardization(
    features: np.ndarray,
    values: np.ndarray,
    train_indices: np.ndarray,
    *,
    chunk_size: int = 65_536,
) -> Standardization:
    feature_mean, feature_scale = _moments(features, train_indices, chunk_size)
    value_mean, value_scale = _moments(values, train_indices, chunk_size)
    return Standardization(feature_mean, feature_scale, value_mean, value_scale)
