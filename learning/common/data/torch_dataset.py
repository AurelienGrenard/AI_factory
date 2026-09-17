"""PyTorch view over prepared NumPy memory maps."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import PreparedPricingDataset


class PricingTorchDataset(Dataset):
    def __init__(self, prepared: PreparedPricingDataset, indices: np.ndarray):
        self.prepared = prepared
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, position: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[position])
        # torch.tensor intentionally copies a single read-only mmap row. This
        # keeps worker lifetime independent of NumPy's file-backed buffer.
        return {
            "index": torch.tensor(index, dtype=torch.int64),
            "features": torch.tensor(self.prepared.features[index], dtype=torch.float32),
            "values": torch.tensor(self.prepared.values[index], dtype=torch.float32),
            "gradients": torch.tensor(self.prepared.gradients[index], dtype=torch.float32),
            "value_standard_errors": torch.tensor(
                self.prepared.value_standard_errors[index], dtype=torch.float32
            ),
            "gradient_standard_errors": torch.tensor(
                self.prepared.gradient_standard_errors[index], dtype=torch.float32
            ),
        }


class PricingBatchLoader:
    """Vectorized batches over mmap arrays, avoiding one Python call per row."""

    def __init__(
        self,
        prepared: PreparedPricingDataset,
        indices: np.ndarray,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ):
        self.prepared = prepared
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

    def __len__(self) -> int:
        return (self.indices.size + self.batch_size - 1) // self.batch_size

    @staticmethod
    def _tensor(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.array(array, copy=True))

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            order = torch.randperm(self.indices.size, generator=generator).numpy()
            indices = self.indices[order]
        else:
            indices = self.indices
        for start in range(0, indices.size, self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            yield {
                "index": self._tensor(batch_indices),
                "features": self._tensor(self.prepared.features[batch_indices]),
                "values": self._tensor(self.prepared.values[batch_indices]),
                "gradients": self._tensor(self.prepared.gradients[batch_indices]),
                "value_standard_errors": self._tensor(
                    self.prepared.value_standard_errors[batch_indices]
                ),
                "gradient_standard_errors": self._tensor(
                    self.prepared.gradient_standard_errors[batch_indices]
                ),
                "entity_ordinals": self._tensor(
                    self.prepared.entity_ordinals[batch_indices]
                ),
            }
