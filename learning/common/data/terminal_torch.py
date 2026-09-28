"""Vectorized PyTorch batches over prepared terminal-sample memory maps."""

from __future__ import annotations

import numpy as np
import torch

from .terminal_cache import PreparedTerminalDataset


class TerminalBatchLoader:
    def __init__(
        self,
        prepared: PreparedTerminalDataset,
        indices: np.ndarray,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
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
            order = torch.randperm(
                self.indices.size, generator=generator
            ).numpy()
            indices = self.indices[order]
        else:
            indices = self.indices
        for start in range(0, indices.size, self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            yield {
                "index": self._tensor(batch_indices),
                "conditions": self._tensor(
                    self.prepared.conditions[batch_indices]
                ),
                "observations": self._tensor(
                    self.prepared.observations[batch_indices]
                ),
                "maturity_days": self._tensor(
                    self.prepared.maturity_days[batch_indices]
                ),
                "parameter_groups": self._tensor(
                    self.prepared.parameter_groups[batch_indices]
                ),
            }
