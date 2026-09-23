"""Stable held-out splits and nested random training subsets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


TRAIN = np.uint8(0)
VALIDATION = np.uint8(1)
TEST = np.uint8(2)
_UINT64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


def _splitmix64(values: np.ndarray, seed: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint64) ^ np.uint64(seed & int(_UINT64_MASK))
    values = values + np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def _digest(indices: np.ndarray) -> str:
    canonical = np.asarray(indices, dtype="<u8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


@dataclass(frozen=True)
class DatasetSelection:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    train_candidate_count: int
    train_size_requested: int | None
    split_seed: int
    selection_seed: int
    split_strategy: str = "row"
    group_manifest: dict[str, object] | None = None

    def manifest(self) -> dict[str, object]:
        return {
            "train_size_requested": self.train_size_requested,
            "train_candidate_count": self.train_candidate_count,
            "counts": {
                "train": int(self.train_indices.size),
                "validation": int(self.validation_indices.size),
                "test": int(self.test_indices.size),
            },
            "split_seed": self.split_seed,
            "selection_seed": self.selection_seed,
            "split_strategy": self.split_strategy,
            "groups": self.group_manifest,
            "digests": {
                "train": _digest(self.train_indices),
                "validation": _digest(self.validation_indices),
                "test": _digest(self.test_indices),
            },
        }


def make_dataset_selection(
    row_count: int,
    *,
    train_size: int | None = None,
    split_seed: int = 20260915,
    selection_seed: int = 1729,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    group_ids: np.ndarray | None = None,
    validation_group_count: int | None = None,
    test_group_count: int | None = None,
    core_group_count: int | None = None,
) -> DatasetSelection:
    """Keep validation/test fixed while choosing a nested random train prefix.

    SplitMix64 supplies a stable random priority for every physical row. Taking
    the N smallest training priorities makes a 30k selection a strict subset of
    the corresponding 100k and 500k selections when all seeds are unchanged.
    """

    if row_count <= 0:
        raise ValueError("row_count must be positive")
    indices = np.arange(row_count, dtype=np.uint64)
    codes = np.full(row_count, TRAIN, dtype=np.uint8)
    group_manifest: dict[str, object] | None = None
    if group_ids is None:
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must lie in (0, 1)")
        if not 0.0 < test_fraction < 1.0:
            raise ValueError("test_fraction must lie in (0, 1)")
        if validation_fraction + test_fraction >= 1.0:
            raise ValueError("validation and test fractions must sum to less than one")
        split_priority = _splitmix64(indices, split_seed)
        bucket_count = 1_000_000
        buckets = split_priority % np.uint64(bucket_count)
        test_limit = int(round(test_fraction * bucket_count))
        validation_limit = test_limit + int(round(validation_fraction * bucket_count))
        codes[buckets < test_limit] = TEST
        codes[(buckets >= test_limit) & (buckets < validation_limit)] = VALIDATION
        split_strategy = "row"
    else:
        groups = np.asarray(group_ids, dtype=np.int64)
        if groups.shape != (row_count,) or np.any(groups < 0):
            raise ValueError("group_ids must contain one nonnegative id per row")
        unique_groups = np.unique(groups)
        validation_groups = (
            int(round(validation_fraction * unique_groups.size))
            if validation_group_count is None
            else validation_group_count
        )
        test_groups = (
            int(round(test_fraction * unique_groups.size))
            if test_group_count is None
            else test_group_count
        )
        if validation_groups <= 0 or test_groups <= 0:
            raise ValueError("Grouped validation and test counts must be positive")
        if validation_groups + test_groups >= unique_groups.size:
            raise ValueError("Grouped holdouts leave no training groups")

        def allocate(
            candidates: np.ndarray, test_count: int, validation_count: int
        ) -> tuple[np.ndarray, np.ndarray]:
            if test_count + validation_count > candidates.size:
                raise ValueError("Requested grouped holdouts exceed available groups")
            priorities = _splitmix64(candidates.astype(np.uint64), split_seed)
            ordered = candidates[np.argsort(priorities, kind="stable")]
            return (
                ordered[:test_count],
                ordered[test_count : test_count + validation_count],
            )

        if core_group_count is None:
            selected_test, selected_validation = allocate(
                unique_groups, test_groups, validation_groups
            )
        else:
            if core_group_count <= 0:
                raise ValueError("core_group_count must be positive")
            core = unique_groups[unique_groups < core_group_count]
            stress = unique_groups[unique_groups >= core_group_count]
            core_share = core.size / unique_groups.size
            test_core_count = int(round(test_groups * core_share))
            validation_core_count = int(round(validation_groups * core_share))
            test_core, validation_core = allocate(
                core, test_core_count, validation_core_count
            )
            test_stress, validation_stress = allocate(
                stress,
                test_groups - test_core_count,
                validation_groups - validation_core_count,
            )
            selected_test = np.concatenate((test_core, test_stress))
            selected_validation = np.concatenate((validation_core, validation_stress))
        codes[np.isin(groups, selected_test)] = TEST
        codes[np.isin(groups, selected_validation)] = VALIDATION
        held_out = np.concatenate((selected_test, selected_validation))
        training_groups = unique_groups[~np.isin(unique_groups, held_out)]
        group_manifest = {
            "total": int(unique_groups.size),
            "train": int(training_groups.size),
            "validation": int(selected_validation.size),
            "test": int(selected_test.size),
            "core_group_count": core_group_count,
            "digests": {
                "train": _digest(np.sort(training_groups)),
                "validation": _digest(np.sort(selected_validation)),
                "test": _digest(np.sort(selected_test)),
            },
        }
        split_strategy = "group"

    candidates = indices[codes == TRAIN]
    requested = train_size
    if train_size is None:
        selected = candidates
    else:
        if train_size <= 0:
            raise ValueError("train_size must be positive or omitted for all")
        if train_size > candidates.size:
            raise ValueError(
                f"train_size={train_size} exceeds the {candidates.size} training "
                "rows left after the fixed validation/test split"
            )
        priorities = _splitmix64(candidates, selection_seed)
        positions = np.argpartition(priorities, train_size - 1)[:train_size]
        selected = candidates[positions]

    return DatasetSelection(
        train_indices=np.sort(selected).astype(np.int64),
        validation_indices=np.flatnonzero(codes == VALIDATION).astype(np.int64),
        test_indices=np.flatnonzero(codes == TEST).astype(np.int64),
        train_candidate_count=int(candidates.size),
        train_size_requested=requested,
        split_seed=split_seed,
        selection_seed=selection_seed,
        split_strategy=split_strategy,
        group_manifest=group_manifest,
    )
