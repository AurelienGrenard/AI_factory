"""Pure NumPy Philox helpers for reproducible registry parameter sampling."""

from __future__ import annotations

from typing import Final

import numpy as np

PHILOX_M0: Final = np.uint64(0xD2511F53)
PHILOX_M1: Final = np.uint64(0xCD9E8D57)
PHILOX_W0: Final = np.uint32(0x9E3779B9)
PHILOX_W1: Final = np.uint32(0xBB67AE85)
UINT32_SCALE: Final = 1.0 / 4294967296.0


def _seed_to_key(seed: int) -> tuple[np.uint32, np.uint32]:
    normalized = seed % (1 << 64)
    return np.uint32(normalized & 0xFFFFFFFF), np.uint32(normalized >> 32)


def _philox4x32_10(counter: np.ndarray, key: tuple[np.uint32, np.uint32]) -> np.ndarray:
    ctr = counter.astype(np.uint32, copy=True)
    key0 = np.uint32(key[0])
    key1 = np.uint32(key[1])

    for round_index in range(10):
        product0 = ctr[:, 0].astype(np.uint64) * PHILOX_M0
        product1 = ctr[:, 2].astype(np.uint64) * PHILOX_M1
        hi0 = (product0 >> np.uint64(32)).astype(np.uint32)
        hi1 = (product1 >> np.uint64(32)).astype(np.uint32)
        lo0 = product0.astype(np.uint32)
        lo1 = product1.astype(np.uint32)

        ctr = np.column_stack(
            (
                hi1 ^ ctr[:, 1] ^ key0,
                lo1,
                hi0 ^ ctr[:, 3] ^ key1,
                lo0,
            ),
        ).astype(np.uint32, copy=False)

        if round_index != 9:
            key0 = np.uint32((int(key0) + int(PHILOX_W0)) & 0xFFFFFFFF)
            key1 = np.uint32((int(key1) + int(PHILOX_W1)) & 0xFFFFFFFF)

    return ctr


def philox_uniform_uint32(seed: int, count: int, *, stream: int = 0) -> np.ndarray:
    """Generate ``count`` uint32 values from Philox-4x32-10."""

    if count < 0:
        raise ValueError("count must be non-negative.")
    if count == 0:
        return np.empty((0,), dtype=np.uint32)

    block_count = (count + 3) // 4
    block_index = np.arange(block_count, dtype=np.uint64)
    counter = np.empty((block_count, 4), dtype=np.uint32)
    counter[:, 0] = (block_index & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    counter[:, 1] = (block_index >> np.uint64(32)).astype(np.uint32)
    counter[:, 2] = np.uint32(stream & 0xFFFFFFFF)
    counter[:, 3] = np.uint32((stream >> 32) & 0xFFFFFFFF)

    return _philox4x32_10(counter, _seed_to_key(seed)).reshape(-1)[:count]


def philox_uniforms(seed: int, count: int, *, stream: int = 0) -> np.ndarray:
    """Generate open-interval uniforms in [0, 1) with 32-bit resolution."""

    raw = philox_uniform_uint32(seed, count, stream=stream).astype(np.float64)
    return (raw + 0.5) * UINT32_SCALE
