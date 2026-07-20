"""Rough Bergomi hybrid-scheme memory autocall pricing in PyTorch."""

from ai_factory.pytorch.common.autocalls import price_batch as _price_batch
from ai_factory.pytorch.rough_bergomi.pathwise import simulate_observation_spots_batch

CPU_BATCH_ROWS = 2
GPU_BATCH_ROWS = 4


def price_batch(*args, batch_rows: int | None = None, **kwargs):
    if batch_rows is None:
        use_gpu = str(kwargs.get("device", "cpu")).startswith("cuda")
        batch_rows = GPU_BATCH_ROWS if use_gpu else CPU_BATCH_ROWS
    return _price_batch(
        *args,
        **kwargs,
        batch_rows=batch_rows,
        simulate_observations=simulate_observation_spots_batch,
    )
