"""Black-Scholes memory autocall pricing in PyTorch."""

from ai_factory.pytorch.black_scholes.pathwise import simulate_observation_spots_batch
from ai_factory.pytorch.common.autocalls import price_batch as _price_batch

CPU_BATCH_ROWS = 16
GPU_BATCH_ROWS = 16


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
