"""Batched Heston European calls in PyTorch."""

from ai_factory.pytorch.common.terminal_calls import price_call_batch
from ai_factory.pytorch.heston.pathwise import simulate_statistic_batch


def _simulate(rows, model_by_id, product_by_id, num_paths, num_steps, device, dtype):
    return simulate_statistic_batch(
        rows, model_by_id, product_by_id, num_paths=num_paths, num_steps=num_steps,
        device=device, dtype=dtype, statistic="terminal_spot"
    )


def price_batch(*args, batch_rows: int = 8, **kwargs):
    return price_call_batch(*args, simulate_batch=_simulate, batch_rows=batch_rows, **kwargs)
