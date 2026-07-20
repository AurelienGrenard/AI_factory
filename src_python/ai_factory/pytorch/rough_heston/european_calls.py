"""Batched rough Heston European calls in PyTorch."""

from ai_factory.pytorch.common.terminal_calls import price_call_batch
from ai_factory.pytorch.rough_heston.pathwise import simulate_terminal_spot_batch


def price_batch(*args, batch_rows: int = 8, **kwargs):
    return price_call_batch(
        *args, simulate_batch=simulate_terminal_spot_batch, batch_rows=batch_rows, **kwargs
    )
