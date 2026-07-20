"""Batched Black-Scholes European calls in PyTorch."""

from ai_factory.pytorch.black_scholes.pathwise import simulate_terminal_spot_batch
from ai_factory.pytorch.common.terminal_calls import price_call_batch


def price_batch(*args, batch_rows: int = 128, **kwargs):
    return price_call_batch(
        *args, simulate_batch=simulate_terminal_spot_batch, batch_rows=batch_rows, **kwargs
    )
