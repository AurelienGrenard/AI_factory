"""Batched rough Bergomi cash-or-nothing digital calls in PyTorch."""

from ai_factory.pytorch.common.terminal_calls import price_digital_batch
from ai_factory.pytorch.rough_bergomi.pathwise import simulate_terminal_spot_batch


def price_batch(*args, batch_rows: int = 8, **kwargs):
    return price_digital_batch(
        *args, simulate_batch=simulate_terminal_spot_batch, batch_rows=batch_rows, **kwargs
    )
