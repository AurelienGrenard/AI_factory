"""Timing helpers for reproducible registry validation benchmarks."""

from __future__ import annotations

from collections.abc import Callable
from statistics import median
from typing import Any, TypeVar


Outputs = TypeVar("Outputs")
PricingCall = Callable[[], tuple[Outputs, dict[str, float]]]


def benchmark_pricing_call(
    call: PricingCall[Outputs],
    *,
    repetitions: int,
    warmup_runs: int,
) -> tuple[Outputs, dict[str, Any]]:
    """Run warmups outside the sample and report a hot-run median separately."""

    if repetitions < 1:
        raise ValueError("A pricing benchmark requires at least one repetition.")
    if warmup_runs < 0:
        raise ValueError("warmup_runs cannot be negative.")

    for _ in range(warmup_runs):
        call()

    outputs, first_timing = call()
    samples = [first_timing]
    for _ in range(1, repetitions):
        _, timing = call()
        samples.append(timing)

    result: dict[str, Any] = dict(first_timing)
    benchmark_seconds = median(
        timing["wall_seconds"] for timing in samples
    )
    result["benchmark_repetitions"] = repetitions
    result["benchmark_statistic"] = "median"
    result["warmup_calls"] = warmup_runs
    kernel_samples = [
        timing["kernel_seconds"]
        for timing in samples
        if "kernel_seconds" in timing
    ]
    if len(kernel_samples) == repetitions:
        benchmark_kernel_seconds = median(kernel_samples)
        result["benchmark_kernel_seconds"] = benchmark_kernel_seconds
        result["benchmark_seconds"] = max(
            benchmark_seconds,
            benchmark_kernel_seconds,
        )
        result["wall_seconds"] = max(
            float(result["wall_seconds"]),
            float(result.get("kernel_seconds", 0.0)),
        )
    else:
        result["benchmark_seconds"] = benchmark_seconds
    return outputs, result
