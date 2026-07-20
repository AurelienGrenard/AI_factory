"""Numerical time-grid conventions shared by pricing and path tools."""

from __future__ import annotations

import math


def parse_target_dt(value: str | float | int) -> float:
    text = str(value).strip()
    if "/" in text:
        numerator, denominator = text.split("/", maxsplit=1)
        return float(numerator) / float(denominator)
    return float(text)


def step_count_for_maturity(
    maturity: float, target_dt: str | float | int
) -> int:
    raw_steps = float(maturity) / parse_target_dt(target_dt)
    return max(1, int(math.floor(raw_steps + 0.5)))
