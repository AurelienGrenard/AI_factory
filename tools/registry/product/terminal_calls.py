"""Shared parameter grid for terminal equity call products."""

from __future__ import annotations

import numpy as np

from tools.registry.common.philox import philox_uniforms

ROW_COUNT = 1_000
SEED = 731_600_001
MIN_MATURITY = 1.0 / 12.0
MAX_MATURITY = 3.0
LOG_MONEYNESS_SLOPE = 0.2


def terminal_call_parameters() -> list[dict[str, float]]:
    maturity_uniforms = philox_uniforms(SEED, ROW_COUNT, stream=0)
    strike_uniforms = philox_uniforms(SEED, ROW_COUNT, stream=1)
    maturities = MIN_MATURITY + (MAX_MATURITY - MIN_MATURITY) * maturity_uniforms
    log_strikes = LOG_MONEYNESS_SLOPE * maturities * (2.0 * strike_uniforms - 1.0)
    strikes = np.exp(log_strikes)
    return [
        {
            "strike": round(float(strikes[index]), 12),
            "maturity": round(float(maturities[index]), 12),
        }
        for index in range(ROW_COUNT)
    ]
