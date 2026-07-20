import math

from tools.registry.common.grids import add_numeric_ids, exponential_strike_grid


def test_exponential_strike_grid_widens_with_maturity() -> None:
    products = exponential_strike_grid(
        spot=100.0,
        maturities=[0.25, 4.0],
        strikes_per_maturity=3,
        log_moneyness_width=0.2,
    )

    short_low, short_atm, short_high = products[:3]
    long_low, long_atm, long_high = products[3:]

    assert short_atm["strike"] == 100.0
    assert long_atm["strike"] == 100.0
    assert math.isclose(short_low["strike"], 100.0 * math.exp(-0.1))
    assert math.isclose(long_low["strike"], 100.0 * math.exp(-0.4))
    assert short_high["strike"] - short_low["strike"] < (
        long_high["strike"] - long_low["strike"]
    )


def test_add_numeric_ids() -> None:
    rows = add_numeric_ids(
        [
            {"strike": 90.0, "maturity": 1.0},
            {"strike": 100.0, "maturity": 1.0},
        ]
    )

    assert rows == [
        {"id": "000001", "parameters": {"strike": 90.0, "maturity": 1.0}},
        {"id": "000002", "parameters": {"strike": 100.0, "maturity": 1.0}},
    ]
