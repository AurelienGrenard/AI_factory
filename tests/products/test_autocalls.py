import math

import torch

from ai_factory.pytorch.common.autocalls import price_from_observations


def test_memory_autocall_cashflows_and_diagnostics() -> None:
    rows = [{"id": "000001", "model_id": "000001", "product_id": "000001"}]
    models = {
        "000001": {"spot": 1.0, "risk_free_rate": 0.0},
    }
    products = {
        "000001": {
            "maturity": 1.0,
            "autocall_barrier": 1.0,
            "coupon_barrier": 0.8,
            "protection_barrier": 0.6,
            "coupon_rate_per_observation": 0.02,
            "observation_count": 4,
            "first_autocall_observation": 2,
        },
    }
    observations = torch.tensor(
        [[[0.7, 0.9, 1.1, 1.1], [0.7, 0.7, 0.7, 0.5]]],
        dtype=torch.float64,
    )

    outputs, diagnostics = price_from_observations(
        observations, rows, models, products
    )

    assert math.isclose(outputs[0]["price"], 0.78)
    assert math.isclose(outputs[0]["standard_error"], 0.28)
    assert diagnostics[0] == {
        "autocall_probability": 0.5,
        "mean_autocall_time": 0.75,
        "maturity_probability": 0.5,
        "coupon_payment_frequency": 0.25,
        "mean_total_coupon": 0.03,
        "capital_loss_probability": 0.5,
        "mean_redemption_given_loss": 0.5,
    }
