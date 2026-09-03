import math
from pathlib import Path
import unittest

import numpy as np
from scipy.special import ndtr

from validation.volterra.rough_sabr import (
    RoughSabrParameters,
    hybrid_european_option_price,
    rough_sabr_path_payoffs,
)
from validation.volterra.fukasawa_gatheral import (
    analyze_fukasawa_gatheral_campaign,
    analyze_fukasawa_gatheral_probe,
    black_scholes_implied_volatility,
    fukasawa_gatheral_g_approximation,
    fukasawa_gatheral_g_half,
    fukasawa_gatheral_g_zero,
    fukasawa_gatheral_implied_volatility,
)


def black_price(
    volatility: float,
    strike: float,
    maturity: float,
    side: str,
) -> float:
    standard_deviation = volatility * math.sqrt(maturity)
    d1 = -math.log(strike) / standard_deviation + 0.5 * standard_deviation
    call = ndtr(d1) - strike * ndtr(d1 - standard_deviation)
    return float(call if side == "call" else call - 1.0 + strike)


class RoughSabrReferenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = RoughSabrParameters(
            1.0, 0.02, 0.01, 0.04, 1.2, 0.1, -0.7, 0.7
        )

    def test_numpy_fft_and_direct_driver_give_the_same_paths(self) -> None:
        generator = np.random.default_rng(123)
        noise = generator.standard_normal((3, 7, 37))
        direct = rough_sabr_path_payoffs(
            self.parameters, 1.0, 1.0, "call", *noise, "direct"
        )
        transformed = rough_sabr_path_payoffs(
            self.parameters, 1.0, 1.0, "call", *noise, "fft"
        )
        np.testing.assert_allclose(
            direct, transformed, rtol=3.0e-13, atol=3.0e-13
        )

    def test_beta_one_path_is_explicit_rough_bergomi_recursion(self) -> None:
        parameters = RoughSabrParameters(
            1.0, 0.02, 0.01, 0.04, 1.2, 0.1, -0.7, 1.0
        )
        generator = np.random.default_rng(456)
        rough, singular, spot_noise = generator.standard_normal((3, 5, 24))
        actual = rough_sabr_path_payoffs(
            parameters,
            1.0,
            1.0,
            "call",
            rough,
            singular,
            spot_noise,
            "direct",
        )

        # Independent explicit rB log-spot recursion using the same variance
        # convention and the already validated hybrid driver.
        from validation.volterra.rough_bergomi import (
            hybrid_normalized_driver_direct,
        )

        dt = 1.0 / rough.shape[1]
        driver = hybrid_normalized_driver_direct(
            parameters.hurst_exponent,
            1.0,
            math.sqrt(dt) * rough,
            rough,
            singular,
        )
        variance = np.full(rough.shape[0], parameters.xi_0)
        log_spot = np.zeros(rough.shape[0])
        correlated = parameters.rho * rough + math.sqrt(
            1.0 - parameters.rho**2
        ) * spot_noise
        for step in range(rough.shape[1]):
            log_spot += (
                (parameters.risk_free_rate - parameters.dividend_yield) * dt
                - 0.5 * variance * dt
                + np.sqrt(variance * dt) * correlated[:, step]
            )
            time = (step + 1) * dt
            variance = parameters.xi_0 * np.exp(
                parameters.eta * driver[:, step]
                - 0.5
                * parameters.eta**2
                * time ** (2.0 * parameters.hurst_exponent)
            )
        expected = math.exp(-parameters.risk_free_rate) * np.maximum(
            np.exp(log_spot) - 1.0, 0.0
        )
        np.testing.assert_allclose(actual, expected, rtol=2.0e-14, atol=2.0e-14)

    def test_spot_rescaling_preserves_normalized_paths(self) -> None:
        low = RoughSabrParameters(
            0.5, 0.0, 0.0, 0.04, 1.1, 0.08, -0.6, 0.65
        )
        high = RoughSabrParameters(
            3.0, 0.0, 0.0, 0.04, 1.1, 0.08, -0.6, 0.65
        )
        noise = np.random.default_rng(991).standard_normal((3, 8, 48))
        low_payoff = rough_sabr_path_payoffs(
            low, low.spot, 0.5, "call", *noise, "fft"
        )
        high_payoff = rough_sabr_path_payoffs(
            high, high.spot, 0.5, "call", *noise, "fft"
        )
        np.testing.assert_allclose(
            high_payoff / high.spot,
            low_payoff / low.spot,
            rtol=5.0e-13,
            atol=5.0e-13,
        )

    def test_eta_zero_beta_one_reduces_to_black_scholes(self) -> None:
        parameters = RoughSabrParameters(
            1.0, 0.02, 0.01, 0.04, 0.0, 0.1, -0.7, 1.0
        )
        estimate = hybrid_european_option_price(
            parameters, 1.0, 1.0, "call", 32, 32768, 789
        )
        standard_deviation = 0.2
        d1 = (0.02 - 0.01 + 0.5 * 0.04) / standard_deviation
        expected = math.exp(-0.01) * ndtr(d1) - math.exp(-0.02) * ndtr(
            d1 - standard_deviation
        )
        self.assertLess(
            abs(estimate.price - expected),
            4.0 * estimate.standard_error + 2.0e-5,
        )

    def test_fukasawa_gatheral_interpolation_has_exact_endpoints(self) -> None:
        for y in (-0.7, -0.2, 0.2, 0.7):
            self.assertAlmostEqual(
                fukasawa_gatheral_g_approximation(y, 0.0, -0.9),
                fukasawa_gatheral_g_zero(y, -0.9),
                places=13,
            )
            self.assertAlmostEqual(
                fukasawa_gatheral_g_approximation(y, 0.5, -0.9),
                fukasawa_gatheral_g_half(y, -0.9),
                places=13,
            )

    def test_paper_eta_is_used_without_factor_two_conversion(self) -> None:
        volatility = fukasawa_gatheral_implied_volatility(
            1.0, math.exp(0.1), 0.25, 0.04, 1.0, 0.05, -0.9, 0.5
        )
        doubled_eta = fukasawa_gatheral_implied_volatility(
            1.0, math.exp(0.1), 0.25, 0.04, 2.0, 0.05, -0.9, 0.5
        )
        self.assertTrue(math.isfinite(volatility))
        self.assertGreater(abs(volatility - doubled_eta), 1.0e-3)
        self.assertEqual(
            fukasawa_gatheral_implied_volatility(
                1.0, 1.0, 0.25, 0.04, 1.0, 0.05, -0.9, 0.5
            ),
            0.2,
        )

    def test_black_scholes_implied_volatility_inverts_otm_sides(self) -> None:
        for strike, side in ((0.85, "put"), (1.0, "call"), (1.15, "call")):
            price = black_price(0.23, strike, 0.5, side)
            volatility, vega = black_scholes_implied_volatility(
                price, 1.0, strike, 0.5, 0.0, 0.0, side
            )
            self.assertAlmostEqual(volatility, 0.23, places=11)
            self.assertGreater(vega, 0.0)

    def test_synthetic_paper_campaign_is_accepted(self) -> None:
        parameters = {
            "spot": 1.0,
            "risk_free_rate": 0.0,
            "dividend_yield": 0.0,
            "xi_0": 0.04,
            "eta": 1.0,
            "hurst_exponent": 0.05,
            "rho": -0.9,
            "beta": 0.5,
        }
        maturity = 0.25
        kernel = (
            parameters["eta"]
            * math.sqrt(2.0 * parameters["hurst_exponent"])
            * maturity ** (parameters["hurst_exponent"] - 0.5)
        )
        rows = []
        for scaled_y in (-0.3, 0.0, 0.3):
            log_moneyness = scaled_y * math.sqrt(0.04) / kernel
            strike = math.exp(log_moneyness)
            side = "put" if strike < 1.0 else "call"
            volatility = fukasawa_gatheral_implied_volatility(
                1.0,
                strike,
                maturity,
                0.04,
                1.0,
                0.05,
                -0.9,
                0.5,
            )
            rows.append({
                "scaled_log_moneyness": scaled_y,
                "log_moneyness": log_moneyness,
                "strike": strike,
                "side": side,
                "estimate": {
                    "price": black_price(
                        volatility, strike, maturity, side
                    ),
                    "standard_error": 1.0e-8,
                },
            })
        document = {
            "paper_case": "Fukasawa--Gatheral Figure 6.4",
            "eta_convention": "d_xi_over_xi",
            "parameters": parameters,
            "runs": [
                {
                    "maturity_days": 63,
                    "maturity": maturity,
                    "time_steps": steps,
                    "rows": rows,
                }
                for steps in (2048, 4096)
            ],
        }
        report = analyze_fukasawa_gatheral_campaign(document)
        self.assertTrue(report["summary"]["all_passed"])
        self.assertLess(
            report["summary"][
                "max_abs_normalized_implied_volatility_difference"
            ],
            1.0e-10,
        )

    def test_locked_cuda_paper_campaign(self) -> None:
        fixture = Path(
            "validation/volterra/fixtures/"
            "rough_sabr_fukasawa_gatheral_figure_6_4_1m.json"
        )
        if not fixture.exists():
            self.skipTest("locked CUDA paper campaign has not been generated")
        report = analyze_fukasawa_gatheral_probe(fixture)
        self.assertTrue(report["summary"]["all_passed"])
        self.assertEqual(report["summary"]["row_count"], 28)
        self.assertAlmostEqual(
            report["summary"][
                "max_abs_normalized_implied_volatility_difference"
            ],
            0.01522506853086747,
            places=12,
        )
        self.assertLess(
            report["summary"]["max_refinement_indicator"], 0.0021
        )


if __name__ == "__main__":
    unittest.main()
