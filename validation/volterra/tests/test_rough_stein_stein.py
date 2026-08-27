import math
import unittest

import numpy as np
from scipy.special import ndtr

from validation.volterra.rough_stein_stein import (
    RoughSteinSteinParameters,
    european_option_price,
    fractional_resolvent,
    gaussian_driver,
    hybrid_gaussian_driver,
    resolvent_equation_residual,
)


class RoughSteinSteinReferenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = RoughSteinSteinParameters(0.20, 1.0, 0.30, 0.10)

    def test_resolvent_satisfies_fractional_volterra_equation(self) -> None:
        for time in (0.01, 0.10, 0.50, 1.0):
            self.assertLess(
                abs(resolvent_equation_residual(time, self.parameters)),
                2.0e-8,
            )

    def test_numpy_fft_reproduces_direct_resolvent_convolution(self) -> None:
        generator = np.random.default_rng(20260828)
        increments = generator.standard_normal((4, 48)) * np.sqrt(1.0 / 48.0)
        direct = gaussian_driver(self.parameters, 1.0, increments, "direct")
        transformed = gaussian_driver(self.parameters, 1.0, increments, "fft")
        np.testing.assert_allclose(transformed, direct, rtol=2.0e-13, atol=2.0e-13)

    def test_numpy_fft_reproduces_production_hybrid_driver(self) -> None:
        generator = np.random.default_rng(20260902)
        rough = generator.standard_normal((4, 48))
        singular = generator.standard_normal((4, 48))
        direct = hybrid_gaussian_driver(
            self.parameters, 1.0, rough, singular, "direct"
        )
        transformed = hybrid_gaussian_driver(
            self.parameters, 1.0, rough, singular, "fft"
        )
        np.testing.assert_allclose(
            transformed, direct, rtol=2.0e-13, atol=2.0e-13
        )

    def test_zero_vol_of_vol_reduces_to_black_scholes(self) -> None:
        constant_volatility = RoughSteinSteinParameters(
            volatility_level=0.20,
            mean_reversion=1.0,
            volatility_of_volatility=0.0,
            hurst_exponent=0.10,
            spot=1.0,
            risk_free_rate=0.02,
            dividend_yield=0.01,
            rho=-0.70,
        )
        estimate = european_option_price(
            constant_volatility, 1.0, 1.0, "call", 32, 32768, 20260903
        )
        d1 = (0.02 - 0.01 + 0.5 * 0.20**2) / 0.20
        expected = math.exp(-0.01) * ndtr(d1)
        expected -= math.exp(-0.02) * ndtr(d1 - 0.20)
        self.assertLess(
            abs(estimate.price - expected),
            4.0 * estimate.standard_error + 2.0e-5,
        )

    def test_stress_mittag_leffler_tail_is_finite_and_positive(self) -> None:
        stress = RoughSteinSteinParameters(0.20, 8.0, 1.5, 0.45)
        values = np.array(
            [fractional_resolvent(time, stress) for time in (1.0, 5.0, 7.0)]
        )
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all(values > 0.0))
        self.assertTrue(np.all(np.diff(values) < 0.0))


if __name__ == "__main__":
    unittest.main()
