import math
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

from validation.volterra.log_modulated_rough_bergomi import (
    LogModulatedRoughBergomiParameters,
    european_option_price,
    hybrid_driver,
    kernel,
)


class LogModulatedRoughBergomiReferenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = LogModulatedRoughBergomiParameters(
            spot=1.0,
            risk_free_rate=0.02,
            dividend_yield=0.01,
            xi_0=0.04,
            eta=1.5,
            hurst_exponent=0.10,
            rho=-0.70,
            log_modulation_scale=0.10,
            log_modulation_power=2.0,
        )

    def test_kernel_has_unit_variance_at_one_year(self) -> None:
        # The logarithmic coordinate keeps the singular endpoint out of the
        # numerical integrand, independently of the production quadrature.
        value, _ = quad(
            lambda x: kernel(np.exp(-x), self.parameters) ** 2 * np.exp(-x),
            0.0,
            80.0,
            epsabs=1.0e-11,
            epsrel=1.0e-11,
        )
        self.assertAlmostEqual(value, 1.0, places=9)

    def test_numpy_fft_reproduces_direct_hybrid_driver(self) -> None:
        generator = np.random.default_rng(20260827)
        rough = generator.standard_normal((5, 63))
        singular = generator.standard_normal((5, 63))
        direct = hybrid_driver(
            self.parameters, 1.3, rough, singular, "direct"
        )
        transformed = hybrid_driver(
            self.parameters, 1.3, rough, singular, "fft"
        )
        np.testing.assert_allclose(transformed, direct, rtol=2.0e-13, atol=2.0e-13)

    def test_h_zero_logarithmic_stress_kernel_remains_integrable(self) -> None:
        stress = LogModulatedRoughBergomiParameters(
            **{**self.parameters.__dict__, "hurst_exponent": 0.0}
        )
        generator = np.random.default_rng(17)
        normals = generator.standard_normal((2, 24))
        values = hybrid_driver(stress, 1.0, normals, -normals, "fft")
        self.assertTrue(np.all(np.isfinite(values)))

    def test_zero_eta_reduces_to_black_scholes(self) -> None:
        constant_variance = LogModulatedRoughBergomiParameters(
            **{**self.parameters.__dict__, "eta": 0.0}
        )
        estimate = european_option_price(
            constant_variance, 1.0, 1.0, "call", 32, 32768, 20260901
        )
        standard_deviation = math.sqrt(constant_variance.xi_0)
        d1 = (
            constant_variance.risk_free_rate
            - constant_variance.dividend_yield
            + 0.5 * constant_variance.xi_0
        ) / standard_deviation
        expected = math.exp(-constant_variance.dividend_yield) * ndtr(d1)
        expected -= math.exp(-constant_variance.risk_free_rate) * ndtr(
            d1 - standard_deviation
        )
        self.assertLess(
            abs(estimate.price - expected),
            4.0 * estimate.standard_error + 2.0e-5,
        )


if __name__ == "__main__":
    unittest.main()
