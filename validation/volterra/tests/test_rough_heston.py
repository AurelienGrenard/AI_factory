import unittest

import numpy as np

from validation.volterra.rough_heston import (
    ExponentialKernel,
    RoughHestonParameters,
    lifted_heston_european_option_price,
    lifted_heston_log_forward_mgf,
    rough_heston_european_option_price,
    rough_heston_log_forward_mgf,
)


class RoughHestonReferenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = RoughHestonParameters(
            spot=1.0,
            risk_free_rate=0.02,
            dividend_yield=0.01,
            initial_variance=0.04,
            mean_reversion=0.30,
            variance_drift=0.02,
            volatility_of_variance=0.30,
            hurst_exponent=0.10,
            rho=-0.70,
        )
        self.kernel = ExponentialKernel(
            nodes=(
                0.19375584,
                1.4204239,
                5.1622634,
                17.537029,
                55.922745,
                179.97806,
                597.90723,
            ),
            weights=(
                0.68750739,
                0.41549641,
                0.75919181,
                1.113852,
                1.7559471,
                2.8505118,
                4.8338389,
            ),
        )

    def test_fractional_and_lifted_mgf_preserve_mass_and_martingale(self) -> None:
        exponents = np.asarray([0.0, 1.0], dtype=np.complex128)
        fractional = rough_heston_log_forward_mgf(
            self.parameters, exponents, 1.0, 64
        )
        lifted = lifted_heston_log_forward_mgf(
            self.parameters, self.kernel, exponents, 1.0
        )
        np.testing.assert_allclose(fractional, 0.0, atol=2.0e-13)
        np.testing.assert_allclose(lifted, 0.0, atol=2.0e-13)

    def test_seven_factor_price_is_compared_to_continuous_rough_price(self) -> None:
        rough_coarse = rough_heston_european_option_price(
            self.parameters, 1.0, 1.0, "call", 128, 40.0, 401
        )
        rough_fine = rough_heston_european_option_price(
            self.parameters, 1.0, 1.0, "call", 256, 40.0, 401
        )
        lifted = lifted_heston_european_option_price(
            self.parameters, self.kernel, 1.0, 1.0, "call", 40.0, 401
        )
        self.assertLess(abs(rough_fine - rough_coarse), 5.0e-6)
        self.assertLess(abs(lifted - rough_fine), 2.0e-4)

    def test_half_hurst_fractional_solver_converges_to_classical_heston(self) -> None:
        classical = RoughHestonParameters(
            1.0, 0.02, 0.01, 0.04, 0.30, 0.02, 0.30, 0.50, -0.70
        )
        constant_kernel = ExponentialKernel(
            nodes=(0.0,), weights=(1.0,), initial_factors=(0.04,)
        )
        fractional = rough_heston_european_option_price(
            classical, 1.0, 1.0, "call", 256, 40.0, 401
        )
        ordinary_riccati = lifted_heston_european_option_price(
            classical, constant_kernel, 1.0, 1.0, "call", 40.0, 401
        )
        self.assertLess(abs(fractional - ordinary_riccati), 1.0e-7)

    def test_zero_volterra_drift_and_vol_of_vol_reduce_to_black_price(self) -> None:
        deterministic = RoughHestonParameters(
            1.0, 0.02, 0.01, 0.04, 0.0, 0.0, 0.0, 0.1, -0.7
        )
        fractional = rough_heston_european_option_price(
            deterministic, 1.0, 1.0, "call", 32
        )
        lifted = lifted_heston_european_option_price(
            deterministic, self.kernel, 1.0, 1.0, "call"
        )
        self.assertAlmostEqual(fractional, 0.08349405767096774, places=11)
        self.assertAlmostEqual(lifted, 0.08349405767096774, places=11)


if __name__ == "__main__":
    unittest.main()
