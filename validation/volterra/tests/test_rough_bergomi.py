import math
import unittest

import numpy as np
from scipy.special import ndtr

from validation.volterra.rough_bergomi import (
    RoughBergomiParameters,
    exact_gaussian_european_option_price,
    gaussian_grid,
    hybrid_european_option_price,
    hybrid_normalized_driver_direct,
    hybrid_normalized_driver_fft,
)


class RoughBergomiReferenceTest(unittest.TestCase):
    def test_joint_gaussian_covariance_is_reconstructed(self) -> None:
        grid = gaussian_grid(0.1, 1.0, 12)
        reconstructed = grid.cholesky @ grid.cholesky.T
        expected_driver_variances = (
            np.arange(1, 13, dtype=np.float64) / 12.0
        ) ** 0.2
        np.testing.assert_allclose(reconstructed, grid.covariance, atol=2.0e-13)
        np.testing.assert_allclose(
            np.diag(grid.covariance)[12:],
            expected_driver_variances,
            rtol=2.0e-13,
        )
        terminal_driver_brownian_covariance = np.sum(
            grid.covariance[:12, -1]
        )
        expected_cross_covariance = math.sqrt(0.2) / 0.6
        self.assertAlmostEqual(
            terminal_driver_brownian_covariance,
            expected_cross_covariance,
            places=13,
        )
        self.assertLessEqual(grid.diagonal_jitter, 1.0e-12)

    def test_numpy_fft_reproduces_direct_hybrid_driver(self) -> None:
        generator = np.random.default_rng(42)
        rough = generator.standard_normal((5, 37))
        singular = generator.standard_normal((5, 37))
        increments = math.sqrt(1.0 / 37.0) * rough
        direct = hybrid_normalized_driver_direct(
            0.07, 1.0, increments, rough, singular
        )
        transformed = hybrid_normalized_driver_fft(
            0.07, 1.0, increments, rough, singular
        )
        np.testing.assert_allclose(direct, transformed, rtol=2.0e-13, atol=2.0e-13)

    def test_zero_eta_and_zero_rho_reduce_to_black_scholes(self) -> None:
        parameters = RoughBergomiParameters(
            1.0, 0.02, 0.01, 0.04, 0.0, 0.1, 0.0
        )
        hybrid = hybrid_european_option_price(
            parameters, 1.0, 1.0, "call", 24, 128, 7
        )
        exact = exact_gaussian_european_option_price(
            parameters, 1.0, 1.0, "call", 24, 128, 8
        )
        standard_deviation = 0.2
        d1 = (0.02 - 0.01 + 0.5 * 0.04) / standard_deviation
        expected = math.exp(-0.01) * ndtr(d1) - math.exp(-0.02) * ndtr(
            d1 - standard_deviation
        )
        self.assertAlmostEqual(hybrid.price, expected, places=13)
        self.assertAlmostEqual(exact.price, expected, places=13)
        self.assertLess(hybrid.standard_error, 1.0e-15)
        self.assertLess(exact.standard_error, 1.0e-15)


if __name__ == "__main__":
    unittest.main()
