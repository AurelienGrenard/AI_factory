import unittest

import numpy as np

from validation.volterra.quadratic_rough_heston import (
    QuadraticRoughHestonParameters,
    exponential_cell_average_weights,
    fractional_cell_average_weights,
    simulate_dense_convolution,
    simulate_exponential_lift,
)
from validation.volterra.rough_heston import ExponentialKernel


class QuadraticRoughHestonReferenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = QuadraticRoughHestonParameters(
            spot=1.0,
            risk_free_rate=0.02,
            dividend_yield=0.01,
            initial_feedback=0.10,
            quadratic_scale=0.40,
            quadratic_shift=0.08,
            variance_floor=0.02,
            feedback_rate=1.0,
            feedback_volatility=0.8,
            hurst_exponent=0.10,
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

    def test_factor_recurrence_equals_dense_exponential_convolution(self) -> None:
        generator = np.random.default_rng(20260829)
        normals = generator.standard_normal((8, 90))
        weights = exponential_cell_average_weights(self.kernel, 1.0 / 90.0, 90)
        dense = simulate_dense_convolution(self.parameters, 1.0, normals, weights)
        lifted = simulate_exponential_lift(self.parameters, self.kernel, 1.0, normals)
        np.testing.assert_allclose(lifted.feedback, dense.feedback, rtol=3.0e-13, atol=3.0e-13)
        np.testing.assert_allclose(lifted.log_spot, dense.log_spot, rtol=3.0e-13, atol=3.0e-13)

    def test_seven_factor_paths_track_dense_fractional_reference(self) -> None:
        generator = np.random.default_rng(20260830)
        normals = generator.standard_normal((256, 360))
        fractional_weights = fractional_cell_average_weights(0.10, 1.0 / 360.0, 360)
        fractional = simulate_dense_convolution(
            self.parameters, 1.0, normals, fractional_weights
        )
        lifted = simulate_exponential_lift(self.parameters, self.kernel, 1.0, normals)
        feedback_scale = np.sqrt(np.mean(fractional.feedback[:, 1:] ** 2))
        relative_feedback_error = np.sqrt(
            np.mean((lifted.feedback[:, 1:] - fractional.feedback[:, 1:]) ** 2)
        ) / feedback_scale
        # This is a kernel-approximation indicator, not a tolerance silently
        # attributed to the time scheme.  The same seven-factor H=0.10 fit is
        # around 14% in this pathwise metric while its paired terminal-price
        # effect remains much smaller.
        self.assertLess(relative_feedback_error, 0.16)
        self.assertLess(
            abs(
                np.mean(np.exp(lifted.log_spot))
                - np.mean(np.exp(fractional.log_spot))
            ),
            0.01,
        )
        fractional_call = np.mean(
            np.maximum(np.exp(fractional.log_spot) - 1.0, 0.0)
        )
        lifted_call = np.mean(
            np.maximum(np.exp(lifted.log_spot) - 1.0, 0.0)
        )
        # Same Brownian paths on both sides make this a paired kernel-bias
        # indicator rather than an uninformative comparison of two noisy MC
        # prices.
        self.assertLess(abs(lifted_call - fractional_call), 0.003)

    def test_quadratic_floor_keeps_variance_strictly_positive(self) -> None:
        feedback = np.linspace(-4.0, 4.0, 1001)
        self.assertGreaterEqual(
            float(np.min(self.parameters.variance(feedback))),
            self.parameters.variance_floor,
        )

    def test_original_paper_restricted_parameter_fixture(self) -> None:
        # Table 1 / restricted QRH calibration in Gatheral, Jusselin and
        # Rosenbaum (2020).  The feedback-noise multiplier is fixed here
        # because it is not part of the reported five-parameter calibration.
        published = QuadraticRoughHestonParameters(
            spot=1.0,
            risk_free_rate=0.0,
            dividend_yield=0.0,
            initial_feedback=0.10,
            quadratic_scale=0.384,
            quadratic_shift=0.095,
            variance_floor=0.0025,
            feedback_rate=1.2,
            feedback_volatility=1.0,
            hurst_exponent=0.01,
        )
        self.assertAlmostEqual(float(published.variance(np.array([0.10]))[0]), 0.0025096)

        generator = np.random.default_rng(20260831)
        normals = generator.standard_normal((32, 32))
        weights = fractional_cell_average_weights(0.01, 1.0 / 32.0, 32)
        paths = simulate_dense_convolution(published, 1.0, normals, weights)
        self.assertTrue(np.all(np.isfinite(paths.feedback)))
        self.assertTrue(np.all(np.isfinite(paths.log_spot)))


if __name__ == "__main__":
    unittest.main()
