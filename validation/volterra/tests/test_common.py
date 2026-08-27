import math
import unittest

import numpy as np
from scipy.special import gammaincc, ndtr

from validation.volterra.common import (
    certify_price,
    lewis_european_option_price,
)


class CommonReferenceTest(unittest.TestCase):
    def test_lewis_inversion_recovers_black_scholes(self) -> None:
        variance = 0.04

        def log_mgf(exponents):
            return 0.5 * variance * (exponents * exponents - exponents)

        price = lewis_european_option_price(
            log_mgf,
            spot=1.0,
            strike=1.0,
            maturity=1.0,
            risk_free_rate=0.02,
            dividend_yield=0.01,
            option_side="call",
        )
        standard_deviation = math.sqrt(variance)
        d1 = (0.02 - 0.01 + 0.5 * variance) / standard_deviation
        expected = math.exp(-0.01) * ndtr(d1) - math.exp(-0.02) * ndtr(
            d1 - standard_deviation
        )
        self.assertAlmostEqual(price, expected, places=12)

    def test_lewis_phase_recovers_asymmetric_gamma_price(self) -> None:
        shape = 3.0
        scale = 0.2
        log_normalization = shape * math.log1p(-scale)
        strike = 1.1

        def log_mgf(exponents):
            return (
                exponents * log_normalization
                - shape * np.log(1.0 - scale * exponents)
            )

        price = lewis_european_option_price(
            log_mgf,
            spot=1.0,
            strike=strike,
            maturity=1.0,
            risk_free_rate=0.0,
            dividend_yield=0.0,
            option_side="call",
        )
        threshold = (math.log(strike) - log_normalization) / scale
        expected = gammaincc(shape, (1.0 - scale) * threshold)
        expected -= strike * gammaincc(shape, threshold)
        self.assertAlmostEqual(price, expected, places=9)

    def test_certification_combines_both_standard_errors(self) -> None:
        accepted = certify_price(1.01, 0.002, 1.0, 0.002, 0.001, 4.0)
        rejected = certify_price(1.02, 0.002, 1.0, 0.002, 0.001, 4.0)
        self.assertTrue(accepted.passed)
        self.assertFalse(rejected.passed)
        self.assertAlmostEqual(
            accepted.allowance,
            0.001 + 4.0 * math.hypot(0.002, 0.002),
        )


if __name__ == "__main__":
    unittest.main()
