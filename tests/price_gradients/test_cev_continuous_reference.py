"""Continuous-time CEV reference checks, including sub-day maturities."""
import math
import unittest

import QuantLib as ql

from validation.quantlib.model.equity.cev.european_option import (
    _price, continuous_time_price,
)


class CevContinuousReferenceTests(unittest.TestCase):
    def test_matches_quantlib_on_representable_dates(self):
        cases = [
            ({"spot": 1.05, "risk_free_rate": .03, "dividend_yield": .01,
              "sigma": .25, "beta": .75}, {"strike": 1., "maturity": 126/252}),
            ({"spot": 1., "risk_free_rate": 0., "dividend_yield": 0.,
              "sigma": .2, "beta": .5}, {"strike": 1., "maturity": 1/252}),
            ({"spot": 1.05, "risk_free_rate": -.01, "dividend_yield": .01,
              "sigma": .25, "beta": .75}, {"strike": 1., "maturity": 504/252}),
        ]
        for model, product in cases:
            for side in (ql.Option.Call, ql.Option.Put):
                self.assertAlmostEqual(
                    continuous_time_price(model, product, side), _price(model, product, side), places=12)

    def test_sub_day_maturity_derivative_is_not_quantized(self):
        model = {"spot": 1., "risk_free_rate": 0., "dividend_yield": 0.,
                 "sigma": .2, "beta": .75}
        maturity, h = 1/252, 1e-5
        lower = continuous_time_price(model, {"strike": 1., "maturity": maturity-h}, ql.Option.Call)
        upper = continuous_time_price(model, {"strike": 1., "maturity": maturity+h}, ql.Option.Call)
        derivative = (upper-lower)/(2*h)
        self.assertTrue(math.isfinite(derivative))
        self.assertGreater(abs(derivative), .1)
        self.assertNotEqual(lower, upper)


if __name__ == "__main__":
    unittest.main()
