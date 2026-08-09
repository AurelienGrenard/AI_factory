"""Tests for the fitted Gaussian-rate validation coverage."""

from __future__ import annotations

import unittest

from validation.model.fixed_income.fitted_gaussian_rate import (
    _PRODUCT_KINDS,
    _SPECS,
    _engine_plan,
)


class FittedGaussianRateValidationTest(unittest.TestCase):
    def test_every_pair_has_the_complete_hierarchy(self) -> None:
        self.assertEqual(
            set(_SPECS),
            {
                ("hull_white", "nelson_siegel"),
                ("hull_white", "svensson"),
                ("g2_plus_plus", "nelson_siegel"),
                ("g2_plus_plus", "svensson"),
            },
        )
        self.assertEqual(len(_PRODUCT_KINDS), 4)
        for model_name, curve_name in _SPECS:
            for product_kind in _PRODUCT_KINDS:
                plan = _engine_plan(model_name, curve_name, product_kind)
                self.assertEqual(
                    tuple(engine.label for engine in plan),
                    (
                        "Premia (specialized pricer)",
                        "QuantLib (specialized pricer)",
                        "QuantLib (Monte Carlo)",
                    ),
                )
                self.assertTrue(plan[0].available)
                self.assertTrue(plan[1].available)
                self.assertFalse(plan[2].available)


if __name__ == "__main__":
    unittest.main()
