"""Tests for the declarative Gaussian-rate validation coverage."""

from __future__ import annotations

import unittest

from validation.model.fixed_income.gaussian_rate import (
    _MODELS,
    _PRODUCT_KINDS,
    _engine_plan,
)


class GaussianRateValidationTest(unittest.TestCase):
    def test_every_model_product_pair_has_the_complete_hierarchy(self) -> None:
        self.assertEqual(set(_MODELS), {"vasicek", "ornstein_uhlenbeck", "g2"})
        self.assertEqual(len(_PRODUCT_KINDS), 4)
        for model_name in _MODELS:
            for product_kind in _PRODUCT_KINDS:
                plan = _engine_plan(model_name, product_kind)
                self.assertEqual(
                    tuple(engine.label for engine in plan),
                    (
                        "Premia (specialized pricer)",
                        "QuantLib (specialized pricer)",
                        "QuantLib (Monte Carlo)",
                    ),
                )
                for engine in plan:
                    self.assertEqual(bool(engine.pricing_method), engine.available)

    def test_premia_is_used_only_for_proven_standalone_mappings(self) -> None:
        for product_kind in _PRODUCT_KINDS:
            self.assertTrue(_engine_plan("vasicek", product_kind)[0].available)
            self.assertTrue(
                _engine_plan("ornstein_uhlenbeck", product_kind)[0].available
            )
            self.assertFalse(_engine_plan("g2", product_kind)[0].available)


if __name__ == "__main__":
    unittest.main()
