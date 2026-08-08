"""Tests for the declarative Black-Scholes validation coverage."""

from __future__ import annotations

import unittest

from validation.model.equity.black_scholes.validation import (
    _PATH_BOUNDS,
    _PRODUCT_KINDS,
    _TERMINAL_PRODUCTS,
    _engine_plan,
    _spec,
)


class BlackScholesValidationCommonTest(unittest.TestCase):
    def test_every_catalogue_product_has_exactly_three_hierarchy_slots(self) -> None:
        self.assertEqual(len(_PRODUCT_KINDS), 29)
        for product_kind in _PRODUCT_KINDS:
            plan = _engine_plan(_spec(product_kind))
            self.assertEqual(len(plan), 3)
            self.assertEqual(
                tuple(engine.label for engine in plan),
                (
                    "Premia (specialized pricer)",
                    "QuantLib (specialized pricer)",
                    "QuantLib (Monte Carlo)",
                ),
            )

    def test_premia_products_are_declared_as_exact_or_directional(self) -> None:
        for product_kind in _TERMINAL_PRODUCTS:
            self.assertEqual(_spec(product_kind).premia_kind, "terminal")
        for product_kind in _PATH_BOUNDS:
            spec = _spec(product_kind)
            self.assertEqual(spec.premia_kind, "path")
            self.assertIsNotNone(spec.bias_explanation)

    def test_only_autocalls_and_cliquet_lack_a_premia_engine(self) -> None:
        quantlib_only = {
            product_kind
            for product_kind in _PRODUCT_KINDS
            if _spec(product_kind).premia_kind is None
        }
        self.assertEqual(
            quantlib_only,
            {
                "athena_autocall",
                "cliquet",
                "phoenix_autocall",
                "phoenix_memory_autocall",
            },
        )
        self.assertEqual(len(_PRODUCT_KINDS) - len(quantlib_only), 25)


if __name__ == "__main__":
    unittest.main()
