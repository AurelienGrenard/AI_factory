"""Structural tests for the uniform stochastic-equity validation pipeline."""

from __future__ import annotations

import importlib
from pathlib import Path
import unittest

from validation.model.equity.bates.validation import SPEC as BATES_SPEC
from validation.model.equity.heston.validation import SPEC as HESTON_SPEC
from validation.model.equity.kou.validation import SPEC as KOU_SPEC
from validation.model.equity.merton.validation import SPEC as MERTON_SPEC
from validation.model.equity.stochastic_equity import engine_plan


_SPECS = (MERTON_SPEC, KOU_SPEC, HESTON_SPEC, BATES_SPEC)


class StochasticEquityValidationTest(unittest.TestCase):
    def test_every_product_has_premia_candidates_before_quantlib(self) -> None:
        for spec in _SPECS:
            for product_kind in spec.product_kinds:
                plan = engine_plan(spec, product_kind)
                self.assertEqual(
                    tuple(engine.label for engine in plan[-2:]),
                    (
                        "QuantLib (specialized pricer)",
                        "QuantLib (Monte Carlo)",
                    ),
                )
                self.assertTrue(plan[:-2])
                self.assertTrue(
                    all(
                        engine.label == "Premia (specialized pricer)"
                        for engine in plan[:-2]
                    )
                )
                for engine in plan:
                    self.assertEqual(bool(engine.pricing_method), engine.available)

    def test_every_declared_product_has_one_thin_cli_module(self) -> None:
        root = Path(__file__).resolve().parents[1] / "model/equity"
        for spec in _SPECS:
            for product_kind in spec.product_kinds:
                module_path = root / spec.model_name / f"{product_kind}.py"
                self.assertTrue(module_path.is_file(), module_path)
                importlib.import_module(
                    f"validation.model.equity.{spec.model_name}.{product_kind}"
                )

    def test_declared_bias_explanations_only_cover_validated_products(self) -> None:
        for spec in _SPECS:
            explanations = spec.bias_explanations or {}
            self.assertTrue(set(explanations) <= spec.premia_products)

    def test_kou_asian_uses_unbiased_mc_before_analytic_fallbacks(self) -> None:
        for product_kind in ("asian_call", "asian_put"):
            plan = engine_plan(KOU_SPEC, product_kind)
            self.assertTrue(
                plan[0].pricing_method.startswith("MC_FixedAsian_IS_Lelong")
            )
            self.assertEqual(
                plan[1].pricing_method,
                "AP_FixedAsian_FusaiMeucci_Kou",
            )

    def test_quantlib_coverage_is_split_without_overlap(self) -> None:
        for spec in (HESTON_SPEC, BATES_SPEC):
            self.assertFalse(
                spec.quantlib_specialized_products
                & spec.quantlib_monte_carlo_products
            )
            self.assertEqual(
                spec.quantlib_specialized_products
                | spec.quantlib_monte_carlo_products,
                spec.product_kinds,
            )


if __name__ == "__main__":
    unittest.main()
