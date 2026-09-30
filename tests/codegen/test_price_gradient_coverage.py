"""Keep the Markovian gradient inventory tied to the pricing manifest."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/codegen/pricing_bindings"))

from capability_manifest import (  # noqa: E402
    MODEL_BY_NAME,
    PRICE_GRADIENT_BINDING_SPECS,
    PRICE_GRADIENT_DATASET_SPECS,
    PRICE_GRADIENT_DATASET_VARIANTS,
    PRODUCT_BINDING_SPECS,
    validate_dataset_spec,
)
from price_gradients.coverage import markovian_coverage  # noqa: E402
from price_gradients.manifest import default_sensitivities  # noqa: E402


class PriceGradientCoverageTest(unittest.TestCase):
    def test_catalogue_publishes_only_diagonal_hessian_recipes(self):
        self.assertEqual(len(PRICE_GRADIENT_DATASET_VARIANTS), 3620)
        self.assertEqual(len(PRICE_GRADIENT_DATASET_SPECS), 1020)
        self.assertEqual(
            {spec.sensitivity_orders for spec in PRICE_GRADIENT_DATASET_SPECS},
            {("first", "diagonal_second")},
        )
        self.assertTrue(all(
            spec.dataset_kind == "price_gradients"
            and spec.dataset_id.endswith(
                (
                    "_price_gradient_diagonal_hessian",
                    "_price_gradient_diagonal_hessian_frozen_policy",
                )
            )
            and "/price_gradients/" in spec.generator_path
            for spec in PRICE_GRADIENT_DATASET_SPECS
        ))
        self.assertEqual(
            {spec.sensitivity_orders for spec in PRICE_GRADIENT_DATASET_VARIANTS},
            {
                ("first",),
                ("first", "diagonal_second"),
                ("first", "diagonal_second", "mixed_second"),
            },
        )

    def test_early_exercise_recipes_have_explicit_distinct_replay(self):
        early_exercise = [
            spec for spec in PRICE_GRADIENT_DATASET_VARIANTS
            if spec.product in {"american_option", "bermudan_swaption"}
        ]
        self.assertTrue(early_exercise)
        self.assertEqual(
            {spec.exercise_replay for spec in early_exercise},
            {"frozen_exercise_time", "frozen_regression_policy"},
        )
        self.assertTrue(all(
            spec.exercise_replay is None
            for spec in PRICE_GRADIENT_DATASET_VARIANTS
            if spec.product not in {"american_option", "bermudan_swaption"}
        ))
        published = [
            spec for spec in PRICE_GRADIENT_DATASET_SPECS
            if spec.product in {"american_option", "bermudan_swaption"}
        ]
        frozen_time_ids = {
            spec.dataset_id for spec in published
            if spec.exercise_replay == "frozen_exercise_time"
        }
        frozen_policy = [
            spec for spec in published
            if spec.exercise_replay == "frozen_regression_policy"
        ]
        self.assertEqual(len(frozen_time_ids), 84)
        self.assertEqual(len(frozen_policy), 84)
        self.assertTrue(all(
            spec.dataset_id.removesuffix("_frozen_policy") in frozen_time_ids
            for spec in frozen_policy
        ))
        for spec in PRICE_GRADIENT_DATASET_SPECS:
            if spec.product not in {"american_option", "bermudan_swaption"}:
                continue
            generator = (ROOT / spec.generator_path).read_text()
            recipe = yaml.safe_load(
                (ROOT / spec.recipe_yaml_path).read_text()
            )
            self.assertEqual(
                recipe.get("exercise_replay"), spec.exercise_replay
            )
            method = recipe["numerical_method"]
            self.assertEqual(method["algorithm"], "longstaff_schwartz")
            self.assertEqual(
                method["regression"]["feature_normalization"],
                "central_lsm_row",
            )
            expected_basis = (
                "laguerre_polynomial_two_factor_6_term"
                if spec.product == "american_option"
                else "hermite_probabilists_two_factor_quadratic_6_term"
                if spec.model in {"g2", "g2_plus_plus"}
                else "hermite_probabilists_one_factor_degree_3"
            )
            self.assertEqual(method["regression"]["basis"], expected_basis)
            self.assertIn(spec.exercise_replay, generator)
            self.assertIn("with_replay", generator)

        original = next(
            spec for spec in early_exercise
            if spec.exercise_replay == "frozen_exercise_time"
        )
        with self.assertRaisesRegex(ValueError, "distinct frozen_policy"):
            validate_dataset_spec(replace(
                original,
                exercise_replay="frozen_regression_policy",
            ))

    def test_monte_carlo_diagonal_generators_expose_both_execution_strategies(self):
        node_graph_products = {
            "european_option",
            "asset_or_nothing_option",
            "digital_option",
            "gap_option",
            "straddle",
            "asian_option",
            "athena_autocall",
            "cliquet",
            "double_knock_out_option",
            "down_and_in_option",
            "down_and_out_option",
            "forward_start_option",
            "geometric_asian_option",
            "lookback_option",
            "phoenix_autocall",
            "phoenix_memory_autocall",
            "range_accrual",
            "up_and_in_option",
            "up_and_out_option",
            "up_no_touch",
            "up_one_touch",
        }
        recipes = [
            spec
            for spec in PRICE_GRADIENT_DATASET_SPECS
            if spec.asset_class == "equity"
            and spec.engine == "equity_markovian"
            and spec.product in node_graph_products
            and spec.sensitivity_orders == ("first", "diagonal_second")
        ]
        self.assertEqual(len(recipes), 668)
        self.assertEqual(
            len({(spec.model, spec.product) for spec in recipes}),
            244,
        )
        for spec in recipes:
            generator = ROOT / spec.generator_path
            metadata = yaml.safe_load(generator.with_name("recipe.yaml").read_text())
            self.assertIn("execute_node_graph_dataset", generator.read_text())
            self.assertNotIn("sensitivity_execution", metadata)

    def test_every_markovian_pricer_has_one_inventory_row(self):
        rows = markovian_coverage(
            PRODUCT_BINDING_SPECS,
            PRICE_GRADIENT_BINDING_SPECS,
            MODEL_BY_NAME,
        )
        keys = [
            (row["model"], row["curve"], row["product"], row["engine"])
            for row in rows
        ]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(len(rows), 313)
        self.assertEqual(
            sum(row["coverage"] != "missing_binding" for row in rows),
            len(PRICE_GRADIENT_BINDING_SPECS),
        )
        self.assertEqual(
            sum(row["coverage"] == "full_hessian" for row in rows),
            len(PRICE_GRADIENT_BINDING_SPECS),
        )
        self.assertEqual(
            {(row["model"], row["product"])
             for row in rows if row["coverage"] == "first_only"},
            set(),
        )

    def test_g2_bermudan_recipes_select_both_initial_states(self):
        names = {
            item["parameter"]
            for item in default_sensitivities("g2", "bermudan_swaption")
        }
        self.assertIn("model.initial_state_x", names)
        self.assertIn("model.initial_state_y", names)
        self.assertEqual(len(names), 10)

    def test_terminal_time_is_selected_except_for_early_exercise(self):
        maturity = "product.maturity_years"
        for model, product in (
            ("heston", "european_option"),
            ("heston", "asian_option"),
            ("black_scholes", "forward_start_option"),
            ("cir", "rate_option"),
            ("cir", "zero_coupon_bond_option"),
            ("cir", "european_swaption"),
        ):
            selected = {
                item["parameter"]: item
                for item in default_sensitivities(model, product)
            }
            self.assertIn(maturity, selected)
            self.assertEqual(selected[maturity]["scale"], "absolute")
            self.assertEqual(
                selected[maturity]["displacement"], 1.0 / 504.0
            )

        for model, product in (
            ("heston", "american_option"),
            ("g2", "bermudan_swaption"),
        ):
            names = {
                item["parameter"]
                for item in default_sensitivities(model, product)
            }
            self.assertNotIn(maturity, names)

    def test_gradient_orders_and_binding_identity_are_checked(self):
        spec = PRICE_GRADIENT_BINDING_SPECS[0]
        with self.assertRaisesRegex(ValueError, "derivative orders"):
            replace(spec, supported_orders=("diagonal_second",))
        with self.assertRaisesRegex(ValueError, "derivative orders"):
            replace(spec, supported_orders=("first", "first"))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            markovian_coverage(
                PRODUCT_BINDING_SPECS,
                (spec, spec),
                MODEL_BY_NAME,
            )


if __name__ == "__main__":
    unittest.main()
