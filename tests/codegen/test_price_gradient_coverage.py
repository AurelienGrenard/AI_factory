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
    PRODUCT_BINDING_SPECS,
)
from price_gradients.coverage import markovian_coverage  # noqa: E402
from price_gradients.manifest import default_sensitivities  # noqa: E402


class PriceGradientCoverageTest(unittest.TestCase):
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
        self.assertEqual(len(rows), 301)
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
