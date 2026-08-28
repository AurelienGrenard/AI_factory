"""Contract tests for the typed repository capability matrix."""

from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from capability_manifest import (  # noqa: E402
    AVAILABLE_DATASET_SPECS,
    DEFERRED_DATASET_SPECS,
    FIXED_INCOME_UNITS,
    MODEL_SPECS,
    PRODUCT_SPECS,
    resolve_price_capability,
)


class CapabilityManifestTest(unittest.TestCase):
    def test_declared_cardinalities_are_complete(self) -> None:
        self.assertEqual(len(MODEL_SPECS), 24)
        self.assertEqual(len(PRODUCT_SPECS), 26)
        self.assertEqual(len(AVAILABLE_DATASET_SPECS), 689)
        self.assertEqual(len(DEFERRED_DATASET_SPECS), 0)
        self.assertEqual(len(FIXED_INCOME_UNITS), 35)

    def test_resolver_selects_equity_and_fixed_income_engines(self) -> None:
        self.assertEqual(
            resolve_price_capability(
                "black_scholes",
                "european_option",
                "european_calls",
            ).engine,
            "equity_closed_form",
        )
        self.assertEqual(
            resolve_price_capability(
                "heston",
                "american_option",
                "american_puts",
            ).engine,
            "equity_lsm_fixed",
        )
        self.assertEqual(
            resolve_price_capability(
                "g2_plus_plus",
                "bermudan_swaption",
                "bermudan_payer_swaptions",
                "svensson",
            ).engine,
            "fixed_income_lsm",
        )

    def test_resolver_rejects_undeclared_combinations(self) -> None:
        with self.assertRaises(KeyError):
            resolve_price_capability(
                "black_scholes",
                "american_option",
                "american_calls",
            )
        with self.assertRaises(KeyError):
            resolve_price_capability(
                "g2",
                "european_swaption",
                "european_payer_swaptions",
            )

    def test_sample_publication_is_two_recipes_per_model(self) -> None:
        available_by_model = {
            model.name: {
                dataset.dataset_id
                for dataset in AVAILABLE_DATASET_SPECS
                if dataset.model == model.name
                and dataset.dataset_kind == "samples"
            }
            for model in MODEL_SPECS
        }
        self.assertTrue(all(
            dataset_ids == {"samples_01", "samples_02"}
            for dataset_ids in available_by_model.values()
        ))


if __name__ == "__main__":
    unittest.main()
