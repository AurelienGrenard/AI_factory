"""Focused contracts for one-factor European-swaption validation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest

from validation.premia.model.fixed_income.swaption import _prepared_inputs


ROOT = Path(__file__).resolve().parents[2]


class SwaptionValidationTest(unittest.TestCase):
    def test_premia_domain_partitions_are_explicit(self) -> None:
        vasicek = (
            ROOT
            / "datasets/model/fixed_income/vasicek/prices/"
            "european_payer_swaptions/"
            "vasicek_01__european_payer_swaptions_01__01.json"
        )
        hull_white = (
            ROOT
            / "datasets/model/fixed_income/hull_white/prices/nelson_siegel/"
            "european_payer_swaptions/"
            "hull_white_01__nelson_siegel_01__"
            "european_payer_swaptions_01__01.json"
        )
        _, vasicek_inputs, vasicek_exceptions = _prepared_inputs(
            vasicek, "vasicek", None, "payer", "all", None
        )
        _, hull_white_inputs, hull_white_exceptions = _prepared_inputs(
            hull_white, "hull_white", "nelson_siegel", "payer", "all", None
        )
        self.assertEqual((len(vasicek_inputs), len(vasicek_exceptions)), (948, 52))
        self.assertEqual(
            (len(hull_white_inputs), len(hull_white_exceptions)), (912, 88)
        )

    @unittest.skipUnless(
        importlib.util.find_spec("QuantLib") is not None,
        "QuantLib is not installed",
    )
    def test_quantlib_jamshidian_obeys_payer_receiver_parity(self) -> None:
        from validation.quantlib.model.fixed_income.vasicek.reference import (
            quantlib_model,
        )
        from validation.quantlib.price_validation import load_parameter_rows
        from validation.quantlib.swaption import swaption_price, swaption_times

        model_parameters = load_parameter_rows(
            ROOT / "datasets/model/fixed_income/vasicek/parameters/vasicek_01.json",
            "models",
        )["000001"]
        product = load_parameter_rows(
            ROOT
            / "datasets/product/european_swaption/"
            "european_swaptions_01.json",
            "products",
        )["000001"]
        model = quantlib_model(model_parameters, None, product)
        payer = swaption_price(model, product, "payer")
        receiver = swaption_price(model, product, "receiver")
        times = swaption_times(product)
        state = float(model_parameters["initial_state"])
        strike = float(product["strike"])
        accrual = float(product["accrual_fraction"])
        swap_value = model.discountBond(0.0, times[0], state) - sum(
            (strike * accrual + (index == len(times) - 1))
            * model.discountBond(0.0, maturity, state)
            for index, maturity in enumerate(times[1:], start=1)
        )
        self.assertAlmostEqual(payer - receiver, swap_value, places=12)

    def test_persisted_pricer_counts_cover_every_swaption_row(self) -> None:
        references = tuple(
            sorted(
                path
                for path in (
                    ROOT / "validation/datasets/price/fixed_income"
                ).rglob("*.json")
                if "swaption" in path.as_posix()
            )
        )
        self.assertEqual(len(references), 10)
        for path in references:
            document = json.loads(path.read_text(encoding="utf-8"))
            core = document["reference_pricers"]["core"]
            stress = document["reference_pricers"]["stress"]
            if "/cir/" in path.as_posix():
                expected = ((0, 900), (0, 100))
            elif "/hull_white/" in path.as_posix():
                expected = ((900, 0), (12, 88))
            else:
                expected = ((900, 0), (48, 52))
            actual = tuple(
                (
                    section["premia"].get("row_priced", 0),
                    section["quantlib_specialized"].get("row_priced", 0),
                )
                for section in (core, stress)
            )
            self.assertEqual(actual, expected, document["database_id"])
            self.assertEqual(document["verification"]["status"], "passed")


if __name__ == "__main__":
    unittest.main()
