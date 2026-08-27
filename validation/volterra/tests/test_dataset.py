"""Tests for the isolated AI_factory dataset adapter."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from validation.volterra.common import black_forward_option_price
from validation.volterra.dataset import validate_rough_bergomi_dataset


class DatasetAdapterTests(unittest.TestCase):
    def test_aligned_row_is_loaded_and_certified(self) -> None:
        maturity_days = 21
        days_per_year = 252
        maturity = maturity_days / days_per_year
        price = float(
            black_forward_option_price(
                [1.0],
                1.0,
                1.0,
                [0.04 * maturity],
                "call",
            )[0]
        )
        convention = {
            "unit": "business_day",
            "days_per_year": days_per_year,
        }
        model = {
            "row_count": 1,
            "models": [
                {
                    "id": "m1",
                    "parameters": {
                        "spot": 1.0,
                        "xi_0": 0.04,
                        "risk_free_rate": 0.0,
                        "dividend_yield": 0.0,
                        "eta": 0.0,
                        "hurst_exponent": 0.1,
                        "rho": 0.0,
                    },
                }
            ],
        }
        product = {
            "row_count": 1,
            "time_convention": convention,
            "products": [
                {
                    "id": "p1",
                    "parameters": {
                        "strike": 1.0,
                        "maturity": maturity_days,
                    },
                }
            ],
        }
        prices = {
            "row_count": 1,
            "time_convention": convention,
            "results": [
                {
                    "id": "r1",
                    "model_id": "m1",
                    "product_id": "p1",
                    "outputs": {"price": price, "standard_error": 0.0},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [
                root / name
                for name in ("prices.json", "model.json", "product.json")
            ]
            for path, document in zip(paths, (prices, model, product)):
                path.write_text(json.dumps(document), encoding="utf-8")
            report = validate_rough_bergomi_dataset(
                paths[0],
                paths[1],
                paths[2],
                "call",
                antithetic_pair_count=16,
                batch_pair_count=8,
            )

        self.assertTrue(report["summary"]["generated_vs_hybrid_all_passed"])
        self.assertEqual(report["rows"][0]["time_steps"], 30)
        self.assertAlmostEqual(
            report["rows"][0]["effective_dt"], maturity / 30
        )

    def test_mismatched_time_conventions_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            price_path = root / "prices.json"
            model_path = root / "model.json"
            product_path = root / "product.json"
            price_path.write_text(
                json.dumps(
                    {
                        "row_count": 0,
                        "time_convention": {
                            "unit": "business_day",
                            "days_per_year": 252,
                        },
                        "results": [],
                    }
                ),
                encoding="utf-8",
            )
            model_path.write_text(
                json.dumps({"row_count": 0, "models": []}),
                encoding="utf-8",
            )
            product_path.write_text(
                json.dumps(
                    {
                        "row_count": 0,
                        "time_convention": {
                            "unit": "business_day",
                            "days_per_year": 365,
                        },
                        "products": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "do not match"):
                validate_rough_bergomi_dataset(
                    price_path, model_path, product_path, "call"
                )


if __name__ == "__main__":
    unittest.main()
