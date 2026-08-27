"""Tests for the regular Bermudan-swaption QuantLib adapter."""

from __future__ import annotations

import unittest

from validation.quantlib.bermudan_swaption import (
    BermudanEngineConfiguration,
    bermudan_swaption_times,
    short_exercise_engine,
)


class BermudanSwaptionTest(unittest.TestCase):
    def test_regular_payment_times_include_swap_start(self) -> None:
        product = {
            "first_exercise_time": 0.5,
            "payment_interval": 1.0,
            "payment_count": 3,
            "exercise_count": 2,
        }

        self.assertEqual(
            bermudan_swaption_times(product), (0.5, 1.5, 2.5, 3.5)
        )

    def test_very_short_exercise_uses_finite_difference(self) -> None:
        short_product = {"first_exercise_time": 1.0 / 252.0}
        regular_product = {"first_exercise_time": 0.5}

        self.assertEqual(
            short_exercise_engine(short_product, "fd_hull_white"),
            "fd_hull_white",
        )
        self.assertEqual(
            short_exercise_engine(regular_product, "fd_hull_white"),
            "tree",
        )

    def test_engine_configuration_rejects_empty_grids(self) -> None:
        with self.assertRaisesRegex(ValueError, "grid sizes"):
            BermudanEngineConfiguration(tree_steps=0)


if __name__ == "__main__":
    unittest.main()
