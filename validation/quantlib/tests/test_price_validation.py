"""Check row tolerances and aggregate signed-bias detection."""

import unittest

from validation.quantlib.price_validation import (
    PriceResultRow,
    PriceValidationInput,
    PriceComparison,
    ValidationTolerances,
    select_validation_regime,
    select_validation_row_ids,
    summarize_price_comparisons,
)


class PriceValidationTest(unittest.TestCase):
    """Exercise the generic report independently of any pricing model."""

    def test_small_symmetric_errors_pass(self) -> None:
        comparisons = (
            PriceComparison("000001", 1.0 + 1.0e-8, 1.0),
            PriceComparison("000002", 1.0 - 1.0e-8, 1.0),
        )
        report = summarize_price_comparisons("symmetric", comparisons)
        self.assertTrue(report.passed)
        self.assertEqual(report.higher_price_count, 1)
        self.assertEqual(report.lower_price_count, 1)

    def test_material_positive_bias_fails(self) -> None:
        comparisons = tuple(
            PriceComparison(f"{index:06d}", 1.01, 1.0)
            for index in range(1, 101)
        )
        report = summarize_price_comparisons(
            "biased",
            comparisons,
            ValidationTolerances(absolute=1.0e-6, relative=1.0e-6),
        )
        self.assertFalse(report.passed)
        self.assertEqual(report.failed_row_count, len(comparisons))
        self.assertTrue(report.systematic_bias)

    def test_independent_monte_carlo_errors_are_combined(self) -> None:
        comparisons = (
            PriceComparison(
                "000001",
                1.01,
                1.0,
                generated_standard_error=0.002,
                quantlib_standard_error=0.002,
            ),
        )
        report = summarize_price_comparisons(
            "monte-carlo",
            comparisons,
            ValidationTolerances(absolute=0.0, relative=0.0),
        )
        self.assertTrue(report.passed)

    def test_core_and_stress_regimes_preserve_order(self) -> None:
        rows = tuple(
            PriceResultRow(str(index), str(index), str(index), float(index))
            for index in range(1000)
        )
        validation_input = PriceValidationInput(
            "ordered", __file__, __file__, rows
        )
        core = select_validation_regime(validation_input, "core")
        stress = select_validation_regime(validation_input, "stress")
        self.assertEqual(len(core.rows), 900)
        self.assertEqual(core.rows[-1].row_id, "899")
        self.assertEqual(len(stress.rows), 100)
        self.assertEqual(stress.rows[0].row_id, "900")

    def test_explicit_fallback_rows_preserve_catalogue_order(self) -> None:
        rows = tuple(
            PriceResultRow(str(index), str(index), str(index), float(index))
            for index in range(5)
        )
        validation_input = PriceValidationInput(
            "fallback", __file__, __file__, rows
        )
        selected = select_validation_row_ids(validation_input, ("4", "1"))
        self.assertEqual([row.row_id for row in selected.rows], ["1", "4"])

        with self.assertRaisesRegex(ValueError, "no rows"):
            select_validation_row_ids(validation_input, ("missing",))


if __name__ == "__main__":
    unittest.main()
