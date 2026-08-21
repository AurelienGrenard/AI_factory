"""Tests for the common validation-notebook report."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from validation.reporting import (
    DatasetValidationReport,
    EnginePlanEntry,
    FallbackDiagnostic,
    SpecialRowDiagnostic,
    ValidationDisplayReport,
    format_validation_report,
    has_directional_bias,
    load_validation_report,
    synchronize_validation_yaml,
    validation_fingerprint,
    write_validation_report,
)


def _write_price_fixture(root: Path) -> tuple[Path, Path]:
    (root / "CMakeLists.txt").write_text("# fixture\n", encoding="utf-8")
    catalog = root / "catalog/model/equity/sample/prices/sample_01"
    catalog.mkdir(parents=True)
    yaml_path = catalog / "dataset.yaml"
    yaml_path.write_text(
        "summary:\n"
        "  pricing_method: \"Monte Carlo\"\n"
        "validation:\n"
        "  status: \"pending\"\n"
        "time_grid:\n"
        "  target_dt: \"1 / 360\"\n"
        "outputs:\n"
        "  price:\n"
        "    estimator: \"mean\"\n"
        "price_construction:\n"
        "  method: \"Aligned\"\n",
        encoding="utf-8",
    )
    dataset_path = root / "prices.json"
    dataset_path.write_text(
        json.dumps(
            {
                "database_id": "sample",
                "catalog": "catalog/model/equity/sample/prices/sample_01",
                "row_count": 1,
                "model_dataset": {"id": "model_01"},
                "product_dataset": {"id": "product_01"},
                "timing": {"wall_seconds": 1.0},
                "results": [
                    {
                        "id": "000001",
                        "model_id": "000001",
                        "product_id": "000001",
                        "seed": 1,
                        "outputs": {"price": 0.1, "standard_error": 0.01},
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return dataset_path, yaml_path


class ValidationReportingTest(unittest.TestCase):
    def test_directional_bias_requires_strictly_more_than_sixty_percent(self) -> None:
        self.assertFalse(has_directional_bias(60, 40, 0))
        self.assertTrue(has_directional_bias(61, 39, 0))

    def test_report_renders_common_metrics_and_special_treatment(self) -> None:
        report = ValidationDisplayReport(
            title="Stress validation",
            status="passed",
            database_id="sample",
            reference="Premia (specialized pricer)",
            pricing_method="CF_Call",
            tolerance="five combined standard errors",
            row_count=2,
            accepted_row_count=1,
            failed_row_count=0,
            higher_price_count=1,
            lower_price_count=0,
            equal_price_count=1,
            mean_signed_price_gap=1.0e-4,
            mean_absolute_price_gap=1.0e-4,
            maximum_absolute_price_gap=2.0e-4,
            maximum_absolute_price_gap_row_id="000001",
            systematic_bias=False,
            engine_plan=(
                EnginePlanEntry(
                    "Premia (specialized pricer)", "CF_Call", True
                ),
                EnginePlanEntry(
                    "QuantLib (specialized pricer)",
                    None,
                    False,
                    "no compatible engine",
                ),
            ),
            special_rows=(
                SpecialRowDiagnostic(
                    "000002", "backend failure", "fallback accepted"
                ),
            ),
            fallbacks=(
                FallbackDiagnostic(
                    "000002",
                    "QuantLib (Monte Carlo)",
                    "_antithetic_path_price (GaussianPathGenerator)",
                    True,
                ),
            ),
        )

        rendered = format_validation_report(report)

        self.assertIn("Stress validation: PASS", rendered)
        self.assertIn("reference                          : Premia", rendered)
        self.assertIn("pricing method                     : CF_Call", rendered)
        self.assertNotIn("criterion", rendered)
        self.assertIn("mean absolute price gap            : 1.000000e-04", rendered)
        self.assertIn(
            "maximum absolute price gap         : 2.000000e-04 (row 000001)",
            rendered,
        )
        self.assertIn("000002\n  diagnostic : backend failure", rendered)
        self.assertIn("QuantLib (Monte Carlo) fallback for 000002: PASS", rendered)
        self.assertIn(
            "pricing method : _antithetic_path_price", rendered
        )

    def test_json_round_trip_and_dataset_fingerprint(self) -> None:
        section = ValidationDisplayReport(
            title="Core validation",
            status="passed",
            database_id="sample",
            reference="QuantLib (specialized pricer)",
            pricing_method="BlackCalculator",
            tolerance="absolute tolerance",
            row_count=1,
            accepted_row_count=1,
            failed_row_count=0,
            higher_price_count=0,
            lower_price_count=0,
            equal_price_count=1,
            mean_signed_price_gap=0.0,
            mean_absolute_price_gap=0.0,
            maximum_absolute_price_gap=0.0,
            maximum_absolute_price_gap_row_id="000001",
            systematic_bias=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "validation_report.json"
            dataset_path, yaml_path = _write_price_fixture(root)
            report = DatasetValidationReport(
                validation_fingerprint(dataset_path), section, section
            )

            write_validation_report(report, report_path)

            self.assertEqual(
                load_validation_report(report_path, dataset_path), report
            )
            original_fingerprint = validation_fingerprint(dataset_path)
            document = json.loads(dataset_path.read_text(encoding="utf-8"))
            document["timing"]["wall_seconds"] = 9.0
            dataset_path.write_text(
                json.dumps(document, indent=2) + "\n", encoding="utf-8"
            )
            self.assertEqual(validation_fingerprint(dataset_path), original_fingerprint)
            yaml_path.write_text(
                yaml_path.read_text(encoding="utf-8").replace("1 / 360", "1 / 365"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "stale"):
                load_validation_report(report_path, dataset_path)

    def test_yaml_is_synchronized_from_report_without_engine_plan(self) -> None:
        section = ValidationDisplayReport(
            title="Core validation",
            status="passed",
            database_id="sample",
            reference="Premia (specialized pricer)",
            pricing_method="CF_Call",
            tolerance="absolute tolerance",
            row_count=1,
            accepted_row_count=1,
            failed_row_count=0,
            higher_price_count=0,
            lower_price_count=0,
            equal_price_count=1,
            mean_signed_price_gap=0.0,
            mean_absolute_price_gap=0.0,
            maximum_absolute_price_gap=0.0,
            maximum_absolute_price_gap_row_id="000001",
            systematic_bias=False,
            engine_plan=(
                EnginePlanEntry(
                    "Premia (specialized pricer)", "CF_Call", True
                ),
                EnginePlanEntry(
                    "QuantLib (specialized pricer)",
                    None,
                    False,
                    "no compatible engine",
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path, yaml_path = _write_price_fixture(root)
            report = DatasetValidationReport(
                validation_fingerprint(dataset_path), section, section
            )

            synchronize_validation_yaml(report, dataset_path)
            yaml_text = yaml_path.read_text(encoding="utf-8")

            self.assertIn('reference: "Premia (specialized pricer)"', yaml_text)
            self.assertIn(
                'notebook: "catalog/model/equity/sample/prices/'
                'sample_01/validation.ipynb"',
                yaml_text,
            )
            self.assertIn('status: "passed"', yaml_text)
            validation_block = yaml_text.split("validation:\n", 1)[1].split(
                "time_grid:", 1
            )[0]
            self.assertNotIn("method:", validation_block)
            self.assertNotIn("relationship:", yaml_text)
            self.assertNotIn("engine_plan:", yaml_text)

    def test_public_status_depends_only_on_the_core_certification(self) -> None:
        core = ValidationDisplayReport(
            title="Core validation",
            status="passed",
            database_id="sample",
            reference="Premia (specialized pricer)",
            pricing_method="CF_Call",
            tolerance="absolute tolerance",
            row_count=1,
            accepted_row_count=1,
            failed_row_count=0,
            higher_price_count=0,
            lower_price_count=0,
            equal_price_count=1,
            mean_signed_price_gap=0.0,
            mean_absolute_price_gap=0.0,
            maximum_absolute_price_gap=0.0,
            maximum_absolute_price_gap_row_id="000001",
            systematic_bias=False,
        )
        stress = replace(
            core,
            title="Stress validation",
            status="failed",
            accepted_row_count=0,
            failed_row_count=1,
            failed_row_ids=("000001",),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path, yaml_path = _write_price_fixture(root)
            report = DatasetValidationReport(
                validation_fingerprint(dataset_path), core, stress
            )

            self.assertTrue(report.passed)
            synchronize_validation_yaml(report, dataset_path)
            yaml_text = yaml_path.read_text(encoding="utf-8")
            self.assertIn('status: "passed"', yaml_text)
            self.assertIn("verified: true", yaml_text)
            self.assertIn('reference: "Premia (specialized pricer)"', yaml_text)
            self.assertIn('stress:\n    status: "failed"', yaml_text)

    def test_unavailable_validation_has_a_dedicated_display(self) -> None:
        report = ValidationDisplayReport(
            title="Core validation",
            status="not_available",
            database_id="sample",
            reference="none",
            pricing_method=None,
            tolerance="none",
            row_count=900,
            accepted_row_count=0,
            failed_row_count=0,
            higher_price_count=0,
            lower_price_count=0,
            equal_price_count=0,
            mean_signed_price_gap=None,
            mean_absolute_price_gap=None,
            maximum_absolute_price_gap=None,
            maximum_absolute_price_gap_row_id=None,
            systematic_bias=False,
            unvalidated_row_count=900,
            no_validation_reason=(
                "no independent validation is available via Premia or QuantLib"
            ),
        )

        rendered = format_validation_report(report)

        self.assertIn("Core validation: NOT AVAILABLE", rendered)
        self.assertIn("reference                          : none", rendered)
        self.assertIn("unvalidated rows                   : 900", rendered)
        self.assertIn("no independent validation is available", rendered)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "validation_report.json"
            dataset_path, _ = _write_price_fixture(root)
            dataset_report = DatasetValidationReport(
                validation_fingerprint(dataset_path), report, report
            )
            write_validation_report(dataset_report, report_path)
            self.assertEqual(
                load_validation_report(report_path, dataset_path), dataset_report
            )


if __name__ == "__main__":
    unittest.main()
