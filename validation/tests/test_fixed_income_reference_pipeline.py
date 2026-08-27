"""Cross-family publication tests for fixed-income reference caches."""

from __future__ import annotations

import builtins
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import yaml

from validation.model.fixed_income.reference_pipeline import (
    validate_cached_reference,
)
from validation.quantlib.price_validation import ValidationTolerances
from validation.reference_price_dataset import validation_policy_fingerprint


ROOT = Path(__file__).resolve().parents[2]


class FixedIncomeReferencePipelineTest(unittest.TestCase):
    @staticmethod
    def _sources() -> tuple[Path, ...]:
        return tuple(
            sorted(
                path
                for path in (ROOT / "datasets/model/fixed_income").rglob(
                    "*.json"
                )
                if "prices" in path.parts
                and not any(
                    part.startswith("bermudan_") for part in path.parts
                )
            )
        )

    @staticmethod
    def _bermudan_sources() -> tuple[Path, ...]:
        return tuple(
            sorted(
                path
                for path in (ROOT / "datasets/model/fixed_income").rglob(
                    "*.json"
                )
                if "prices" in path.parts
                and any(part.startswith("bermudan_") for part in path.parts)
            )
        )

    @staticmethod
    def _reference(source: Path) -> Path:
        relative = source.relative_to(ROOT / "datasets/model/fixed_income")
        model_name, marker, *price_path = relative.parts
        if marker != "prices":
            raise ValueError(f"Unexpected price path: {source}")
        return (
            ROOT
            / "validation/datasets/price/fixed_income"
            / model_name
            / Path(*price_path)
        )

    @staticmethod
    def _catalog_validation(source: Path) -> dict:
        source_document = json.loads(source.read_text(encoding="utf-8"))
        catalog = ROOT / source_document["catalog"] / "dataset.yaml"
        document = yaml.safe_load(catalog.read_text(encoding="utf-8"))
        return document["validation"]

    def test_all_42_catalogs_publish_only_a_verified_cache(self) -> None:
        sources = self._sources()
        self.assertEqual(len(sources), 42)
        for source in sources:
            source_document = json.loads(source.read_text(encoding="utf-8"))
            reference = self._reference(source)
            report = validate_cached_reference(source, reference)
            self.assertTrue(report.verified, source_document["database_id"])
            reference_document = json.loads(
                reference.read_text(encoding="utf-8")
            )
            self.assertRegex(
                reference_document["validation_policy_fingerprint"],
                r"^sha256:[0-9a-f]{64}$",
            )

            catalog = ROOT / source_document["catalog"]
            yaml_document = yaml.safe_load(
                (catalog / "dataset.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(
                yaml_document["validation"],
                {
                    "status": "available",
                    "verified": True,
                    "dataset": reference.relative_to(ROOT).as_posix(),
                },
            )
            self.assertFalse((catalog / "validation.ipynb").exists())
            self.assertFalse((catalog / "validation_report.json").exists())

    def test_new_bermudan_catalogs_remain_explicitly_pending(self) -> None:
        sources = self._bermudan_sources()
        self.assertEqual(len(sources), 16)
        for source in sources:
            self.assertEqual(
                self._catalog_validation(source),
                {
                    "status": "pending",
                    "verified": False,
                    "reference": "none",
                    "notebook": (
                        json.loads(source.read_text(encoding="utf-8"))[
                            "catalog"
                        ]
                        + "/validation.ipynb"
                    ),
                },
            )

    def test_cache_validation_never_imports_external_pricers(self) -> None:
        """Routine publication checks must work without Premia or QuantLib."""

        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "QuantLib" or name.startswith("validation.premia"):
                raise AssertionError(f"Unexpected external backend import: {name}")
            return original_import(name, *args, **kwargs)

        representatives = (
            self._sources()[0],
            next(path for path in self._sources() if "/g2/" in path.as_posix()),
            next(
                path
                for path in self._sources()
                if "/hull_white/" in path.as_posix()
            ),
        )
        with patch("builtins.__import__", side_effect=guarded_import):
            for source in representatives:
                report = validate_cached_reference(
                    source, self._reference(source)
                )
                self.assertTrue(report.verified)

    def test_policy_fingerprint_commits_the_current_tolerances(self) -> None:
        default = validation_policy_fingerprint(ValidationTolerances())
        changed = validation_policy_fingerprint(
            ValidationTolerances(absolute=1.0e-6)
        )
        self.assertRegex(default, r"^sha256:[0-9a-f]{64}$")
        self.assertNotEqual(default, changed)


if __name__ == "__main__":
    unittest.main()
