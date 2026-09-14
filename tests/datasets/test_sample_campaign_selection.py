"""Check sample-campaign selection, especially existing-output protection."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.datasets import generate_sample_campaign as campaign
from tools.datasets.watch_generation_progress import format_progress


def spec(name: str, *, family: str = "markovian") -> SimpleNamespace:
    return SimpleNamespace(
        cmake_target=f"generate_{name}_samples_01",
        dataset_kind="samples",
        asset_class="equity",
        source_prefix=f"model/equity/{family}/{name}",
        model=name,
        dataset_path=f"datasets/model/equity/{family}/{name}/samples/samples_01.json",
        catalog_yaml_path=f"catalog/model/equity/{family}/{name}/samples/samples_01/dataset.yaml",
    )


class SampleCampaignSelectionTest(unittest.TestCase):
    def test_skips_existing_pair_and_selects_missing(self) -> None:
        existing, missing = spec("existing"), spec("missing", family="rough")
        with tempfile.TemporaryDirectory() as directory, patch.object(
            campaign, "AVAILABLE_DATASET_SPECS", (existing, missing)
        ):
            root = Path(directory)
            for relative in (existing.dataset_path, existing.catalog_yaml_path):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("published")
            selected, published = campaign.select_specs(root=root)
            self.assertEqual(selected, [missing])
            self.assertEqual(published, [existing])
            selected, _ = campaign.select_specs(root=root, include_published=True)
            self.assertEqual({item.model for item in selected}, {"existing", "missing"})
            selected, _ = campaign.select_specs(root=root, model_family="rough")
            self.assertEqual(selected, [missing])

    def test_rejects_incomplete_publication_pair(self) -> None:
        item = spec("incomplete")
        with tempfile.TemporaryDirectory() as directory, patch.object(
            campaign, "AVAILABLE_DATASET_SPECS", (item,)
        ):
            root = Path(directory)
            path = root / item.dataset_path
            path.parent.mkdir(parents=True)
            path.write_text("partial")
            with self.assertRaisesRegex(ValueError, "Partial published pair"):
                campaign.select_specs(root=root)

    def test_rejects_unknown_target(self) -> None:
        with patch.object(campaign, "AVAILABLE_DATASET_SPECS", (spec("known"),)):
            with self.assertRaisesRegex(ValueError, "Targets outside"):
                campaign.select_specs(targets={"generate_unknown_samples_01"})

    def test_sample_progress_uses_sample_units_and_writing_phase(self) -> None:
        job = {"kind": "samples", "state": "running", "rows": 3_000_000,
               "dataset": "datasets/model/x/samples/samples_01.json"}
        waiting = format_progress(job, {}, 100.0)
        self.assertIn("en attente de la phase d'écriture", waiting)
        self.assertNotIn("Prix calculés", waiting)
        writing = format_progress(job, {
            "state": "running", "completed_samples": 1_500_000,
            "total_samples": 3_000_000, "samples_per_second": 100_000.0,
            "elapsed_seconds": 15.0, "estimated_seconds_remaining": 15.0,
        }, 100.0)
        self.assertIn("Échantillons écrits : 1 500 000 / 3 000 000", writing)
        self.assertIn("50,00 % de l'écriture", writing)
        self.assertIn("Temps restant estimé (écriture JSON)", writing)


if __name__ == "__main__":
    unittest.main()
