"""Check generic catalog selection and targeted compilation."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.datasets import generate_catalog as campaign


def spec(
    name: str,
    *,
    family: str = "markovian",
    construction: str = "cartesian_parameter_paths",
) -> SimpleNamespace:
    return SimpleNamespace(
        cmake_target=f"generate_{name}_samples_01",
        dataset_kind="samples",
        asset_class="equity",
        source_prefix=f"model/equity/{family}/{name}",
        model=name,
        construction=construction,
        dataset_path=f"datasets/model/equity/{family}/{name}/samples/samples_01.json",
        generation_yaml_path=f"catalog/model/equity/{family}/{name}/samples/samples_01/generation.yaml",
    )


class CatalogSelectionTest(unittest.TestCase):
    def select(self, root: Path, specs, **filters):
        return campaign.select_specs(
            root,
            specs,
            {"samples"},
            set(),
            set(),
            **filters,
        )

    def test_skips_complete_pairs_and_keeps_manifest_filters(self) -> None:
        existing = spec("existing")
        missing = spec("missing", family="rough")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in (existing.dataset_path, existing.generation_yaml_path):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("published")
            self.assertEqual(
                self.select(root, (existing, missing), skip_published=True),
                [missing],
            )
            self.assertEqual(
                self.select(root, (existing, missing), model_families={"rough"}),
                [missing],
            )
            self.assertEqual(
                self.select(root, (existing, missing), asset_classes={"equity"}),
                [existing, missing],
            )

    def test_rejects_incomplete_publication_pair(self) -> None:
        item = spec("incomplete")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / item.dataset_path
            path.parent.mkdir(parents=True)
            path.write_text("partial")
            with self.assertRaisesRegex(ValueError, "Partial published pair"):
                self.select(root, (item,), skip_published=True)

    def test_rejects_target_outside_other_filters(self) -> None:
        known = spec("known")
        with self.assertRaisesRegex(ValueError, "Targets outside the selection"):
            campaign.select_specs(
                Path("."),
                (known,),
                {"samples"},
                set(),
                {"generate_known_samples_01"},
                model_families={"rough"},
            )

    def test_compiles_selected_targets_and_price_inspector(self) -> None:
        jobs = [
            {"target": "generate_z", "kind": "samples"},
            {"target": "generate_a", "kind": "price_delta"},
        ]
        with patch.object(campaign.subprocess, "run") as run:
            campaign.compile_selected(Path("/repo"), Path("/repo/build"), jobs, 2)
        command = run.call_args.args[0]
        self.assertEqual(
            command[4:-1],
            ["generate_a", "generate_z", "inspect_pricing_launch_plan"],
        )
        self.assertEqual(command[-1], "-j2")
        self.assertTrue(run.call_args.kwargs["check"])


if __name__ == "__main__":
    unittest.main()
