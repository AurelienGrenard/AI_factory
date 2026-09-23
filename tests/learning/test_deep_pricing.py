"""Check generic price joins, nested selections and derivative-aware losses."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from learning.common.data.cache import prepare_pricing_dataset
from learning.common.data.selection import make_dataset_selection
from learning.common.data.transforms import Standardization
from learning.common.training import LossContext, build_composite_loss, register_loss
from learning.deep_pricing.compare import compare
from learning.deep_pricing.campaign import expand_runs
from learning.deep_pricing.engine import train_experiment
from learning.deep_pricing.representations import UnitSpotHomogeneousModel


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _pricing_fixture(root: Path, *, repetitions: int = 1) -> Path:
    (root / "catalog").mkdir()
    model_path = root / "datasets/model/test/parameters/model_01.json"
    product_path = root / "datasets/product/test/products_01.json"
    _write_json(
        model_path,
        {
            "database_id": "model_01",
            "models": [
                {"id": "m1", "parameters": {"spot": 1.0, "volatility": 0.2}},
                {"id": "m2", "parameters": {"spot": 1.0, "volatility": 0.3}},
            ],
        },
    )
    _write_json(
        product_path,
        {
            "database_id": "products_01",
            "products": [
                {"id": "p1", "parameters": {"strike": 0.9, "maturity": 21}},
                {"id": "p2", "parameters": {"strike": 1.1, "maturity": 252}},
            ],
        },
    )
    results = []
    pairs = (("m1", "p1"), ("m1", "p2"), ("m2", "p1"), ("m2", "p2"))
    for index, (model_id, product_id) in enumerate(pairs * repetitions):
        results.append(
            {
                "id": str(index),
                "model_id": model_id,
                "product_id": product_id,
                "outputs": {
                    "price": 0.1 + index,
                    "delta": 0.5 + index,
                    "standard_error": 0.01,
                    "delta_standard_error": 0.02,
                },
            }
        )
    dataset_path = root / "datasets/model/test/price_delta/product/prices.json"
    _write_json(
        dataset_path,
        {
            "database_id": "test_prices",
            "row_count": len(results),
            "model_dataset": {
                "id": "model_01",
                "catalog": "catalog/model/test/parameters/model_01",
            },
            "product_dataset": {
                "id": "products_01",
                "catalog": "catalog/product/test/products_01",
            },
            "sensitivity": {"parameter": "spot"},
            "results": results,
        },
    )
    return dataset_path


class PricingDataTest(unittest.TestCase):
    def test_streams_joins_and_caches_price_delta(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _pricing_fixture(root)
            prepared = prepare_pricing_dataset(dataset, cache_root=root / "cache")
            reopened = prepare_pricing_dataset(dataset, cache_root=root / "cache")

            self.assertEqual(prepared.fingerprint, reopened.fingerprint)
            self.assertEqual(
                prepared.schema.feature_names,
                ("model.spot", "model.volatility", "product.maturity", "product.strike"),
            )
            self.assertEqual(prepared.schema.gradients[0].wrt_name, "model.spot")
            np.testing.assert_allclose(prepared.features[3], [1.0, 0.3, 252.0, 1.1])
            np.testing.assert_allclose(prepared.values[:, 0], [0.1, 1.1, 2.1, 3.1])
            np.testing.assert_allclose(prepared.gradients[:, 0], [0.5, 1.5, 2.5, 3.5])

    def test_streams_multiple_selected_price_gradients(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _pricing_fixture(root)
            document = json.loads(dataset.read_text())
            document["sensitivity"] = {
                "method": "finite_difference_shared_innovations",
                "parameters": [
                    {"parameter": "model.volatility"},
                    {"parameter": "product.strike"},
                ],
            }
            for index, row in enumerate(document["results"]):
                outputs = row["outputs"]
                outputs.pop("delta")
                outputs.pop("delta_standard_error")
                outputs["gradients"] = {
                    "model.volatility": 1.0 + index,
                    "product.strike": -2.0 - index,
                }
                outputs["gradient_standard_errors"] = {
                    "model.volatility": 0.03,
                    "product.strike": 0.04,
                }
            _write_json(dataset, document)

            prepared = prepare_pricing_dataset(dataset, cache_root=root / "cache")

            self.assertEqual(
                tuple(target.wrt_name for target in prepared.schema.gradients),
                ("model.volatility", "product.strike"),
            )
            np.testing.assert_allclose(
                prepared.gradients,
                [[1.0, -2.0], [2.0, -3.0], [3.0, -4.0], [4.0, -5.0]],
            )
            np.testing.assert_allclose(
                prepared.gradient_standard_errors,
                [[0.03, 0.04]] * 4,
            )

    def test_rejects_ambiguous_sensitivity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _pricing_fixture(root)
            product = root / "datasets/product/test/products_01.json"
            value = json.loads(product.read_text())
            for row in value["products"]:
                row["parameters"]["spot"] = 1.0
            _write_json(product, value)
            with self.assertRaisesRegex(ValueError, "resolves to 2 features"):
                prepare_pricing_dataset(dataset, cache_root=root / "cache")


class SelectionAndLossTest(unittest.TestCase):
    def test_homogeneous_wrapper_reconstructs_price_and_spot_delta(self) -> None:
        class SquareMoneyness(torch.nn.Module):
            def forward(self, inputs):
                return torch.square(inputs[:, 0:1])

        identity = Standardization(
            feature_mean=np.zeros(2, dtype=np.float32),
            feature_scale=np.ones(2, dtype=np.float32),
            value_mean=np.zeros(1, dtype=np.float32),
            value_scale=np.ones(1, dtype=np.float32),
        )
        model = UnitSpotHomogeneousModel(
            SquareMoneyness(),
            spot_index=0,
            strike_index=1,
            transform=identity,
        )
        inputs = torch.tensor([[1.0, 2.0]], requires_grad=True)
        price = model(inputs)
        delta = torch.autograd.grad(price.sum(), inputs)[0][:, 0:1]
        torch.testing.assert_close(price, torch.tensor([[4.0]]))
        torch.testing.assert_close(delta, torch.tensor([[-4.0]]))

    def test_nested_train_samples_keep_held_out_rows_fixed(self) -> None:
        small = make_dataset_selection(10_000, train_size=100, selection_seed=7)
        large = make_dataset_selection(10_000, train_size=500, selection_seed=7)
        self.assertTrue(set(small.train_indices).issubset(set(large.train_indices)))
        np.testing.assert_array_equal(small.validation_indices, large.validation_indices)
        np.testing.assert_array_equal(small.test_indices, large.test_indices)
        self.assertEqual(
            small.manifest()["digests"]["test"], large.manifest()["digests"]["test"]
        )

    def test_grouped_split_holds_out_complete_exact_surfaces(self) -> None:
        groups = np.repeat(np.arange(1000), 1000)
        selection = make_dataset_selection(
            1_000_000,
            train_size=30_000,
            group_ids=groups,
            validation_group_count=100,
            test_group_count=100,
            core_group_count=900,
        )
        self.assertEqual(selection.train_candidate_count, 800_000)
        self.assertEqual(selection.validation_indices.size, 100_000)
        self.assertEqual(selection.test_indices.size, 100_000)
        self.assertEqual(selection.group_manifest["train"], 800)
        self.assertEqual(selection.group_manifest["validation"], 100)
        self.assertEqual(selection.group_manifest["test"], 100)
        self.assertEqual(np.unique(groups[selection.test_indices]).size, 100)
        self.assertFalse(
            set(groups[selection.train_indices]) & set(groups[selection.test_indices])
        )

    def test_gradient_loss_uses_normalized_derivative_units(self) -> None:
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        inputs = torch.tensor([[0.0], [1.0]], requires_grad=True)
        predictions = model(inputs)
        predicted = torch.autograd.grad(predictions.sum(), inputs, create_graph=True)[0]
        transform = Standardization(
            feature_mean=np.array([10.0], dtype=np.float32),
            feature_scale=np.array([2.0], dtype=np.float32),
            value_mean=np.array([3.0], dtype=np.float32),
            value_scale=np.array([4.0], dtype=np.float32),
        )
        raw_delta = torch.full((2, 1), 2.0)
        normalized_delta = raw_delta * float(transform.normalized_gradient_scales(np.array([0]))[0])
        loss = build_composite_loss([{"name": "gradient_mse", "weight": 1.0}])
        value, components = loss(
            LossContext(
                model=model,
                normalized_inputs=inputs,
                normalized_predictions=predictions,
                normalized_targets=predictions.detach(),
                normalized_predicted_gradients=predicted,
                normalized_target_gradients=normalized_delta,
                raw_inputs=inputs,
                raw_predictions=predictions,
                raw_targets=predictions.detach(),
                raw_predicted_gradients=predicted,
                raw_target_gradients=raw_delta,
                batch={},
                epoch=1,
                global_step=0,
            )
        )
        self.assertEqual(float(value.detach()), 0.0)
        self.assertEqual(components["gradient_mse"], 0.0)

    def test_custom_penalty_can_be_registered_without_changing_trainer(self) -> None:
        @register_loss("test_constant_penalty")
        def build_penalty(specification):
            class Penalty:
                requires_input_gradients = False

                def __call__(self, context):
                    return context.normalized_predictions.new_tensor(
                        float(specification["constant"])
                    )

            return Penalty()

        loss = build_composite_loss(
            [{"name": "test_constant_penalty", "constant": 3.0, "weight": 2.0}]
        )
        model = torch.nn.Linear(1, 1)
        prediction = model(torch.ones(1, 1))
        value, _ = loss(LossContext(
            model=model,
            normalized_inputs=prediction,
            normalized_predictions=prediction,
            normalized_targets=prediction,
            normalized_predicted_gradients=None,
            normalized_target_gradients=None,
            raw_inputs=prediction,
            raw_predictions=prediction,
            raw_targets=prediction,
            raw_predicted_gradients=None,
            raw_target_gradients=None,
            batch={},
            epoch=1,
            global_step=0,
        ))
        self.assertEqual(float(value.detach()), 6.0)

    def test_run_comparison_requires_identical_test_rows(self) -> None:
        base = {
            "dataset_fingerprint": "same",
            "selection": {
                "counts": {"train": 10},
                "digests": {"validation": "v", "test": "t"},
            },
            "metrics": {"test": {"price_mae": 2.0}},
        }
        other = json.loads(json.dumps(base))
        other["metrics"]["test"]["price_mae"] = 1.5
        result = compare(base, other)
        self.assertEqual(result["test_metric_right_minus_left"]["price_mae"], -0.5)
        other["selection"]["digests"]["test"] = "different"
        with self.assertRaisesRegex(ValueError, "different test rows"):
            compare(base, other)

    def test_tiny_sobolev_experiment_writes_a_reproducible_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _pricing_fixture(root, repetitions=50)
            prepared = prepare_pricing_dataset(dataset, cache_root=root / "cache")
            selection = make_dataset_selection(200, train_size=100)
            config = {
                "dataset": str(dataset),
                "method": "sobolev",
                "data": {},
                "network": {
                    "name": "mlp",
                    "hidden_sizes": [8],
                    "activation": "tanh",
                },
                "loss": {
                    "terms": [
                        {"name": "value_mse", "weight": 1.0},
                        {"name": "gradient_mse", "weight": 0.1},
                    ]
                },
                "training": {
                    "seed": 11,
                    "epochs": 1,
                    "batch_size": 32,
                    "learning_rate": 0.001,
                    "num_workers": 0,
                    "device": "cpu",
                },
            }
            result = train_experiment(prepared, selection, config, root / "run")
            self.assertTrue((result.run_directory / "initial_model.pt").is_file())
            self.assertTrue((result.run_directory / "last_model.pt").is_file())
            self.assertTrue((result.run_directory / "checkpoint.pt").is_file())
            self.assertTrue((result.run_directory / "run.json").is_file())
            self.assertEqual(result.manifest["selection"]["counts"]["train"], 100)
            self.assertIn("gradient_mae", result.manifest["metrics"]["test"])
            resumed = train_experiment(
                prepared, selection, config, root / "run", resume=True
            )
            self.assertEqual(
                resumed.manifest["initial_model_hash"],
                result.manifest["initial_model_hash"],
            )

    def test_campaign_expands_one_identical_triplet_per_configuration(self) -> None:
        runs = expand_runs(
            {
                "dataset": "prices.json",
                "architectures": [
                    {"id": "small", "network": {"name": "mlp", "hidden_sizes": [8]}}
                ],
                "train_sizes": [30_000],
                "seeds": [17],
                "gradient_weights": [0.1, 1.0],
                "data": {},
                "training": {"epochs": 2},
            }
        )
        self.assertEqual(len(runs), 3)
        self.assertEqual({run.pair_id for run in runs}, {"small__n30000__seed17"})
        self.assertEqual(
            [run.gradient_weight for run in runs], [None, 0.1, 1.0]
        )
        self.assertTrue(all(run.config["network"] == runs[0].config["network"] for run in runs))
        self.assertTrue(all(run.config["training"] == runs[0].config["training"] for run in runs))

    def test_campaign_pairs_each_explicit_representation_independently(self) -> None:
        runs = expand_runs(
            {
                "dataset": "prices.json",
                "representations": [
                    {"id": "raw", "name": "identity"},
                    {
                        "id": "homogeneous",
                        "name": "unit_spot_homogeneous",
                        "spot_feature": "model.spot",
                        "strike_feature": "product.strike",
                    },
                ],
                "architectures": [
                    {"id": "small", "network": {"name": "mlp", "hidden_sizes": [8]}}
                ],
                "train_sizes": [30_000],
                "seeds": [17],
                "gradient_weights": [0.1],
            }
        )
        self.assertEqual(len(runs), 4)
        self.assertEqual(
            {run.pair_id for run in runs},
            {
                "raw__small__n30000__seed17",
                "homogeneous__small__n30000__seed17",
            },
        )
        self.assertEqual(
            {run.representation_id for run in runs}, {"raw", "homogeneous"}
        )
        self.assertEqual(
            runs[2].config["representation"]["name"], "unit_spot_homogeneous"
        )


if __name__ == "__main__":
    unittest.main()
