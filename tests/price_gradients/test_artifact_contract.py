"""Checks for the compact price-gradient dataset and its frozen recipe."""

import copy
import unittest

from tools.datasets.dataset_provenance import fingerprint
from tools.datasets.price_gradients.contract import check_outputs


class ArtifactContractTests(unittest.TestCase):
    def setUp(self):
        self.selection = {
            "method": "finite_difference_shared_innovations",
            "source_price_recipe": "source",
            "parameters": [{
                "parameter": "model.rho", "displacement": 0.125,
                "scale": "absolute", "boundary": "central_then_one_sided_order2",
            }],
        }
        self.grid = {
            "steps_per_year": 504, "simulation_steps_per_day": 2,
            "delta_t": "1 / 504",
        }
        self.plan = {
            "paths_per_price": 32, "threads_per_block": 128,
            "sensitivity_count": 1, "scenario_count": None,
            "materialized_scenario_count": 0,
            "gradient_layout": "row_major_selection_order",
            "sensitivity_batch_size": 1,
            "sensitivity_batches_per_price": 1,
            "maximum_live_scenarios": 3,
            "price_moment_batches_per_price": 1,
            "central_work_policy": "sensitivity_zero_owns_price",
            "kernel_launches_per_price_batch": 1,
            "sensitivity_block_count": 1,
            "work_distribution": "grid (row, sensitivity); one sensitivity per block",
        }
        self.job = {
            "sensitivity": self.selection, "time_key": "time_grid",
            "time_configuration": self.grid, "launch_plan": self.plan,
            "rng_stream_seeds": {"dynamics": 719},
        }
        self.catalog = {
            "sensitivity": self.selection, "time_grid": self.grid,
            "summary": {**self.plan, "seed": 719},
        }
        self.document = {
            "database_id": "test_gradients", "model_family": "Test model",
            "product_family": "Test product",
            "catalog": "catalog/model/test_gradients", "row_count": 1,
            "model_dataset": {
                "id": "test_model", "catalog": "catalog/model/test_model",
            },
            "product_dataset": {
                "id": "test_product", "catalog": "catalog/product/test_product",
            },
            "results": [{
                "model_id": "000001", "product_id": "000002",
                "price": 0.1, "gradients": {"model.rho": 0.2},
                "standard_error": 0.01,
                "gradient_standard_errors": {"model.rho": 0.02},
            }],
        }

    def test_compact_contract_and_option_side(self):
        check_outputs(self.job, self.catalog, self.document)
        sided_job = self.job
        sided_catalog = {**self.catalog, "option_side": "call"}
        sided_document = {**self.document, "option_side": "call"}
        check_outputs(sided_job, sided_catalog, sided_document)
        with self.assertRaises(ValueError):
            check_outputs(self.job, self.catalog, sided_document)
        for side in (None, "put", "unknown"):
            changed = dict(sided_document)
            if side is None:
                changed.pop("option_side")
            else:
                changed["option_side"] = side
            with self.subTest(side=side), self.assertRaises(ValueError):
                check_outputs(sided_job, sided_catalog, changed)

    def test_curve_and_sampling_error_columns(self):
        curve_job = {**self.job, "declared_method": {"has_curve": True}}
        with self.assertRaises(ValueError):
            check_outputs(curve_job, self.catalog, self.document)
        document = copy.deepcopy(self.document)
        document["curve_family"] = "Test curve"
        document["curve_dataset"] = {
            "id": "test_curve", "catalog": "catalog/curve/test_curve",
        }
        document["results"][0]["curve_id"] = "000003"
        check_outputs(curve_job, self.catalog, document)
        document["results"][0]["gradient_standard_errors"]["model.rho"] = -1
        with self.assertRaises(ValueError):
            check_outputs(curve_job, self.catalog, document)

    def test_rejects_legacy_rows_and_execution_in_json(self):
        for extra in (
            {"summary": self.catalog["summary"]},
            {"sensitivity": self.selection},
        ):
            with self.subTest(extra=next(iter(extra))), self.assertRaises(ValueError):
                check_outputs(self.job, self.catalog, {**self.document, **extra})
        document = copy.deepcopy(self.document)
        document["results"][0] = {
            "model_id": "000001", "product_id": "000002",
            "outputs": {"price": 0.1},
        }
        with self.assertRaises(ValueError):
            check_outputs(self.job, self.catalog, document)

    def test_frozen_sensitivity_and_execution_are_checked(self):
        for key, value in (
            ("seed", 720),
            ("sensitivity_count", 2),
            ("price_moment_batches_per_price", 2),
            ("central_work_policy", "always_replayed"),
        ):
            catalog = copy.deepcopy(self.catalog)
            catalog["summary"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_outputs(self.job, catalog, self.document)
        catalog = copy.deepcopy(self.catalog)
        catalog["sensitivity"]["parameters"][0]["displacement"] *= 2
        with self.assertRaises(ValueError):
            check_outputs(self.job, catalog, self.document)

    def test_selected_second_order_columns(self):
        job = copy.deepcopy(self.job)
        job["sensitivity"]["orders"] = ["first", "diagonal_second"]
        catalog = copy.deepcopy(self.catalog)
        catalog["sensitivity"] = job["sensitivity"]
        document = copy.deepcopy(self.document)
        document["results"][0]["diagonal_hessians"] = {"model.rho": 0.3}
        document["results"][0]["diagonal_hessian_standard_errors"] = {
            "model.rho": 0.03,
        }
        check_outputs(job, catalog, document)
        del document["results"][0]["diagonal_hessians"]
        with self.assertRaises(ValueError):
            check_outputs(job, catalog, document)

    def test_full_hessian_compact_rows_and_frozen_selection(self):
        job = copy.deepcopy(self.job)
        job["sensitivity"]["parameters"].append({
            "parameter": "model.volatility", "displacement": 0.01,
            "scale": "relative", "boundary": "central_then_one_sided_order2",
        })
        job["sensitivity"]["orders"] = [
            "first", "diagonal_second", "mixed_second",
        ]
        job["sensitivity"]["mixed_second"] = "all"
        job["launch_plan"]["sensitivity_count"] = 2
        catalog = copy.deepcopy(self.catalog)
        catalog["sensitivity"] = copy.deepcopy(job["sensitivity"])
        catalog["summary"]["sensitivity_count"] = 2
        document = copy.deepcopy(self.document)
        row = document["results"][0]
        row["gradients"]["model.volatility"] = 0.4
        row["gradient_standard_errors"]["model.volatility"] = 0.04
        row["diagonal_hessians"] = {
            "model.rho": 0.3, "model.volatility": 0.5,
        }
        row["diagonal_hessian_standard_errors"] = {
            "model.rho": 0.03, "model.volatility": 0.05,
        }
        row["mixed_hessians"] = {"model.rho|model.volatility": 0.6}
        row["mixed_hessian_standard_errors"] = {
            "model.rho|model.volatility": 0.06,
        }
        check_outputs(job, catalog, document)
        for column in ("mixed_hessian_standard_errors", "diagonal_hessians"):
            changed = copy.deepcopy(document)
            del changed["results"][0][column]
            with self.subTest(column=column), self.assertRaises(ValueError):
                check_outputs(job, catalog, changed)
        changed = copy.deepcopy(document)
        changed["results"][0]["mixed_hessians"] = {
            "model.volatility|model.rho": 0.6,
        }
        with self.assertRaises(ValueError):
            check_outputs(job, catalog, changed)
        changed_recipe = copy.deepcopy(job)
        changed_recipe["sensitivity"]["mixed_second"] = [
            {"first": "model.rho", "second": "model.volatility"},
        ]
        with self.assertRaises(ValueError):
            check_outputs(changed_recipe, catalog, document)

    def test_selection_changes_semantic_identity(self):
        job = {
            **self.job, "kind": "price_gradients",
            "identity": "heston/european_option",
            "dataset": "test.json", "rows": 1, "sample_shape": None,
        }
        changed = copy.deepcopy(job)
        changed["sensitivity"]["parameters"][0]["displacement"] *= 2
        self.assertNotEqual(fingerprint(job), fingerprint(changed))


if __name__ == "__main__":
    unittest.main()
