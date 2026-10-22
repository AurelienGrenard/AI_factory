"""Small content checks for compact price and sensitivity datasets."""

import copy
import unittest

from tools.datasets.price_gradients.contract import check_outputs


class ArtifactContractTests(unittest.TestCase):
    def setUp(self):
        sensitivity = {
            "method": "finite_difference_shared_innovations",
            "source_price_recipe": "source",
            "parameters": [{
                "parameter": "model.rho", "displacement": 0.125,
                "scale": "absolute", "boundary": "central_then_one_sided_order2",
            }],
        }
        grid = {"steps_per_year": 504}
        plan = {"paths_per_price": 32, "sensitivity_count": 1}
        self.job = {
            "sensitivity": sensitivity, "time_key": "time_grid",
            "time_configuration": grid, "launch_plan": plan,
            "rng_stream_seeds": {"dynamics": 719},
        }
        self.catalog = {
            "sensitivity": sensitivity, "time_grid": grid,
            "summary": {**plan, "seed": 719},
        }
        self.document = {
            "database_id": "test_gradients", "model_family": "Test model",
            "product_family": "Test product", "row_count": 1,
            "model_dataset": {"id": "test_model"},
            "product_dataset": {"id": "test_product"},
            "results": [{
                "model_id": "000001", "product_id": "000002",
                "price": 0.1, "standard_error": 0.01,
                "gradients": {"model.rho": 0.2},
                "gradient_standard_errors": {"model.rho": 0.02},
            }],
        }

    def test_current_compact_output_and_quote_metadata(self):
        document = copy.deepcopy(self.document)
        document["quote_convention"] = {"strike": {"unit": "relative"}}
        check_outputs(self.job, self.catalog, document)
        # Presentation metadata may evolve without changing numerical rows.
        document["description"] = "extra context"
        document["model_dataset"]["label"] = "model"
        check_outputs(self.job, self.catalog, document)

    def test_curve_and_option_side(self):
        job = {**self.job, "declared_method": {"has_curve": True}}
        catalog = {**self.catalog, "option_side": "call"}
        document = copy.deepcopy(self.document)
        document.update(curve_family="Test curve", curve_dataset={"id": "curve"},
                        option_side="call")
        document["results"][0]["curve_id"] = "000003"
        check_outputs(job, catalog, document)
        document["option_side"] = "put"
        with self.assertRaises(ValueError):
            check_outputs(job, catalog, document)

    def test_rejects_missing_or_invalid_numerical_data(self):
        for mutation in ("missing_gradient", "nonfinite_price", "negative_error",
                         "wrong_row_count", "missing_input_id"):
            document = copy.deepcopy(self.document)
            row = document["results"][0]
            if mutation == "missing_gradient":
                row["gradients"].clear()
            elif mutation == "nonfinite_price":
                row["price"] = float("nan")
            elif mutation == "negative_error":
                row["gradient_standard_errors"]["model.rho"] = -1.0
            elif mutation == "wrong_row_count":
                document["row_count"] = 2
            else:
                document["model_dataset"] = {}
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                check_outputs(self.job, self.catalog, document)

    def test_second_order_columns_follow_requested_orders(self):
        job = copy.deepcopy(self.job)
        job["sensitivity"]["orders"] = ["first", "diagonal_second"]
        catalog = copy.deepcopy(self.catalog)
        catalog["sensitivity"] = job["sensitivity"]
        document = copy.deepcopy(self.document)
        row = document["results"][0]
        row["diagonal_hessians"] = {"model.rho": 0.3}
        row["diagonal_hessian_standard_errors"] = {"model.rho": 0.03}
        check_outputs(job, catalog, document)
        del row["diagonal_hessians"]
        with self.assertRaises(ValueError):
            check_outputs(job, catalog, document)


if __name__ == "__main__":
    unittest.main()
