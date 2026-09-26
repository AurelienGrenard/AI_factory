"""Mutation checks for gradient metadata, column identity, finite errors and stencil geometry."""
import copy
from pathlib import Path
import sys
import unittest

from tools.datasets.price_gradients.contract import check_outputs
from tools.datasets.dataset_provenance import fingerprint


class ArtifactContractTests(unittest.TestCase):
    def setUp(self):
        self.selection = {"method":"finite_difference_shared_innovations", "source_price_recipe":"source",
            "parameters":[{"parameter":"model.rho", "displacement":.125, "scale":"absolute",
                           "boundary":"central_then_one_sided_order2"}]}
        self.grid = {"steps_per_year":504,"simulation_steps_per_day":2,"delta_t":"1 / 504"}
        self.plan = {"paths_per_price":32,"threads_per_block":128,"sensitivity_count":1,
                     "scenario_count":None,"materialized_scenario_count":0,
                     "gradient_layout":"row_major_selection_order",
                     "sensitivity_batch_size":1,"sensitivity_batches_per_price":1,
                     "maximum_live_scenarios":3,"price_moment_batches_per_price":1,
                     "central_work_policy":"sensitivity_zero_owns_price",
                     "kernel_launches_per_price_batch":1,
                     "sensitivity_block_count":1,
                     "work_distribution":"grid (row, sensitivity); one sensitivity per block"}
        self.job = {"sensitivity":self.selection,"time_key":"time_grid","time_configuration":self.grid,"launch_plan":self.plan,
                    "rng_stream_seeds":{"dynamics":719}}
        self.catalog = {"sensitivity":self.selection,"time_grid":self.grid,"summary":{**self.plan,"seed":719}}
        self.document = {**self.catalog,"results":[{"stencils":{"model.rho":{"central":1.,"first":.875,"second":.75,
            "displacement":.125,"kind":"backward","represented_width":-.125,"first_weight":-16.,"second_weight":4.}},
            "outputs":{"price":.1,"standard_error":.01,"gradients":{"model.rho":.2},"gradient_standard_errors":{"model.rho":.02}}}]}

    def test_valid_one_sided_contract(self):
        check_outputs(self.job,self.catalog,self.document)

    def test_diagonal_contract_checks_fourth_node_and_weights(self):
        job = copy.deepcopy(self.job)
        document = copy.deepcopy(self.document)
        job["sensitivity"]["orders"] = ["first", "diagonal_second"]
        job["launch_plan"]["scenario_count"] = None
        job["launch_plan"]["maximum_live_scenarios"] = 4
        job["launch_plan"]["represented_nodes_per_sensitivity"] = 4
        job["launch_plan"]["requested_orders"] = ["first", "diagonal_second"]
        document["sensitivity"] = copy.deepcopy(job["sensitivity"])
        document["summary"] = {**job["launch_plan"], "seed":719}
        document["results"][0]["outputs"].update(
            diagonal_hessians={"model.rho":.3},
            diagonal_hessian_standard_errors={"model.rho":.03})
        document["results"][0]["stencils"]["model.rho"].update(
            node_count=4, third=.625,
            second_weights=[128.,-320.,256.,-64.])
        check_outputs(job,document,document)
        for mutation in (
            lambda d: d["results"][0]["stencils"]["model.rho"].update(third=.5),
            lambda d: d["results"][0]["stencils"]["model.rho"]["second_weights"].__setitem__(2,512.),
            lambda d: d["results"][0]["outputs"]["diagonal_hessian_standard_errors"].update({"model.rho":-1}),
        ):
            changed = copy.deepcopy(document)
            mutation(changed)
            with self.assertRaises(ValueError):
                check_outputs(job,changed,changed)

    def test_full_hessian_contract_checks_pairs_and_tensor_stencils(self):
        job = copy.deepcopy(self.job)
        second = {
            "parameter": "model.volatility",
            "displacement": 0.25,
            "scale": "absolute",
            "boundary": "central_then_one_sided_order2",
        }
        job["sensitivity"]["parameters"].append(second)
        job["sensitivity"]["orders"] = [
            "first", "diagonal_second", "mixed_second"
        ]
        job["sensitivity"]["mixed_second"] = "all"
        job["launch_plan"].update(
            sensitivity_count=2,
            scenario_count=None,
            materialized_scenario_count=0,
            requested_orders=[
                "first", "diagonal_second", "mixed_second"
            ],
            first_sensitivity_count=2,
            diagonal_hessian_count=2,
            mixed_hessian_count=1,
            sensitivity_graph_node_capacity=11,
        )
        document = copy.deepcopy(self.document)
        document["sensitivity"] = copy.deepcopy(job["sensitivity"])
        document["summary"] = {
            **job["launch_plan"],
            "seed": 719,
        }
        row = document["results"][0]
        row["stencils"]["model.rho"].update(
            node_count=4,
            third=0.625,
            second_weights=[128.0, -320.0, 256.0, -64.0],
        )
        row["stencils"]["model.volatility"] = {
            "central": 2.0,
            "first": 1.75,
            "second": 2.25,
            "displacement": 0.25,
            "kind": "centered",
            "represented_width": 0.5,
            "first_weight": 0.0,
            "second_weight": 0.0,
            "node_count": 3,
            "second_weights": [-32.0, 16.0, 16.0],
        }
        row["outputs"].update(
            gradients={"model.rho": 0.2, "model.volatility": 0.4},
            gradient_standard_errors={
                "model.rho": 0.02,
                "model.volatility": 0.04,
            },
            diagonal_hessians={
                "model.rho": 0.3,
                "model.volatility": 0.5,
            },
            diagonal_hessian_standard_errors={
                "model.rho": 0.03,
                "model.volatility": 0.05,
            },
            mixed_hessians={"model.rho|model.volatility": 0.6},
            mixed_hessian_standard_errors={
                "model.rho|model.volatility": 0.06
            },
        )
        row["mixed_stencils"] = {
            "model.rho|model.volatility": {
                "node_count": 6,
                "first_local_nodes": [0, 0, 1, 1, 2, 2],
                "second_local_nodes": [1, 2, 1, 2, 1, 2],
                "weights": [-24.0, 24.0, 32.0, -32.0, -8.0, 8.0],
            }
        }
        check_outputs(job, document, document)
        for mutation in (
            lambda data: data["results"][0]["outputs"][
                "mixed_hessians"
            ].update({"model.rho|model.volatility": float("nan")}),
            lambda data: data["results"][0]["mixed_stencils"][
                "model.rho|model.volatility"
            ]["weights"].__setitem__(0, -12.0),
            lambda data: data["results"][0]["outputs"][
                "mixed_hessian_standard_errors"
            ].update({"model.rho|model.volatility": -1.0}),
        ):
            changed = copy.deepcopy(document)
            mutation(changed)
            with self.assertRaises(ValueError):
                check_outputs(job, changed, changed)

    def test_sparse_mixed_request_serializes_only_used_coordinates(self):
        parameters = [
            {
                "parameter": "model.rho",
                "displacement": 0.125,
                "scale": "absolute",
                "boundary": "central_then_one_sided_order2",
            },
            {
                "parameter": "model.volatility",
                "displacement": 0.25,
                "scale": "absolute",
                "boundary": "central_then_one_sided_order2",
            },
            {
                "parameter": "product.strike",
                "displacement": 0.5,
                "scale": "absolute",
                "boundary": "central_then_one_sided_order2",
            },
        ]
        sensitivity = {
            "method": "finite_difference_shared_innovations",
            "source_price_recipe": "source",
            "parameters": parameters,
            "orders": ["mixed_second"],
            "mixed_second": [
                {
                    "first": "model.rho",
                    "second": "model.volatility",
                }
            ],
        }
        plan = {
            "paths_per_price": 32,
            "threads_per_block": 128,
            "sensitivity_count": 3,
            "scenario_count": None,
            "materialized_scenario_count": 0,
            "gradient_layout": "row_major_selected_sensitivity_graph",
            "maximum_live_scenarios": 9,
            "represented_nodes_per_sensitivity": 4,
            "requested_orders": ["mixed_second"],
            "sensitivity_graph_node_capacity": 9,
            "first_sensitivity_count": 0,
            "diagonal_hessian_count": 0,
            "mixed_hessian_count": 1,
        }
        job = {
            "sensitivity": sensitivity,
            "time_key": "time_grid",
            "time_configuration": self.grid,
            "launch_plan": plan,
            "rng_stream_seeds": {"dynamics": 719},
        }
        centered_axes = {
            "model.rho": {
                "central": 1.0,
                "first": 0.875,
                "second": 1.125,
                "displacement": 0.125,
                "kind": "centered",
                "represented_width": 0.25,
                "first_weight": 0.0,
                "second_weight": 0.0,
            },
            "model.volatility": {
                "central": 2.0,
                "first": 1.75,
                "second": 2.25,
                "displacement": 0.25,
                "kind": "centered",
                "represented_width": 0.5,
                "first_weight": 0.0,
                "second_weight": 0.0,
            },
        }
        pair = "model.rho|model.volatility"
        document = {
            "sensitivity": copy.deepcopy(sensitivity),
            "time_grid": self.grid,
            "summary": {**plan, "seed": 719},
            "results": [{
                "stencils": centered_axes,
                "mixed_stencils": {
                    pair: {
                        "node_count": 4,
                        "first_local_nodes": [1, 1, 2, 2],
                        "second_local_nodes": [1, 2, 1, 2],
                        "weights": [8.0, -8.0, -8.0, 8.0],
                    }
                },
                "outputs": {
                    "price": 0.1,
                    "standard_error": 0.01,
                    "mixed_hessians": {pair: 0.6},
                    "mixed_hessian_standard_errors": {pair: 0.06},
                },
            }],
        }
        check_outputs(job, document, document)

        sensitivity["orders"] = ["first", "mixed_second"]
        sensitivity["first"] = ["product.strike"]
        plan.update(
            requested_orders=["first", "mixed_second"],
            sensitivity_graph_node_capacity=11,
            maximum_live_scenarios=11,
            first_sensitivity_count=1,
        )
        row = document["results"][0]
        row["stencils"]["product.strike"] = {
            "central": 3.0,
            "first": 2.5,
            "second": 3.5,
            "displacement": 0.5,
            "kind": "centered",
            "represented_width": 1.0,
            "first_weight": 0.0,
            "second_weight": 0.0,
        }
        row["outputs"]["gradients"] = {"product.strike": 0.7}
        row["outputs"]["gradient_standard_errors"] = {
            "product.strike": 0.07
        }
        document["sensitivity"] = copy.deepcopy(sensitivity)
        document["summary"] = {**plan, "seed": 719}
        check_outputs(job, document, document)

    def test_mutations_are_rejected(self):
        mutations = [
            lambda d: d["summary"].update(seed=720),
            lambda d: d["summary"].update(sensitivity_count=2),
            lambda d: d["summary"].update(price_moment_batches_per_price=2),
            lambda d: d["summary"].update(central_work_policy="always_replayed"),
            lambda d: d["summary"].update(sensitivity_batch_size=2),
            lambda d: d["summary"].update(sensitivity_batches_per_price=2),
            lambda d: d["summary"].update(maximum_live_scenarios=5),
            lambda d: d["summary"].update(kernel_launches_per_price_batch=2),
            lambda d: d["summary"].update(sensitivity_block_count=2),
            lambda d: d["summary"].update(work_distribution="host materialized"),
            lambda d: d["results"][0]["outputs"]["gradient_standard_errors"].update({"model.rho":-1}),
            lambda d: d["results"][0]["outputs"]["gradients"].update({"extra":.3}),
            lambda d: d["results"][0]["stencils"]["model.rho"].update(first_weight=-15),
            lambda d: d["results"][0]["stencils"]["model.rho"].update(first=.9),
            lambda d: d["results"][0]["stencils"]["model.rho"].update(displacement=.0625),
            lambda d: d["sensitivity"]["parameters"][0].update(boundary="central_only"),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                document = copy.deepcopy(self.document)
                mutate(document)
                with self.assertRaises(ValueError): check_outputs(self.job,self.catalog,document)

    def test_selection_changes_semantic_identity(self):
        job = {**self.job,"kind":"price_gradients","identity":"heston/european_option",
               "dataset":"test.json","rows":1,"sample_shape":None}
        previous = fingerprint(job)
        changed = copy.deepcopy(job)
        changed["sensitivity"]["parameters"][0]["displacement"] *= 2
        self.assertNotEqual(previous,fingerprint(changed))

    def test_manifest_aliases_and_selection_inventory(self):
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0,str(root/"tools/codegen/pricing_bindings"))
        from capability_manifest import (
            PRICE_GRADIENT_DATASET_SPECS,
            PRICE_GRADIENT_SOURCE_BY_GENERATOR,
            PRICE_VARIANTS,
            resolve_rng_domain,
        )
        from tools.datasets.generate_catalog import inventory
        jobs = inventory(root,{"price_gradients"},set(),set())
        self.assertEqual(len(jobs),len(PRICE_GRADIENT_DATASET_SPECS))
        self.assertEqual({spec.model for spec in PRICE_GRADIENT_DATASET_SPECS},
                         {"bates", "black_scholes", "heston", "heston_3_2",
                          "cev", "kou", "merton", "normal_inverse_gaussian",
                          "sabr", "schobel_zhu", "stein_stein",
                          "variance_gamma", "cir", "cir_plus_plus", "g2",
                          "g2_plus_plus", "hull_white",
                          "ornstein_uhlenbeck", "vasicek"})
        self.assertEqual({job["target"] for job in jobs},{spec.cmake_target for spec in PRICE_GRADIENT_DATASET_SPECS})
        diagonal = [
            spec for spec in PRICE_GRADIENT_DATASET_SPECS
            if "diagonal_second" in spec.sensitivity_orders
        ]
        self.assertEqual(len(diagonal), 2 * len(PRICE_GRADIENT_DATASET_SPECS) // 3)
        self.assertEqual(
            {spec.model for spec in diagonal},
            {"bates", "black_scholes", "cev", "heston", "heston_3_2",
             "kou", "merton", "normal_inverse_gaussian", "sabr",
             "schobel_zhu", "stein_stein", "variance_gamma", "cir",
             "cir_plus_plus", "g2", "g2_plus_plus", "hull_white",
             "ornstein_uhlenbeck", "vasicek"},
        )
        equity_products = {variant.product for variant in PRICE_VARIANTS}
        self.assertTrue(all(
            spec.sensitivity_orders in {
                ("first", "diagonal_second"),
                ("first", "diagonal_second", "mixed_second"),
            }
            and spec.product in equity_products | {
                "american_option", "european_swaption", "bermudan_swaption",
                "rate_option", "zero_coupon_bond_option",
            }
            for spec in diagonal
        ))
        full_hessian = [
            spec for spec in PRICE_GRADIENT_DATASET_SPECS
            if "mixed_second" in spec.sensitivity_orders
        ]
        self.assertEqual(
            len(full_hessian), len(PRICE_GRADIENT_DATASET_SPECS) // 3
        )
        self.assertTrue(all(
            spec.sensitivity_orders
                == ("first", "diagonal_second", "mixed_second")
            for spec in full_hessian
        ))
        self.assertEqual(
            sum(spec.product == "american_option" for spec in diagonal),
            64,
        )
        self.assertEqual(
            sum(spec.product == "european_swaption" for spec in diagonal),
            80,
        )
        self.assertEqual(
            sum(spec.product == "bermudan_swaption" for spec in diagonal),
            80,
        )
        import json
        for spec in PRICE_GRADIENT_DATASET_SPECS:
            recipe = json.loads((root/spec.generator_path).with_name("recipe.yaml").read_text())
            if spec.product == "american_option":
                product_dataset = "american_options_01.json"
            elif spec.product == "european_swaption":
                product_dataset = "european_swaptions_01.json"
            elif spec.product == "bermudan_swaption":
                product_dataset = "bermudan_swaptions_01.json"
            elif spec.product == "rate_option":
                product_dataset = "rate_options_01.json"
            elif spec.product == "zero_coupon_bond_option":
                product_dataset = "zero_coupon_bond_options_01.json"
            else:
                product_dataset = next(
                    variant.product_dataset_id + ".json"
                    for variant in PRICE_VARIANTS
                    if variant.name == spec.variant
                )
            expected_product = (
                f"datasets/product/{spec.product}/{product_dataset}"
            )
            self.assertEqual(recipe["product_input"], expected_product)
            if spec.engine not in {"equity_closed_form", "fixed_income_closed_form"}:
                self.assertEqual(resolve_rng_domain(spec).seed("dynamics"),
                    resolve_rng_domain(PRICE_GRADIENT_SOURCE_BY_GENERATOR[spec.generator_path]).seed("dynamics"))


if __name__ == "__main__":
    unittest.main()
