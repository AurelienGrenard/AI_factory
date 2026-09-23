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
        from capability_manifest import PRICE_GRADIENT_DATASET_SPECS, PRICE_GRADIENT_SOURCE_BY_GENERATOR, resolve_rng_domain
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
        self.assertEqual(len(diagonal), len(PRICE_GRADIENT_DATASET_SPECS) // 2)
        self.assertEqual(
            {spec.model for spec in diagonal},
            {"bates", "black_scholes", "cev", "heston", "heston_3_2",
             "kou", "merton", "normal_inverse_gaussian", "sabr",
             "schobel_zhu", "stein_stein", "variance_gamma", "cir",
             "cir_plus_plus", "g2", "g2_plus_plus", "hull_white",
             "ornstein_uhlenbeck", "vasicek"},
        )
        self.assertTrue(all(
            spec.sensitivity_orders == ("first", "diagonal_second")
            and spec.product in {
                "european_option", "asset_or_nothing_option",
                "digital_option", "american_option", "european_swaption",
                "bermudan_swaption"
            }
            for spec in diagonal
        ))
        self.assertEqual(
            sum(spec.product == "american_option" for spec in diagonal),
            32,
        )
        self.assertEqual(
            sum(spec.product == "european_swaption" for spec in diagonal),
            4,
        )
        self.assertEqual(
            sum(spec.product == "bermudan_swaption" for spec in diagonal),
            40,
        )
        import json
        for spec in PRICE_GRADIENT_DATASET_SPECS:
            recipe = json.loads((root/spec.generator_path).with_name("recipe.yaml").read_text())
            product_dataset = {
                "european_option": "european_options_01.json",
                "asset_or_nothing_option":
                    "asset_or_nothing_options_01.json",
                "digital_option": "digital_options_01.json",
                "american_option": "american_options_01.json",
                "european_swaption": "european_swaptions_01.json",
                "bermudan_swaption": "bermudan_swaptions_01.json",
            }[spec.product]
            expected_product = (
                f"datasets/product/{spec.product}/{product_dataset}"
            )
            self.assertEqual(recipe["product_input"], expected_product)
            if spec.engine not in {"equity_closed_form", "fixed_income_closed_form"}:
                self.assertEqual(resolve_rng_domain(spec).seed("dynamics"),
                    resolve_rng_domain(PRICE_GRADIENT_SOURCE_BY_GENERATOR[spec.generator_path]).seed("dynamics"))


if __name__ == "__main__":
    unittest.main()
