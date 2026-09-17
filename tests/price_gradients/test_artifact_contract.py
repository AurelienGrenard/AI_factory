"""Mutation checks for gradient metadata, column identity, finite errors and stencil geometry."""
import copy
from pathlib import Path
import sys
import unittest

from tools.datasets.price_gradients.contract import check_outputs
from tools.datasets.dataset_provenance import specification


class ArtifactContractTests(unittest.TestCase):
    def setUp(self):
        self.selection = {"method":"finite_difference_shared_innovations", "source_price_recipe":"source",
            "parameters":[{"parameter":"model.rho", "displacement":.125, "scale":"absolute",
                           "boundary":"central_then_one_sided_order2"}]}
        self.grid = {"steps_per_year":504,"simulation_steps_per_day":2,"delta_t":"1 / 504"}
        self.plan = {"paths_per_price":32,"threads_per_block":128,"sensitivity_count":1,
                     "scenario_count":3,"gradient_layout":"row_major_selection_order",
                     "sensitivity_batch_size":2,"sensitivity_batches_per_price":1,"maximum_live_scenarios":3,
                     "price_moment_batches_per_price":1,"central_work_policy":"host_classified_payoff_state_or_none",
                     "kernel_launches_per_price_batch":1,"full_batch_block_count":0,"tail_batch_block_count":1}
        self.job = {"sensitivity":self.selection,"time_key":"time_grid","time_configuration":self.grid,"launch_plan":self.plan,
                    "rng_stream_seeds":{"dynamics":719}}
        self.catalog = {"sensitivity":self.selection,"time_grid":self.grid,"summary":{**self.plan,"seed":719}}
        self.document = {**self.catalog,"results":[{"stencils":{"model.rho":{"central":1.,"first":.875,"second":.75,
            "displacement":.125,"kind":"backward","represented_width":-.125,"first_weight":-16.,"second_weight":4.}},
            "outputs":{"price":.1,"standard_error":.01,"gradients":{"model.rho":.2},"gradient_standard_errors":{"model.rho":.02}}}]}

    def test_valid_one_sided_contract(self):
        check_outputs(self.job,self.catalog,self.document)

    def test_mutations_are_rejected(self):
        mutations = [
            lambda d: d["summary"].update(seed=720),
            lambda d: d["summary"].update(sensitivity_count=2),
            lambda d: d["summary"].update(price_moment_batches_per_price=2),
            lambda d: d["summary"].update(central_work_policy="always_replayed"),
            lambda d: d["summary"].update(sensitivity_batch_size=4),
            lambda d: d["summary"].update(sensitivity_batches_per_price=2),
            lambda d: d["summary"].update(maximum_live_scenarios=5),
            lambda d: d["summary"].update(kernel_launches_per_price_batch=2),
            lambda d: d["summary"].update(tail_batch_block_count=2),
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
        previous = specification(job)
        changed = copy.deepcopy(job)
        changed["sensitivity"]["parameters"][0]["displacement"] *= 2
        self.assertNotEqual(previous,specification(changed))

    def test_manifest_aliases_and_selection_inventory(self):
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0,str(root/"tools/codegen/pricing_bindings"))
        from capability_manifest import PRICE_GRADIENT_DATASET_SPECS, PRICE_GRADIENT_SOURCE_BY_RECIPE, resolve_rng_domain
        from tools.datasets.generate_catalog import inventory
        jobs = inventory(root,{"price_gradients"},set(),set())
        self.assertEqual(len(jobs),20)
        self.assertEqual({spec.model for spec in PRICE_GRADIENT_DATASET_SPECS},
                         {"black_scholes", "heston", "cev", "merton"})
        self.assertEqual({job["target"] for job in jobs},{spec.cmake_target for spec in PRICE_GRADIENT_DATASET_SPECS})
        import json
        for spec in PRICE_GRADIENT_DATASET_SPECS:
            recipe = json.loads((root/spec.recipe_path).with_name("recipe.yaml").read_text())
            expected_product = (
                "datasets/product/american_option/american_options_01.json"
                if spec.product == "american_option"
                else "datasets/product/european_option/european_options_01.json"
            )
            self.assertEqual(recipe["product_input"], expected_product)
            if spec.engine != "equity_closed_form":
                self.assertEqual(resolve_rng_domain(spec).seed("dynamics"),
                    resolve_rng_domain(PRICE_GRADIENT_SOURCE_BY_RECIPE[spec.recipe_path]).seed("dynamics"))


if __name__ == "__main__":
    unittest.main()
