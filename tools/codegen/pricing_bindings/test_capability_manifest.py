"""Contract tests for the typed repository capability matrix."""

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from capability_manifest import (  # noqa: E402
    AVAILABLE_DATASET_SPECS,
    CURVE_SPECS,
    DATASET_SPECS,
    DECLARED_PRODUCT_BINDING_PATHS,
    DEFERRED_DATASET_SPECS,
    ENGINE_SPECS,
    EQUITY_EARLY_EXERCISE_UNITS,
    FIXED_INCOME_UNITS,
    GENERATED_PRODUCT_BINDING_PATHS,
    MODEL_SPECS,
    PRODUCT_BINDING_SPECS,
    PRODUCT_SPECS,
    RNG_COMMON_RANDOM_NUMBER_ALLOWLIST,
    RNG_DOMAIN_SPECS,
    RngDomainSpec,
    derive_equity_product_specs,
    resolve_rng_domain,
    resolve_complete_price_capability,
    resolve_price_capability,
    validate_dataset_spec,
    validate_price_capability_graph,
    validate_rng_domain_specs,
)
from manifest import (  # noqa: E402
    MODEL_RECIPE_SPECS,
    PRICE_VARIANTS,
    ROUGH_PRODUCT_BINDINGS,
    PriceVariant,
    RoughProductBinding,
    derive_model_recipe_specs,
    validate_derived_model_recipe_specs,
)
from sample_manifest import SAMPLE_MODELS  # noqa: E402
from generate import (  # noqa: E402
    _repository_inventory_diagnostics,
    compare,
)


ROOT = HERE.parents[2]


class CapabilityManifestTest(unittest.TestCase):
    def test_declared_cardinalities_are_complete(self) -> None:
        self.assertEqual(len(MODEL_SPECS), 24)
        self.assertEqual(len(PRODUCT_SPECS), 26)
        self.assertEqual(len(AVAILABLE_DATASET_SPECS), 697)
        self.assertEqual(len(DEFERRED_DATASET_SPECS), 0)
        self.assertEqual(len(FIXED_INCOME_UNITS), 35)
        self.assertEqual(len(PRODUCT_BINDING_SPECS), 416)
        self.assertEqual(len(DECLARED_PRODUCT_BINDING_PATHS), 832)
        self.assertEqual(len(GENERATED_PRODUCT_BINDING_PATHS), 798)

    def test_every_domain_object_has_the_src_taxonomy_prefix(self) -> None:
        for model in MODEL_SPECS:
            self.assertTrue(
                (ROOT / "src" / model.source_prefix).is_dir(),
                model.source_prefix,
            )
        for product in PRODUCT_SPECS:
            self.assertTrue(
                (ROOT / "src" / product.source_prefix).is_dir(),
                product.source_prefix,
            )
        for curve in CURVE_SPECS:
            self.assertTrue(
                (ROOT / "src" / curve.source_prefix).is_dir(),
                curve.source_prefix,
            )

    def test_dataset_paths_inherit_one_canonical_prefix(self) -> None:
        for dataset in DATASET_SPECS:
            self.assertTrue(
                dataset.recipe_path.startswith(
                    f"catalog/{dataset.source_prefix}/"
                ),
                dataset.recipe_path,
            )
            self.assertTrue(
                dataset.dataset_path.startswith(
                    f"datasets/{dataset.source_prefix}/"
                ),
                dataset.dataset_path,
            )
            self.assertEqual(
                dataset.url,
                "https://datasets.ai-factory.example/v1/"
                + dataset.dataset_path.removeprefix("datasets/"),
            )

    def test_owners_renderers_contracts_and_targets_are_complete(self) -> None:
        self.assertEqual(
            {dataset.owner for dataset in DATASET_SPECS},
            {"generated", "hand_written"},
        )
        for dataset in DATASET_SPECS:
            self.assertEqual(dataset.owner == "generated", dataset.template is not None)
        for engine in ENGINE_SPECS:
            self.assertTrue(engine.binding_template_family)
            self.assertTrue(engine.recipe_template_family)
            self.assertTrue(engine.schedule_contract)
            self.assertTrue(engine.transition_contract)
            self.assertTrue(engine.analytics_contract)
            self.assertTrue(engine.concepts)
            self.assertTrue(engine.launchers)
            self.assertTrue(engine.runners)
            self.assertTrue(engine.instantiation_strategy)
        for model in MODEL_SPECS:
            self.assertTrue(model.transition_contract)
            self.assertTrue(model.analytics_contract)
            self.assertTrue(model.state_contract)
            self.assertTrue(model.observables)
            self.assertTrue(model.supported_architectures)
        for product in PRODUCT_SPECS:
            self.assertTrue(product.path_policy)
            self.assertTrue(product.schedule_contract)
            self.assertTrue(product.observation_contract)
            self.assertTrue(product.exercise_contract)
            self.assertTrue(product.required_capabilities)
        for dataset in DATASET_SPECS:
            self.assertTrue(dataset.construction)
            self.assertTrue(dataset.numerical_profile)
            self.assertTrue(dataset.layout)
        targets = [dataset.cmake_target for dataset in DATASET_SPECS]
        self.assertEqual(len(targets), len(set(targets)))

    def test_every_stochastic_dataset_has_a_disjoint_rng_domain(self) -> None:
        stochastic_engines = {
            "equity_markovian",
            "equity_volterra_fft",
            "equity_n_factor",
            "equity_lsm_fixed",
            "equity_lsm_exact",
            "fixed_income_lsm",
        }
        stochastic_datasets = {
            dataset.recipe_path
            for dataset in AVAILABLE_DATASET_SPECS
            if dataset.dataset_kind == "samples"
            or dataset.engine in stochastic_engines
        }
        self.assertEqual(
            {domain.recipe_path for domain in RNG_DOMAIN_SPECS},
            stochastic_datasets,
        )
        self.assertEqual(RNG_COMMON_RANDOM_NUMBER_ALLOWLIST, frozenset())
        validate_rng_domain_specs(RNG_DOMAIN_SPECS)

    def test_bates_sample_recipes_do_not_share_dynamics_keys(self) -> None:
        samples = [
            dataset for dataset in AVAILABLE_DATASET_SPECS
            if dataset.model == "bates" and dataset.dataset_kind == "samples"
        ]
        first = resolve_rng_domain(samples[0]).interval("dynamics")
        second = resolve_rng_domain(samples[1]).interval("dynamics")
        self.assertLessEqual(first[1], second[0])

    def test_rng_collision_fixture_is_rejected(self) -> None:
        first, second = RNG_DOMAIN_SPECS[:2]
        collision = RngDomainSpec(
            version=second.version,
            recipe_path=second.recipe_path,
            ordinal=first.ordinal,
            streams=second.streams,
        )
        with self.assertRaisesRegex(ValueError, "overlapping RNG intervals"):
            validate_rng_domain_specs((first, collision))

    def test_noncanonical_and_unowned_dataset_fixtures_are_rejected(self) -> None:
        rough_dataset = next(
            dataset for dataset in DATASET_SPECS
            if dataset.source_prefix.startswith("model/equity/rough/")
        )
        with self.assertRaisesRegex(ValueError, "does not inherit"):
            validate_dataset_spec(replace(
                rough_dataset,
                recipe_path=rough_dataset.recipe_path.replace("/rough", "", 1),
            ))
        product_dataset = next(
            dataset for dataset in DATASET_SPECS
            if dataset.source_prefix == "product/asian_option"
        )
        with self.assertRaisesRegex(ValueError, "does not inherit"):
            validate_dataset_spec(replace(
                product_dataset,
                recipe_path=product_dataset.recipe_path.replace(
                    "product/asian_option", "product/equity/asian_options"
                ),
            ))
        generated = next(
            dataset for dataset in DATASET_SPECS
            if dataset.owner == "generated"
        )
        with self.assertRaisesRegex(ValueError, "lacks a template"):
            validate_dataset_spec(replace(generated, template=None))

    def test_compare_reports_missing_and_mismatched_outputs(self) -> None:
        with TemporaryDirectory() as output_text, TemporaryDirectory() as reference_text:
            output = Path(output_text)
            reference = Path(reference_text)
            generated = output / "generated.txt"
            generated.write_text("new\n")
            capture = StringIO()
            with redirect_stdout(capture):
                self.assertEqual(compare([generated], output, reference), 1)
            self.assertIn("CODEGEN_MISSING", capture.getvalue())
            expected = reference / "generated.txt"
            expected.write_text("old\n")
            capture = StringIO()
            with redirect_stdout(capture):
                self.assertEqual(compare([generated], output, reference), 1)
            self.assertIn("CODEGEN_MISMATCH", capture.getvalue())

    def test_inventory_reports_orphan_and_renamed_bindings(self) -> None:
        with TemporaryDirectory() as reference_text:
            reference = Path(reference_text)
            orphan = (
                reference / "src/model/equity/markovian/fixture/product/"
                "orphan_option.cu"
            )
            orphan.parent.mkdir(parents=True)
            orphan.write_text("// fixture\n")
            diagnostics = _repository_inventory_diagnostics(reference)
            self.assertIn(
                "CODEGEN_EXTRA [product binding] "
                "src/model/equity/markovian/fixture/product/orphan_option.cu",
                diagnostics,
            )
            self.assertTrue(any(
                diagnostic.startswith("CODEGEN_MISSING [product binding]")
                for diagnostic in diagnostics
            ))

    def test_early_exercise_units_cover_every_published_american_model(
        self,
    ) -> None:
        self.assertEqual(
            EQUITY_EARLY_EXERCISE_UNITS,
            (
                "bates/product/american_option",
                "cev/product/american_option",
                "heston/product/american_option",
                "kou/product/american_option",
                "merton/product/american_option",
                "normal_inverse_gaussian/product/american_option",
                "schobel_zhu/product/american_option",
                "variance_gamma/product/american_option",
            ),
        )

    def test_resolver_selects_equity_and_fixed_income_engines(self) -> None:
        self.assertEqual(
            resolve_price_capability(
                "black_scholes",
                "european_option",
                "european_calls",
            ).engine,
            "equity_closed_form",
        )
        self.assertEqual(
            resolve_price_capability(
                "heston",
                "american_option",
                "american_puts",
            ).engine,
            "equity_lsm_fixed",
        )
        self.assertEqual(
            resolve_price_capability(
                "merton",
                "american_option",
                "american_calls",
            ).engine,
            "equity_lsm_exact",
        )
        self.assertEqual(
            resolve_price_capability(
                "g2_plus_plus",
                "bermudan_swaption",
                "bermudan_payer_swaptions",
                "svensson",
            ).engine,
            "fixed_income_lsm",
        )

    def test_complete_resolver_reaches_binding_target_and_recipe(self) -> None:
        resolved = resolve_complete_price_capability(
            "heston", "asian_option", "asian_calls"
        )
        self.assertEqual(resolved.engine.name, "equity_markovian")
        self.assertEqual(
            resolved.binding.unit_path,
            "src/model/equity/markovian/heston/product/asian_option",
        )
        self.assertEqual(
            resolved.target,
            "generate_heston_asian_calls_01",
        )
        self.assertEqual(resolved.recipe_path, resolved.dataset.recipe_path)

    def test_fixed_income_lsm_transitions_are_binding_specific(self) -> None:
        bindings = [
            binding for binding in PRODUCT_BINDING_SPECS
            if binding.engine == "fixed_income_lsm"
        ]
        self.assertEqual(len(bindings), 8)
        cir = next(binding for binding in bindings if binding.model == "cir")
        self.assertEqual(
            cir.transition_contract,
            "fixed-step joint state/integral transition",
        )
        self.assertTrue(all(
            binding.transition_contract
                == "exact joint state/integral transition"
            for binding in bindings
            if binding.model != "cir"
        ))

    def test_model_views_are_derived_from_one_canonical_entry(self) -> None:
        fixture = replace(
            SAMPLE_MODELS[0],
            name="fixture_model",
            display="Fixture model",
            legacy_url_name=None,
        )
        expanded = derive_model_recipe_specs(SAMPLE_MODELS + (fixture,))
        self.assertEqual(len(expanded), len(MODEL_RECIPE_SPECS) + 1)
        self.assertEqual(
            len(derive_model_recipe_specs(SAMPLE_MODELS[1:])),
            len(MODEL_RECIPE_SPECS) - 1,
        )
        divergent = (
            replace(MODEL_RECIPE_SPECS[0], display="Divergent"),
            *MODEL_RECIPE_SPECS[1:],
        )
        with self.assertRaisesRegex(ValueError, "diverges"):
            validate_derived_model_recipe_specs(SAMPLE_MODELS, divergent)

    def test_product_add_remove_dry_run_has_no_parallel_table(self) -> None:
        fixture_binding = RoughProductBinding(
            "fixture_option",
            "FixtureOption",
            "FixtureOptionPathPolicy",
            "terminal",
        )
        fixture_variant = PriceVariant(
            "fixture_calls",
            "fixture_option",
            "fixture_options",
            "fixture_options_01",
            "load_fixture_options",
            "call",
        )
        baseline = derive_equity_product_specs()
        expanded = derive_equity_product_specs(
            PRICE_VARIANTS + (fixture_variant,),
            ROUGH_PRODUCT_BINDINGS + (fixture_binding,),
        )
        self.assertEqual(len(expanded), len(baseline) + 1)
        reduced = derive_equity_product_specs(
            tuple(
                variant for variant in PRICE_VARIANTS
                if variant.product != "asian_option"
            ),
            tuple(
                binding for binding in ROUGH_PRODUCT_BINDINGS
                if binding.product != "asian_option"
            ),
        )
        self.assertEqual(len(reduced), len(baseline) - 1)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            derive_equity_product_specs(
                PRICE_VARIANTS,
                ROUGH_PRODUCT_BINDINGS + (fixture_binding,),
            )

    def test_composition_add_remove_dry_run_is_rejected(self) -> None:
        dataset = next(
            dataset for dataset in AVAILABLE_DATASET_SPECS
            if dataset.model == "heston"
            and dataset.product == "asian_option"
            and dataset.variant == "asian_calls"
        )
        binding = next(
            binding for binding in PRODUCT_BINDING_SPECS
            if binding.model == dataset.model
            and binding.product == dataset.product
            and binding.curve == dataset.curve
            and binding.engine == dataset.engine
        )
        without_binding = tuple(
            candidate for candidate in PRODUCT_BINDING_SPECS
            if candidate is not binding
        )
        with self.assertRaisesRegex(ValueError, "resolved 0"):
            validate_price_capability_graph(DATASET_SPECS, without_binding)
        with self.assertRaisesRegex(ValueError, "resolved 2"):
            validate_price_capability_graph(
                DATASET_SPECS,
                PRODUCT_BINDING_SPECS + (binding,),
            )

    def test_resolver_rejects_undeclared_combinations(self) -> None:
        with self.assertRaises(KeyError):
            resolve_price_capability(
                "black_scholes",
                "american_option",
                "american_calls",
            )
        with self.assertRaises(KeyError):
            resolve_price_capability(
                "g2",
                "european_swaption",
                "european_payer_swaptions",
            )

    def test_sample_publication_is_two_recipes_per_model(self) -> None:
        available_by_model = {
            model.name: {
                dataset.dataset_id
                for dataset in AVAILABLE_DATASET_SPECS
                if dataset.model == model.name
                and dataset.dataset_kind == "samples"
            }
            for model in MODEL_SPECS
        }
        self.assertTrue(all(
            dataset_ids == {"samples_01", "samples_02"}
            for dataset_ids in available_by_model.values()
        ))


if __name__ == "__main__":
    unittest.main()
