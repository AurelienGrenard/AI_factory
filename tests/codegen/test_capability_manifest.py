"""Contract tests for the typed repository capability matrix."""

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
import hashlib
import json
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[2]
CODEGEN = ROOT / "tools/codegen/pricing_bindings"
sys.path.insert(0, str(CODEGEN))

from capability_manifest import (  # noqa: E402
    AVAILABLE_DATASET_SPECS,
    CARTESIAN_PRICE_DATASET_SPECS,
    CARTESIAN_PRICE_SOURCE_BY_RECIPE,
    CURVE_BY_NAME,
    CURVE_SPECS,
    DATASET_SPECS,
    DECLARED_PRODUCT_BINDING_PATHS,
    DEFERRED_DATASET_SPECS,
    ENGINE_SPECS,
    EQUITY_EARLY_EXERCISE_UNITS,
    FIXED_INCOME_CAPABILITIES,
    FIXED_INCOME_UNITS,
    GENERATED_PRODUCT_BINDING_PATHS,
    MODEL_SPECS,
    MODEL_BY_NAME,
    PriceCapabilityState,
    PRODUCT_BINDING_SPECS,
    PRICE_DELTA_BINDING_SPECS,
    GENERATED_PRICE_DELTA_BINDING_PATHS,
    PriceDeltaBindingSpec,
    PRICE_DELTA_DATASET_SPECS,
    PRICE_DELTA_SOURCE_BY_RECIPE,
    PRICE_GRADIENT_DATASET_SPECS,
    PRICE_GRADIENT_SOURCE_BY_RECIPE,
    pricing_launch_family,
    PRODUCT_SPECS,
    RNG_COMMON_RANDOM_NUMBER_ALLOWLIST,
    RNG_DOMAIN_SPECS,
    RngDomainSpec,
    classify_price_capability,
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
from sample_manifest import SAMPLE_MODELS, SAMPLE_MODEL_BY_NAME, validate_model_contracts  # noqa: E402
from generate import (  # noqa: E402
    _fixed_income_template_values,
    _repository_inventory_diagnostics,
    cmake_manifest_text,
    compare,
)

class CapabilityManifestTest(unittest.TestCase):
    def test_unchanged_generated_output_preserves_timestamp(self):
        from generate import _write_generated
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "generated.cu"
            _write_generated(path, "// fixture\n")
            timestamp = path.stat().st_mtime_ns
            _write_generated(path, "// fixture\n")
            self.assertEqual(path.stat().st_mtime_ns, timestamp)
            _write_generated(path, "// changed\n")
            self.assertEqual(path.read_text(), "// changed\n")

    def test_price_delta_equity_coverage_is_explicit_and_generated(self):
        self.assertEqual(sum(s.pricing.engine not in {"equity_n_factor", "equity_volterra_fft"}
                             for s in PRICE_DELTA_BINDING_SPECS), 261)
        rough = [s for s in PRICE_DELTA_BINDING_SPECS
                 if s.pricing.engine in {"equity_n_factor", "equity_volterra_fft"}]
        self.assertEqual(len(rough), 126)
        for spec in rough:
            self.assertEqual(spec.path_strategy, "coupled" if spec.pricing.model == "rough_sabr" else "multiplicative")
        self.assertEqual(len(set(GENERATED_PRICE_DELTA_BINDING_PATHS)), 774)
        for spec in PRICE_DELTA_BINDING_SPECS:
            self.assertIn(spec.pricing, PRODUCT_BINDING_SPECS)
            self.assertEqual(spec.unit_path, spec.pricing.unit_path + "_price_delta")
            for path in spec.paths:
                self.assertTrue((ROOT / path).is_file(), path)
            with self.assertRaises(ValueError):
                PriceDeltaBindingSpec(spec.pricing, "guess_from_name")
        with self.assertRaises(ValueError):
            PriceDeltaBindingSpec(PRICE_DELTA_BINDING_SPECS[0].pricing, "closed_form_bump")

        by_source = {}
        for dataset in PRICE_DELTA_DATASET_SPECS:
            source = PRICE_DELTA_SOURCE_BY_RECIPE[dataset.recipe_path].recipe_path
            if source in CARTESIAN_PRICE_SOURCE_BY_RECIPE:
                source = CARTESIAN_PRICE_SOURCE_BY_RECIPE[source].recipe_path
            by_source.setdefault(source, []).append(dataset)
        self.assertEqual(len(by_source), len(PRICE_DELTA_DATASET_SPECS) // 2)
        for variants in by_source.values():
            self.assertEqual(len(variants), 2)
            self.assertEqual({dataset.construction for dataset in variants}, {"aligned", "cartesian"})
            cartesian = next(dataset for dataset in variants if dataset.construction == "cartesian")
            self.assertTrue(cartesian.dataset_id.endswith("_cartesian_price_delta"))
            self.assertEqual(cartesian.layout, "model_major_product_fastest_price_delta_rows")

    def test_sample_parameter_laws_cover_factories_and_public_fields(self) -> None:
        from generate import _render_sample_generation_header
        for model in SAMPLE_MODELS:
            validate_model_contracts((model,))
            rendered = _render_sample_generation_header(model)
            for name, law in model.derived_parameter_laws:
                self.assertIn(json.dumps(name), rendered)
                self.assertIn(json.dumps(law), rendered)
                incomplete = replace(model, derived_parameter_laws=tuple(
                    pair for pair in model.derived_parameter_laws if pair[0] != name))
                with self.assertRaisesRegex(ValueError, "undocumented sample parameter laws"):
                    validate_model_contracts((incomplete,))
        self.assertEqual(dict(SAMPLE_MODEL_BY_NAME["cir"].derived_parameter_laws),
                         dict(SAMPLE_MODEL_BY_NAME["cir_plus_plus"].derived_parameter_laws))

    def test_fft_sample_helpers_use_compiled_geometry(self) -> None:
        from generate import _render_sample_generation_header, _sample_launch_lambda
        for model in SAMPLE_MODELS:
            header = _render_sample_generation_header(model)
            if model.backend == "volterra":
                self.assertIn(f"{model.name}_sample_block_dimensions(value.maximum_maturity_days)", header)
                self.assertIn('"volterra_samples", native_block', header)
                self.assertNotIn("threads_per_block, block_count,", _sample_launch_lambda(model))
            else:
                self.assertNotIn("sample_block_dimensions", header)

    def test_pricing_launch_families_cover_the_canonical_bindings(self) -> None:
        families = {pricing_launch_family(binding) for binding in PRODUCT_BINDING_SPECS}
        self.assertEqual(families, {
            "closed_form", "jamshidian", "equity_exact_mc", "equity_step_mc",
            "fixed_income_mc", "equity_lsm", "gaussian_rate_lsm",
            "terminal_forward_lsm", "rough_n_factor", "rough_fft",
        })
        for binding in PRODUCT_BINDING_SPECS:
            family = pricing_launch_family(binding)
            if family == "terminal_forward_lsm":
                self.assertTrue(binding.terminal_forward_dynamics)
            if family == "jamshidian":
                self.assertEqual(binding.product, "european_swaption")
                self.assertEqual(binding.engine, "fixed_income_closed_form")

    def test_declared_cardinalities_are_complete(self) -> None:
        self.assertEqual(len(MODEL_SPECS), 25)
        self.assertEqual(len(PRODUCT_SPECS), 26)
        self.assertEqual(
            len(AVAILABLE_DATASET_SPECS),
            722 + len(CARTESIAN_PRICE_DATASET_SPECS) + len(PRICE_DELTA_DATASET_SPECS) + len(PRICE_GRADIENT_DATASET_SPECS),
        )
        self.assertEqual(len(CARTESIAN_PRICE_DATASET_SPECS), 618)
        self.assertEqual(len(DEFERRED_DATASET_SPECS), 0)
        self.assertEqual(len(FIXED_INCOME_UNITS), 47)
        self.assertEqual(len(PRODUCT_BINDING_SPECS), 427)
        self.assertEqual(len(DECLARED_PRODUCT_BINDING_PATHS), 854)
        self.assertEqual(len(GENERATED_PRODUCT_BINDING_PATHS), 820)

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
            for symbol in (*engine.concepts, *engine.launchers, *engine.runners):
                source = ROOT / symbol.path
                self.assertTrue(source.is_file(), symbol.path)
                self.assertIn(symbol.name.rsplit("::", 1)[-1], source.read_text())
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
            "fixed_income_monte_carlo",
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
        self.assertTrue(RNG_COMMON_RANDOM_NUMBER_ALLOWLIST)
        for delta in PRICE_DELTA_DATASET_SPECS:
            source = PRICE_DELTA_SOURCE_BY_RECIPE[delta.recipe_path]
            if delta.engine != "equity_closed_form":
                self.assertEqual(resolve_rng_domain(delta).seed("dynamics"),
                                 resolve_rng_domain(source).seed("dynamics"))
                self.assertIn(tuple(sorted((delta.recipe_path, source.recipe_path))),
                              RNG_COMMON_RANDOM_NUMBER_ALLOWLIST)
        validate_rng_domain_specs(RNG_DOMAIN_SPECS)

    def test_bates_sample_recipes_do_not_share_dynamics_keys(self) -> None:
        samples = [
            dataset for dataset in AVAILABLE_DATASET_SPECS
            if dataset.model == "bates" and dataset.dataset_kind == "samples"
        ]
        first = resolve_rng_domain(samples[0]).interval("dynamics")
        second = resolve_rng_domain(samples[1]).interval("dynamics")
        self.assertLessEqual(first[1], second[0])

    def test_rng_v2_appends_g2_mc_without_rekeying_v1(self) -> None:
        aliases = set(CARTESIAN_PRICE_SOURCE_BY_RECIPE) | set(PRICE_DELTA_SOURCE_BY_RECIPE) | set(PRICE_GRADIENT_SOURCE_BY_RECIPE)
        legacy = [(d.recipe_path, d.ordinal, d.seed("dynamics"))
                  for d in RNG_DOMAIN_SPECS if d.ordinal < 588 and d.recipe_path not in aliases]
        self.assertEqual(len(legacy), 588)
        self.assertEqual(hashlib.sha256(json.dumps(legacy, separators=(",", ":")).encode()).hexdigest(),
            "7472f3fff75219add3eb1b993d5d27dcfe30db2bdc5b1457c7e3f918b6ea0638")
        appended = [d for d in RNG_DOMAIN_SPECS
                    if 588 <= d.ordinal < 594 and d.recipe_path not in aliases]
        self.assertEqual(len(appended), 6)
        self.assertTrue(all(d.version == 3 for d in RNG_DOMAIN_SPECS))
        self.assertTrue(all("/european_" in d.recipe_path and "/g2" in d.recipe_path
                            for d in appended))

    def test_rng_v3_appends_cir_plus_plus_without_rekeying_v2(self) -> None:
        aliases = set(CARTESIAN_PRICE_SOURCE_BY_RECIPE) | set(PRICE_DELTA_SOURCE_BY_RECIPE) | set(PRICE_GRADIENT_SOURCE_BY_RECIPE)
        legacy = [(d.recipe_path, d.ordinal, d.seed("dynamics"))
                  for d in RNG_DOMAIN_SPECS if d.ordinal < 594 and d.recipe_path not in aliases]
        self.assertEqual(len(legacy), 594)
        self.assertEqual(hashlib.sha256(json.dumps(legacy, separators=(",", ":")).encode()).hexdigest(),
            "d73cd893f1a5fb413e3a1921a3631c0e3f26f00b86886772c1f33d8d61f2c617")
        appended = [d for d in RNG_DOMAIN_SPECS
                    if d.ordinal >= 594 and d.recipe_path not in aliases]
        self.assertEqual(len(appended), 6)
        self.assertTrue(all("/cir_plus_plus/" in d.recipe_path for d in appended))

    def test_cir_plus_plus_composes_all_products_for_both_curves(self) -> None:
        self.assertEqual(set(SAMPLE_MODEL_BY_NAME), {model.name for model in SAMPLE_MODELS})
        self.assertEqual(SAMPLE_MODEL_BY_NAME["cir_plus_plus"].constructor,
                         SAMPLE_MODEL_BY_NAME["cir"].constructor)
        self.assertEqual(SAMPLE_MODEL_BY_NAME["cir_plus_plus"].derived_parameter_laws,
                         SAMPLE_MODEL_BY_NAME["cir"].derived_parameter_laws)
        self.assertIn("volatility", dict(SAMPLE_MODEL_BY_NAME["cir_plus_plus"].derived_parameter_laws))
        for curve in ("nelson_siegel", "svensson"):
            for product, variant in (
                ("rate_option", "caplets"), ("rate_option", "floorlets"),
                ("zero_coupon_bond_option", "zero_coupon_bond_calls"),
                ("zero_coupon_bond_option", "zero_coupon_bond_puts"),
                ("european_swaption", "european_payer_swaptions"),
                ("european_swaption", "european_receiver_swaptions"),
                ("bermudan_swaption", "bermudan_payer_swaptions"),
                ("bermudan_swaption", "bermudan_receiver_swaptions"),
            ):
                resolved = resolve_complete_price_capability("cir_plus_plus", product, variant, curve)
                self.assertEqual(resolved.binding.owner, "generated")
                self.assertEqual(resolved.dataset.owner, "generated")
                self.assertEqual(resolved.engine.name, "fixed_income_lsm"
                    if product == "bermudan_swaption" else "fixed_income_closed_form")

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
        self.assertEqual(len(bindings), 10)
        cir = next(binding for binding in bindings if binding.model == "cir")
        self.assertEqual(
            cir.transition_contract,
            "exact terminal-forward state transition (last exercise bond numeraire)",
        )
        self.assertTrue(all(
            binding.transition_contract
                == "exact joint state/integral transition"
            for binding in bindings
            if not binding.terminal_forward_dynamics
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
                "american_option",
                "american_calls",
            )

    def test_price_capability_states_are_total_and_distinct(self) -> None:
        available = next(
            dataset for dataset in AVAILABLE_DATASET_SPECS
            if dataset.dataset_kind == "prices"
        )
        identity = (
            available.model or "",
            available.product or "",
            available.variant or "",
            available.curve,
        )
        self.assertIs(
            classify_price_capability(*identity).state,
            PriceCapabilityState.AVAILABLE,
        )
        deferred = replace(available, status="deferred")
        self.assertIs(
            classify_price_capability(*identity, datasets=(deferred,)).state,
            PriceCapabilityState.DEFERRED,
        )
        self.assertIs(
            classify_price_capability(*identity, datasets=(available, available)).state,
            PriceCapabilityState.AMBIGUOUS,
        )
        self.assertIs(
            classify_price_capability(
                "g2", "european_swaption", "european_payer_swaptions"
            ).state,
            PriceCapabilityState.AVAILABLE,
        )
        self.assertIs(
            classify_price_capability(
                "unknown_model", "european_option", "european_calls"
            ).state,
            PriceCapabilityState.UNCLASSIFIED,
        )

    def test_manifest_extensions_project_without_parallel_lookup_tables(self) -> None:
        base_model = next(
            model for model in MODEL_SPECS if model.asset_class == "fixed_income"
        )
        model = replace(
            base_model,
            name="fixture_model",
            display="Fixture model",
            renderer_alias="fixture_alias",
        )
        curve = replace(
            CURVE_SPECS[0],
            name="fixture_curve",
            display="Fixture curve",
            cpp_type="FixtureCurve",
            parameter_dataset_id="fixture_curve_01",
        )
        capability = replace(
            FIXED_INCOME_CAPABILITIES[0],
            model=model.name,
            curve=curve.name,
        )
        values = _fixed_income_template_values(
            capability,
            {**MODEL_BY_NAME, model.name: model},
            {**CURVE_BY_NAME, curve.name: curve},
        )
        self.assertEqual(values["model_alias"], "fixture_alias")
        self.assertEqual(values["curve_type"], "FixtureCurve")

        product = replace(
            PRODUCT_SPECS[-1],
            name="fixture_product",
            parameter_dataset_ids=("fixture_product_01",),
        )
        parameter_dataset = next(
            item for item in AVAILABLE_DATASET_SPECS
            if item.dataset_kind == "model_parameters"
        )
        dataset = replace(
            parameter_dataset,
            dataset_id="fixture_model_01",
            model=model.name,
            source_prefix="model/fixed_income/fixture_model",
            recipe_path=(
                "catalog/model/fixed_income/fixture_model/parameters/"
                "fixture_model_01/generator.cpp"
            ),
        )
        rendered = cmake_manifest_text(
            (*MODEL_SPECS, model),
            (*PRODUCT_SPECS, product),
            (*CURVE_SPECS, curve),
            (*AVAILABLE_DATASET_SPECS, dataset),
        )
        self.assertIn("fixture_model", rendered)
        self.assertIn("fixture_product", rendered)
        self.assertIn("fixture_curve", rendered)
        self.assertIn(dataset.recipe_path, rendered)
        self.assertEqual(dataset.cmake_target, "generate_fixture_model_01")

    def test_optional_mathdx_never_removes_parameter_generators(self) -> None:
        conditional = {
            dataset.recipe_path for dataset in AVAILABLE_DATASET_SPECS
            if dataset.condition == "AI_FACTORY_MATHDX_ROOT"
        }
        parameter_sources = {
            dataset.recipe_path for dataset in AVAILABLE_DATASET_SPECS
            if dataset.dataset_kind.endswith("_parameters")
        }
        self.assertFalse(conditional & parameter_sources)
        for model in (
            "rough_bergomi",
            "rough_sabr",
            "log_modulated_rough_bergomi",
            "rough_stein_stein",
        ):
            self.assertTrue(any(
                dataset.model == model
                and dataset.dataset_kind == "model_parameters"
                for dataset in AVAILABLE_DATASET_SPECS
            ))

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
