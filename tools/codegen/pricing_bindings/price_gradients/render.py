"""Render gradient wrappers from complete engine-owned C++ templates."""

from pathlib import Path
from string import Template
import json
from .manifest import default_sensitivities
from manifest import PRICE_VARIANTS


def render_bindings(output_root, specifications, template_root, write_generated):
    generated = []
    for spec in specifications:
        values = {"model": spec.pricing.model, "maximum": str(spec.maximum_sensitivities)}
        for suffix in ("cuh", "cu"):
            if spec.pricing.product == "american_option":
                template = (
                    "pricing/longstaff_schwartz/equity/"
                    "american_option_price_gradients/product."
                    + suffix + ".tpl"
                )
            else:
                template = "pricing/markovian/price_gradients/product." + suffix + ".tpl"
            if suffix == "cu" and spec.closed_form:
                template = "pricing/closed_form/black_scholes/price_gradients/european_option.cu.tpl"
            destination = output_root / (spec.unit_path + "." + suffix)
            destination.parent.mkdir(parents=True, exist_ok=True)
            write_generated(destination, Template((template_root / template).read_text()).substitute(values))
            generated.append(destination)
    return generated


def render_recipes(output_root, specifications, sources, model_specs, resolve_rng_domain, template_root, write_generated):
    generated = []
    for spec in specifications:
        template_path = "catalog/pricing/price_gradients/generator.cpp.tpl"
        if spec.product == "american_option":
            template_path = (
                "catalog/pricing/price_gradients/"
                "american_option_generator.cpp.tpl"
            )
        template = Template((template_root / template_path).read_text())
        source = sources[spec.recipe_path]
        model = model_specs[spec.model]
        stochastic = spec.engine != "equity_closed_form"
        exact_transition = model.transition_contract == "exact transition"
        selected = default_sensitivities(spec.model)
        seed = resolve_rng_domain(spec).seed("dynamics") if stochastic else 0
        model_input = f"datasets/{model.source_prefix}/parameters/{model.parameter_dataset_id}.json"
        if spec.product == "american_option":
            product_input = (
                "datasets/product/american_option/"
                "american_options_01.json"
            )
        else:
            variant = next(
                item for item in PRICE_VARIANTS
                if item.name == spec.variant
            )
            product_input = (
                f"datasets/product/{spec.product}/"
                f"{variant.product_dataset_id}.json"
            )
        values = {"model": spec.model, "model_input": model_input, "product_input": product_input,
            "dataset": spec.dataset_path, "catalog": spec.catalog_yaml_path, "url": spec.url,
            "source_recipe": source.recipe_path, "seed": str(seed), "stochastic": str(stochastic).lower(),
            "side": "call" if spec.variant.endswith("calls") else "put",
            "construction": "Aligned" if spec.construction == "aligned" else "CartesianProduct",
            "family": "closed_form" if not stochastic else "equity_exact_mc" if exact_transition else "equity_step_mc",
            "exact_transition": str(exact_transition).lower(),
            "selections": ",\n        ".join('{"' + item["parameter"] + '", {' + repr(item["displacement"])
                + ', pg::BumpScale::' + item["scale"] + '}}' for item in selected)}
        destination = output_root / spec.recipe_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_generated(destination, template.substitute(values))
        generated.append(destination)
        metadata = {"schema_version":1, "kind":"price_gradients", "database_id":spec.dataset_id,
            "generator":"generator.cpp", "model_input":model_input, "product_input":product_input,
            "dataset":spec.dataset_path, "catalog_output":spec.catalog_yaml_path, "construction":spec.construction,
            "paths_per_price":1048576 if stochastic else 0, "dynamics_seed":seed,
            "sensitivity":{"method":"finite_difference_shared_innovations", "parameters":selected,
                           "source_price_recipe":source.recipe_path},
            "launch_profile":"gradient candidate; inspect compiled specialization; not performance-qualified",
            "validation":{"status":"pending","verified":False}}
        if exact_transition:
            metadata["time_representation"] = {"kind":"exact_terminal_transition",
                "contractual_days_per_year":252,"maturity_bump_steps_per_year":504}
        else:
            metadata["time_grid"] = {"steps_per_year":504,"simulation_steps_per_day":2,"delta_t":"1 / 504"}
        recipe_path = destination.with_name("recipe.yaml")
        write_generated(recipe_path, json.dumps(metadata, indent=2) + "\n")
        generated.append(recipe_path)
    return generated
