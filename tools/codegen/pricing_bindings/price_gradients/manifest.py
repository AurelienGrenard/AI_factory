"""Derive the implemented gradient integration surface from existing pricing bindings."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from capability_manifest import ProductBindingSpec


@dataclass(frozen=True)
class PriceGradientBindingSpec:
    pricing: ProductBindingSpec
    maximum_sensitivities: int
    closed_form: bool

    @property
    def unit_path(self):
        return self.pricing.unit_path + "_price_gradients"

    @property
    def paths(self):
        return tuple(self.unit_path + "." + suffix for suffix in ("cuh", "cu"))


def compose_bindings(pricing_bindings):
    """Expose only implemented engines; further models are added after their coupling."""
    maximum = {"black_scholes": 6, "heston": 10, "cev": 7, "merton": 7}
    return tuple(
        PriceGradientBindingSpec(binding,
                                 9 if binding.product == "american_option"
                                 else maximum[binding.model],
                                 binding.model == "black_scholes")
        for binding in pricing_bindings
        if binding.model in maximum and (
            binding.product == "european_option"
            or (binding.model == "heston"
                and binding.product == "american_option")
        )
    )


def compose_datasets(delta_datasets, bindings):
    supported = {(spec.pricing.model, spec.pricing.product) for spec in bindings}
    return tuple(replace(dataset,
        dataset_id=dataset.dataset_id.removesuffix("_price_delta") + "_price_gradients",
        dataset_kind="price_gradients",
        recipe_path=dataset.recipe_path.replace("/price_delta/", "/price_gradients/")
            .replace("_price_delta/", "_price_gradients/"),
        template="catalog/pricing/price_gradients/generator.cpp.tpl",
        numerical_profile="selected_gradients_production_paths", layout="row_major_selected_gradients")
        for dataset in delta_datasets
        if (dataset.model, dataset.product) in supported)


def default_sensitivities(model):
    """Initial recipes select useful coordinates without inferring inclusion from zero values."""
    coordinates = {
        "black_scholes": (("model.spot", .005, "relative"),
                          ("model.volatility", .005, "relative"),
                          ("product.strike", .005, "relative")),
        "heston": (("model.spot", .005, "relative"),
                   ("model.initial_variance", .001, "absolute"),
                   ("model.rho", .002, "absolute"),
                   ("product.strike", .005, "relative")),
        "cev": (("model.spot", .005, "relative"), ("model.sigma", .005, "relative"),
                ("model.beta", .002, "absolute"), ("product.strike", .005, "relative")),
        "merton": (("model.spot", .005, "relative"), ("model.volatility", .005, "relative"),
                   ("model.jump_log_mean", .002, "absolute"),
                   ("product.strike", .005, "relative")),
    }[model]
    return [{"parameter": name, "displacement": h, "scale": scale,
             "boundary": "central_then_one_sided_order2"} for name, h, scale in coordinates]
