"""Equity strategy decisions for diagonal Hessians and first-order gradients.

The diagonal matrix uses SM89 Kmax measurements on four baseline products and
independent holdouts. The first-order map uses PPTI V100 SM70 measurements at
2**20 paths per aligned row. Unmeasured bindings inherit a decision from a
measured payoff family, with provenance returned by the decision functions.
The diagonal geometry remains available in generated CUDA even when a
first-order recipe selects mono.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyDecision:
    strategy: str
    graph_threads_per_path: int | None
    provenance: str
    reference_products: tuple[str, ...]


# Order: European call, Asian call, up-and-out call, Phoenix autocall.
# None means mono; the Black-Scholes European call is analytic.
_BASELINE = {
    "bates": (4, None, None, 4),
    "black_scholes": (None, None, None, None),
    "cev": (None, None, 2, 2),
    "heston": (4, 2, None, 2),
    "heston_3_2": (2, 2, 2, 2),
    "kou": (None, None, None, None),
    "merton": (None, None, None, None),
    "normal_inverse_gaussian": (None, 2, 2, None),
    "sabr": (None, 2, 2, 2),
    "schobel_zhu": (2, 2, 2, 2),
    "stein_stein": (2, 2, 2, 2),
    "variance_gamma": (None, 2, 2, None),
}
_BASELINE_PRODUCTS = (
    "european_option", "asian_option", "up_and_out_option",
    "phoenix_autocall",
)

# Independent seeded holdout, two paired repetitions per binding.
_HOLDOUT = {
    ("bates", "geometric_asian_option"): None,
    ("black_scholes", "double_knock_out_option"): None,
    ("cev", "cliquet"): None,
    ("heston", "phoenix_memory_autocall"): 2,
    ("heston_3_2", "forward_start_option"): 2,
    ("kou", "forward_start_option"): None,
    ("normal_inverse_gaussian", "geometric_asian_option"): 2,
    ("stein_stein", "gap_option"): 2,
}

_TERMINAL = {
    "asset_or_nothing_option", "digital_option", "gap_option", "straddle",
}
_BARRIER = {
    "double_knock_out_option", "down_and_in_option", "down_and_out_option",
    "up_and_in_option", "up_no_touch", "up_one_touch",
}
_AUTOCALL = {"athena_autocall", "phoenix_memory_autocall"}
_ALL_PRODUCTS = {
    *_BASELINE_PRODUCTS, *_TERMINAL, *_BARRIER, *_AUTOCALL,
    "american_option", "cliquet", "forward_start_option",
    "geometric_asian_option", "lookback_option", "range_accrual",
}


def covers(model: str) -> bool:
    return model in _BASELINE


def _result(group: int | None, provenance: str,
            references: tuple[str, ...]) -> StrategyDecision:
    return StrategyDecision("node_graph" if group is not None else "mono",
                            group, provenance, references)


def decision_for(model: str, product: str) -> StrategyDecision:
    """Return a measured decision or a documented family extrapolation."""
    if model not in _BASELINE or product not in _ALL_PRODUCTS:
        raise KeyError((model, product))
    if product == "american_option":
        return _result(None, "lsm_separate_fallback", ())
    if model == "black_scholes" and product == "european_option":
        return StrategyDecision("analytic", None, "measured_analytic", ())
    baseline = dict(zip(_BASELINE_PRODUCTS, _BASELINE[model]))
    if product in baseline:
        return _result(baseline[product], "measured_baseline", (product,))
    if (model, product) in _HOLDOUT:
        return _result(_HOLDOUT[(model, product)], "measured_holdout", (product,))
    if product in _TERMINAL:
        return _result(baseline["european_option"], "inferred_terminal",
                       ("european_option",))
    if product == "geometric_asian_option":
        return _result(baseline["asian_option"], "inferred_average",
                       ("asian_option",))
    if product in _BARRIER:
        return _result(baseline["up_and_out_option"], "inferred_barrier",
                       ("up_and_out_option",))
    if product in _AUTOCALL:
        return _result(baseline["phoenix_autocall"], "inferred_autocall",
                       ("phoenix_autocall",))

    consensus = {
        "forward_start_option": ("european_option", "asian_option"),
        "lookback_option": ("asian_option", "up_and_out_option"),
        "cliquet": ("asian_option", "phoenix_autocall"),
        "range_accrual": ("up_and_out_option", "phoenix_autocall"),
    }
    references = consensus[product]
    groups = tuple(baseline[reference] for reference in references)
    if all(group is not None for group in groups):
        # Forward start is a path engine; take the measured path geometry.
        group = groups[1] if product == "forward_start_option" else groups[0]
        return _result(group, "inferred_consensus", references)
    return _result(None, "conservative_fallback", references)


# First-order Kmax timings on PPTI V100 (SM70). Each binding below used
# aligned model/product rows and 2**20 Philox paths per row. The performance
# audit and graph geometries are in maintainer/experiments/gradient_only/
# first_order_strategy_sm70.md. None denotes a measured mono winner.
_FIRST_ORDER_PRODUCTION: dict[tuple[str, str], int | None] = {
    ("bates", "european_option"): None,
    ("bates", "phoenix_autocall"): None,
    ("black_scholes", "asian_option"): None,
    ("cev", "european_option"): None,
    ("cev", "up_and_out_option"): None,
    ("heston", "european_option"): None,
    ("heston", "asian_option"): None,
    ("heston", "phoenix_autocall"): None,
    ("heston_3_2", "asian_option"): None,
    ("heston_3_2", "up_and_out_option"): None,
    ("kou", "european_option"): None,
    ("merton", "european_option"): None,
    ("normal_inverse_gaussian", "up_and_out_option"): None,
    ("sabr", "asian_option"): None,
    ("sabr", "up_and_out_option"): None,
    ("schobel_zhu", "european_option"): None,
    ("schobel_zhu", "up_and_out_option"): None,
    ("stein_stein", "asian_option"): None,
    ("variance_gamma", "asian_option"): None,
    ("variance_gamma", "up_and_out_option"): None,
}


def _first_order_anchor_preference(product: str) -> tuple[str, ...]:
    """Prefer a measured binding with comparable path and payoff work."""
    if product in _TERMINAL or product == "european_option":
        return ("european_option", "asian_option", "up_and_out_option",
                "phoenix_autocall")
    if product in _BARRIER or product in {
        "up_and_out_option", "range_accrual", "lookback_option",
    }:
        return ("up_and_out_option", "phoenix_autocall", "asian_option",
                "european_option")
    if product in _AUTOCALL or product == "phoenix_autocall":
        return ("phoenix_autocall", "up_and_out_option", "asian_option",
                "european_option")
    return ("asian_option", "up_and_out_option", "phoenix_autocall",
            "european_option")


def first_order_decision_for(model: str, product: str) -> StrategyDecision:
    """Choose first-order strategy from production-path-count measurements."""
    diagonal = decision_for(model, product)
    if diagonal.strategy == "analytic" or product == "american_option":
        return diagonal
    key = (model, product)
    if key in _FIRST_ORDER_PRODUCTION:
        return _result(_FIRST_ORDER_PRODUCTION[key],
                       "measured_first_order_sm70_2p20", (product,))
    for reference in _first_order_anchor_preference(product):
        anchor = (model, reference)
        if anchor in _FIRST_ORDER_PRODUCTION:
            return _result(
                _FIRST_ORDER_PRODUCTION[anchor],
                "inferred_first_order_sm70_2p20_family",
                (reference,),
            )
    # A new model requires its own timing campaign before graph is selected.
    return _result(None, "first_order_unmeasured_mono_fallback", ())
