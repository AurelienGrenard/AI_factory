"""Declarative validation coverage for every Heston price dataset."""

from validation.model.equity.stochastic_equity import (
    COMMON_PRODUCT_KINDS,
    EquityValidationSpec,
    run_product_validation_cli as run_validation_cli,
    validate_dataset as validate_with_spec,
)
from validation.quantlib.model.equity.heston.equity_option import (
    validation_from_quantlib_heston_option,
)


_PRODUCTS = COMMON_PRODUCT_KINDS | {"american_call", "american_put"}
_PREMIA_TERMINAL_PRODUCTS = frozenset(
    {"european_call", "european_put", "straddle"}
)
_PREMIA_PATH_PRODUCTS = frozenset(
    {
        "american_call",
        "american_put",
        "asian_call",
        "asian_put",
        "up_and_out_call",
        "up_and_in_call",
        "down_and_out_put",
        "down_and_in_put",
    }
)
_PREMIA_METHODS = {
    "european_call": "CF_Call_Heston",
    "european_put": "CF_Put_Heston",
    "straddle": "CF_Call_Heston + CF_Put_Heston",
    "american_call": "AP_FastWHAmer_HES (256 time steps)",
    "american_put": "AP_FastWHAmer_HES (256 time steps)",
    "asian_call": "AP_FJM_ASIAN_HESTON",
    "asian_put": "AP_FJM_ASIAN_HESTON (continuous-average put-call parity)",
    "up_and_out_call": "AP_FastWHBar_HES (256 time steps)",
    "up_and_in_call": (
        "CF_Call_Heston - AP_FastWHBar_HES (in/out parity; 256 time steps)"
    ),
    "down_and_out_put": "AP_FastWHBar_HES (256 time steps)",
    "down_and_in_put": (
        "CF_Put_Heston - AP_FastWHBar_HES (in/out parity; 256 time steps)"
    ),
}
_BIAS_EXPLANATIONS = {
    "american_call": "expected; a continuously exercisable American option dominates its Bermudan approximation",
    "american_put": "expected; a continuously exercisable American option dominates its Bermudan approximation",
    "asian_call": "expected; CUDA uses daily discrete averaging while Premia uses a continuous-average approximation",
    "asian_put": "expected; CUDA uses daily discrete averaging while Premia uses a continuous-average approximation",
    "up_and_out_call": "expected; discrete monitoring knocks out fewer paths than continuous monitoring",
    "up_and_in_call": "expected; discrete monitoring activates fewer paths than continuous monitoring",
    "down_and_out_put": "expected; discrete monitoring knocks out fewer paths than continuous monitoring",
    "down_and_in_put": "expected; discrete monitoring activates fewer paths than continuous monitoring",
}
_SPECIALIZED_PRODUCTS = frozenset(
    {
        "american_call",
        "american_put",
        "asset_or_nothing_call",
        "asset_or_nothing_put",
        "digital_call",
        "digital_put",
        "european_call",
        "european_put",
        "gap_call",
        "gap_put",
        "straddle",
    }
)
_SPECIALIZED_METHODS = {
    "european_call": "AnalyticHestonEngine",
    "european_put": "AnalyticHestonEngine",
    "straddle": "AnalyticHestonEngine (call + put)",
    "digital_call": "HestonRNDCalculator.cdf",
    "digital_put": "HestonRNDCalculator.cdf",
    "asset_or_nothing_call": (
        "AnalyticHestonEngine + HestonRNDCalculator.cdf (static replication)"
    ),
    "asset_or_nothing_put": (
        "AnalyticHestonEngine + HestonRNDCalculator.cdf (static replication)"
    ),
    "gap_call": (
        "AnalyticHestonEngine + HestonRNDCalculator.cdf (static replication)"
    ),
    "gap_put": (
        "AnalyticHestonEngine + HestonRNDCalculator.cdf (static replication)"
    ),
    "american_call": "FdHestonVanillaEngine",
    "american_put": "FdHestonVanillaEngine",
}
_MONTE_CARLO_PRODUCTS = _PRODUCTS - _SPECIALIZED_PRODUCTS
_MONTE_CARLO_METHODS = {
    product: "_antithetic_heston_path_price (GaussianMultiPathGenerator)"
    for product in _MONTE_CARLO_PRODUCTS
}
_MONTE_CARLO_METHODS.update(
    {
        "asian_call": "MCDiscreteArithmeticAPHestonEngine",
        "asian_put": "MCDiscreteArithmeticAPHestonEngine",
        "forward_start_call": "MCForwardEuropeanHestonEngine",
        "forward_start_put": "MCForwardEuropeanHestonEngine",
    }
)


SPEC = EquityValidationSpec(
    model_name="heston",
    product_kinds=frozenset(_PRODUCTS),
    premia_products=_PREMIA_TERMINAL_PRODUCTS | _PREMIA_PATH_PRODUCTS,
    premia_methods=_PREMIA_METHODS,
    premia_path_products=_PREMIA_PATH_PRODUCTS,
    bias_explanations=_BIAS_EXPLANATIONS,
    quantlib_specialized_products=_SPECIALIZED_PRODUCTS,
    quantlib_specialized_methods=_SPECIALIZED_METHODS,
    quantlib_monte_carlo_products=frozenset(_MONTE_CARLO_PRODUCTS),
    quantlib_monte_carlo_methods=_MONTE_CARLO_METHODS,
    quantlib_validator=validation_from_quantlib_heston_option,
)


def validate_dataset(path, product_kind):
    """Validate one Heston dataset through the common hierarchy."""

    return validate_with_spec(path, SPEC, product_kind)


def run_product_validation_cli(product_kind):
    """Run one thin Heston product validator."""

    return run_validation_cli(SPEC, product_kind)
