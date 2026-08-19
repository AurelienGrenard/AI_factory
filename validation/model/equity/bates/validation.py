"""Declarative validation coverage for every Bates price dataset."""

from validation.model.equity.stochastic_equity import (
    COMMON_PRODUCT_KINDS,
    EquityValidationSpec,
    run_product_validation_cli as run_validation_cli,
    validate_dataset as validate_with_spec,
)
from validation.quantlib.model.equity.bates.equity_option import (
    validation_from_quantlib_bates_option,
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
    "european_call": "CF_Call_MerHes",
    "european_put": "CF_Put_MerHes",
    "straddle": "CF_Call_MerHes + CF_Put_MerHes",
    "american_call": (
        "MC_AM_Alfonsi_LongstaffSchwartz_Bates "
        "(4,096 paths; 64 exercise dates)"
    ),
    "american_put": (
        "MC_AM_Alfonsi_LongstaffSchwartz_Bates "
        "(4,096 paths; 64 exercise dates)"
    ),
    "asian_call": "MC_Alfonsi_Asian_Bates (4,096 paths; 256 steps)",
    "asian_put": "MC_Alfonsi_Asian_Bates (4,096 paths; 256 steps)",
    "up_and_out_call": "MC_Alfonsi_Bates_Out (4,096 paths; 256 steps)",
    "up_and_in_call": (
        "CF_Call_MerHes - MC_Alfonsi_Bates_Out "
        "(in/out parity; 4,096 paths; 256 steps)"
    ),
    "down_and_out_put": "MC_Alfonsi_Bates_Out (4,096 paths; 256 steps)",
    "down_and_in_put": (
        "CF_Put_MerHes - MC_Alfonsi_Bates_Out "
        "(in/out parity; 4,096 paths; 256 steps)"
    ),
}
_BIAS_EXPLANATIONS = {
    "american_call": "expected; a continuously exercisable American option dominates its Bermudan approximation",
    "american_put": "expected; a continuously exercisable American option dominates its Bermudan approximation",
    "asian_call": "expected; CUDA and Premia use independent daily-grid Monte Carlo schemes",
    "asian_put": "expected; CUDA and Premia use independent daily-grid Monte Carlo schemes",
    "up_and_out_call": "expected; CUDA and Premia use independent discrete-grid Monte Carlo schemes",
    "up_and_in_call": "expected; CUDA and Premia use independent discrete-grid Monte Carlo schemes",
    "down_and_out_put": "expected; CUDA and Premia use independent discrete-grid Monte Carlo schemes",
    "down_and_in_put": "expected; CUDA and Premia use independent discrete-grid Monte Carlo schemes",
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
    "european_call": "BatesEngine",
    "european_put": "BatesEngine",
    "straddle": "BatesEngine (call + put)",
    "digital_call": "BatesEngine (central strike differentiation)",
    "digital_put": "BatesEngine (central strike differentiation)",
    "asset_or_nothing_call": (
        "BatesEngine + central strike differentiation (static replication)"
    ),
    "asset_or_nothing_put": (
        "BatesEngine + central strike differentiation (static replication)"
    ),
    "gap_call": (
        "BatesEngine + central strike differentiation (static replication)"
    ),
    "gap_put": (
        "BatesEngine + central strike differentiation (static replication)"
    ),
    "american_call": "FdBatesVanillaEngine",
    "american_put": "FdBatesVanillaEngine",
}
_MONTE_CARLO_PRODUCTS = _PRODUCTS - _SPECIALIZED_PRODUCTS
_MONTE_CARLO_METHODS = {
    product: "_antithetic_bates_path_price (GaussianMultiPathGenerator)"
    for product in _MONTE_CARLO_PRODUCTS
}


SPEC = EquityValidationSpec(
    model_name="bates",
    product_kinds=frozenset(_PRODUCTS),
    premia_products=_PREMIA_TERMINAL_PRODUCTS | _PREMIA_PATH_PRODUCTS,
    premia_methods=_PREMIA_METHODS,
    premia_path_products=_PREMIA_PATH_PRODUCTS,
    bias_explanations=_BIAS_EXPLANATIONS,
    quantlib_specialized_products=_SPECIALIZED_PRODUCTS,
    quantlib_specialized_methods=_SPECIALIZED_METHODS,
    quantlib_monte_carlo_products=frozenset(_MONTE_CARLO_PRODUCTS),
    quantlib_monte_carlo_methods=_MONTE_CARLO_METHODS,
    quantlib_validator=validation_from_quantlib_bates_option,
)


def validate_dataset(path, product_kind):
    """Validate one Bates dataset through the common hierarchy."""

    return validate_with_spec(path, SPEC, product_kind)


def run_product_validation_cli(product_kind):
    """Run one thin Bates product validator."""

    return run_validation_cli(SPEC, product_kind)
