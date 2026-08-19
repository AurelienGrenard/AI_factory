"""Declarative validation coverage for every Merton price dataset."""

from validation.model.equity.stochastic_equity import (
    COMMON_PRODUCT_KINDS,
    EquityValidationSpec,
    run_product_validation_cli as run_validation_cli,
    validate_dataset as validate_with_spec,
)


_PREMIA_TERMINAL_PRODUCTS = frozenset(
    {
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
_PREMIA_PATH_PRODUCTS = frozenset(
    {
        "asian_call",
        "asian_put",
        "lookback_option",
        "up_and_out_call",
        "up_and_in_call",
        "down_and_out_put",
        "down_and_in_put",
    }
)

_PREMIA_METHODS = {
    "european_call": "CF_Call_Merton",
    "european_put": "CF_Put_Merton",
    "straddle": "CF_Call_Merton + CF_Put_Merton",
    "digital_call": "MC_Merton",
    "digital_put": "MC_Merton (put parity)",
    "asset_or_nothing_call": (
        "CF_Call_Merton + MC_Merton (static replication)"
    ),
    "asset_or_nothing_put": (
        "CF_Put_Merton + MC_Merton (static replication)"
    ),
    "gap_call": "CF_Call_Merton + MC_Merton (static replication)",
    "gap_put": "CF_Put_Merton + MC_Merton (static replication)",
    "asian_call": "AP_Asian_FMM_Mer (52 monitoring dates; 1,024 integration points)",
    "asian_put": (
        "AP_Asian_FMM_Mer call + arithmetic-average put-call parity "
        "(52 monitoring dates; 1,024 integration points)"
    ),
    "lookback_option": "MC_Merton_FixedLookback (4,096 paths; 256 steps)",
    "up_and_out_call": "FD_ImpExpUpOut (256 time steps)",
    "up_and_in_call": (
        "CF_Call_Merton - FD_ImpExpUpOut (in/out parity; 256 time steps)"
    ),
    "down_and_out_put": "FD_ImpExpDownOut (256 time steps)",
    "down_and_in_put": (
        "CF_Put_Merton - FD_ImpExpDownOut (in/out parity; 256 time steps)"
    ),
}

_BIAS_EXPLANATIONS = {
    "asian_call": "expected; CUDA uses daily discrete averaging while Premia uses its fixed-Asian monitoring convention",
    "asian_put": "expected; CUDA uses daily discrete averaging while Premia uses its fixed-Asian monitoring convention",
    "lookback_option": "expected; CUDA and Premia use independent discrete-grid Monte Carlo schemes",
    "up_and_out_call": "expected; discrete monitoring knocks out fewer paths than continuous monitoring",
    "up_and_in_call": "expected; discrete monitoring activates fewer paths than continuous monitoring",
    "down_and_out_put": "expected; discrete monitoring knocks out fewer paths than continuous monitoring",
    "down_and_in_put": "expected; discrete monitoring activates fewer paths than continuous monitoring",
}

SPEC = EquityValidationSpec(
    model_name="merton",
    product_kinds=COMMON_PRODUCT_KINDS,
    premia_products=_PREMIA_TERMINAL_PRODUCTS | _PREMIA_PATH_PRODUCTS,
    premia_methods=_PREMIA_METHODS,
    premia_path_products=_PREMIA_PATH_PRODUCTS,
    bias_explanations=_BIAS_EXPLANATIONS,
    quantlib_monte_carlo_unavailable_reason=(
        "QuantLib Merton76Process does not implement the path evolution "
        "required by a generic Monte Carlo engine"
    ),
)


def validate_dataset(path, product_kind):
    """Validate one Merton dataset through the common hierarchy."""

    return validate_with_spec(path, SPEC, product_kind)


def run_product_validation_cli(product_kind):
    """Run one thin Merton product validator."""

    return run_validation_cli(SPEC, product_kind)
