"""Declarative, exhaustively inventoried validation coverage for Kou prices."""

from functools import partial

from validation.model.equity.stochastic_equity import (
    COMMON_PRODUCT_KINDS,
    EquityValidationSpec,
    PremiaMethodCandidate,
    run_product_validation_cli as run_validation_cli,
    validate_dataset as validate_with_spec,
)
from validation.premia.model.equity.kou.specialized_option import (
    validation_from_premia_kou_specialized_option,
)


def _terminal_candidate(
    pricing_method: str,
    *,
    vanilla: str | None = None,
    digital: str | None = None,
) -> PremiaMethodCandidate:
    return PremiaMethodCandidate(
        pricing_method,
        vanilla_method=vanilla,
        digital_method=digital,
    )


def _path_candidate(
    method: str,
    *,
    pricing_method: str | None = None,
    asian_put_via_call_parity: bool = True,
) -> PremiaMethodCandidate:
    return PremiaMethodCandidate(
        pricing_method or method,
        path_method=method,
        asian_put_via_call_parity=asian_put_via_call_parity,
    )


def _specialized_candidate(
    pricing_method: str,
    *,
    vanilla: str = "AP_Carr_Kou",
    digital: str = "AP_Kou_Eu",
    barrier: str = "AP_Kou_Barrier_In",
) -> PremiaMethodCandidate:
    return PremiaMethodCandidate(
        pricing_method,
        validator=partial(
            validation_from_premia_kou_specialized_option,
            vanilla_method=vanilla,
            digital_method=digital,
            barrier_method=barrier,
        ),
    )


_VANILLA_METHODS = (
    ("AP_Carr_Kou", "AP_Carr_Kou"),
    ("AP_Kou_Eu", "AP_Kou_Eu"),
    ("FD_ImpExp", "FD_ImpExp"),
    ("MC_Kou", "MC_Kou"),
)
_DIGITAL_METHODS = (
    ("AP_Kou_Eu", "AP_Kou_Eu"),
    ("MC_Kou_Digital_LRM", "MC_Kou_Digital_LRM"),
)


_PREMIA_CANDIDATES: dict[str, tuple[PremiaMethodCandidate, ...]] = {}
for _product in ("european_call", "european_put", "straddle"):
    _PREMIA_CANDIDATES[_product] = tuple(
        _terminal_candidate(label, vanilla=method)
        for label, method in _VANILLA_METHODS
    )
for _product in ("digital_call", "digital_put"):
    _PREMIA_CANDIDATES[_product] = tuple(
        _terminal_candidate(label, digital=method)
        for label, method in _DIGITAL_METHODS
    )
for _product in (
    "asset_or_nothing_call",
    "asset_or_nothing_put",
    "gap_call",
    "gap_put",
):
    _PREMIA_CANDIDATES[_product] = (
        _terminal_candidate(
            "AP_Carr_Kou + AP_Kou_Eu (exact static replication)",
            vanilla="AP_Carr_Kou",
            digital="AP_Kou_Eu",
        ),
        _terminal_candidate(
            "AP_Kou_Eu vanilla + digital (exact static replication)",
            vanilla="AP_Kou_Eu",
            digital="AP_Kou_Eu",
        ),
        _terminal_candidate(
            "FD_ImpExp + AP_Kou_Eu (exact static replication)",
            vanilla="FD_ImpExp",
            digital="AP_Kou_Eu",
        ),
        _terminal_candidate(
            "MC_Kou + MC_Kou_Digital_LRM (exact static replication)",
            vanilla="MC_Kou",
            digital="MC_Kou_Digital_LRM",
        ),
    )

_PREMIA_CANDIDATES.update(
    {
        "asian_call": (
            _path_candidate("MC_FixedAsian_IS_Lelong"),
            _path_candidate("AP_FixedAsian_FusaiMeucci_Kou"),
            _path_candidate("AP_Asian_FMM_KOU"),
        ),
        "asian_put": (
            _path_candidate(
                "MC_FixedAsian_IS_Lelong",
                pricing_method=(
                    "MC_FixedAsian_IS_Lelong call + exact arithmetic-average "
                    "put-call parity"
                ),
            ),
            _path_candidate(
                "AP_FixedAsian_FusaiMeucci_Kou",
                asian_put_via_call_parity=False,
            ),
            _path_candidate(
                "AP_Asian_FMM_KOU", asian_put_via_call_parity=False
            ),
        ),
        "lookback_option": (
            _path_candidate("AP_Kou_LookbackFixed"),
            _path_candidate("FFT_Kou_LookbackFixed"),
            _path_candidate("AP_FastWH_Kou"),
            _path_candidate("MC_Kou_LookbackFixed"),
        ),
        "up_and_out_call": (
            _path_candidate("AP_FastWHBar_Kou"),
            _path_candidate("AP_Kou_Barrier_Out"),
            _path_candidate("FD_ImpExpUpOut"),
            _path_candidate("MC_WHBar_Kou"),
            _path_candidate("MC_Kou_Out_LRM"),
        ),
        "down_and_out_put": (
            _path_candidate("AP_FastWHBar_Kou"),
            _path_candidate("AP_Kou_Barrier_Out"),
            _path_candidate("FD_ImpExpDownOut"),
            _path_candidate("MC_WHBar_Kou"),
            _path_candidate("MC_Kou_Out_LRM"),
        ),
        "up_and_in_call": (
            _path_candidate("AP_Kou_Barrier_In"),
            _path_candidate("MC_Kou_In_LRM"),
        ),
        "down_and_in_put": (
            _path_candidate("AP_Kou_Barrier_In"),
            _path_candidate("MC_Kou_In_LRM"),
        ),
    }
)

for _product in ("forward_start_call", "forward_start_put"):
    _PREMIA_CANDIDATES[_product] = tuple(
        _specialized_candidate(
            f"{method} (exact forward-start reduction)",
            vanilla=method,
        )
        for _, method in _VANILLA_METHODS
    )
_PREMIA_CANDIDATES["range_accrual"] = tuple(
    _specialized_candidate(
        f"{method} digitals (exact marginal-probability reduction)",
        digital=method,
    )
    for _, method in _DIGITAL_METHODS
)
for _product in ("up_no_touch", "up_one_touch"):
    _PREMIA_CANDIDATES[_product] = (
        _specialized_candidate(
            "AP_Kou_Barrier_In + AP_Carr_Kou (exact rebate reduction)"
        ),
        _specialized_candidate(
            "MC_Kou_In_LRM + AP_Carr_Kou (exact rebate reduction)",
            barrier="MC_Kou_In_LRM",
        ),
    )


_PREMIA_PRODUCTS = frozenset(_PREMIA_CANDIDATES)
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
    product: candidates[0].pricing_method
    for product, candidates in _PREMIA_CANDIDATES.items()
}

_BIAS_EXPLANATIONS = {
    "lookback_option": (
        "expected; a discrete maximum cannot exceed the continuous-time "
        "reference maximum"
    ),
    "up_and_out_call": (
        "expected; discrete monitoring knocks out fewer paths than continuous "
        "monitoring"
    ),
    "up_and_in_call": (
        "expected; discrete monitoring activates fewer paths than continuous "
        "monitoring"
    ),
    "down_and_out_put": (
        "expected; discrete monitoring knocks out fewer paths than continuous "
        "monitoring"
    ),
    "down_and_in_put": (
        "expected; discrete monitoring activates fewer paths than continuous "
        "monitoring"
    ),
    "up_no_touch": (
        "expected; discrete monitoring misses some continuous upper-barrier hits"
    ),
    "up_one_touch": (
        "expected; discrete monitoring misses some continuous upper-barrier hits"
    ),
}


SPEC = EquityValidationSpec(
    model_name="kou",
    product_kinds=COMMON_PRODUCT_KINDS,
    premia_products=_PREMIA_PRODUCTS,
    premia_methods=_PREMIA_METHODS,
    premia_method_candidates=_PREMIA_CANDIDATES,
    premia_path_products=_PREMIA_PATH_PRODUCTS,
    bias_explanations=_BIAS_EXPLANATIONS,
    quantlib_monte_carlo_unavailable_reason=(
        "QuantLib exposes no Kou process for an independent Monte Carlo engine"
    ),
    enforce_directional_bias=True,
    enforce_statistical_bias=False,
    near_zero_relative_materiality=2.0e-3,
)


def validate_dataset(path, product_kind):
    """Validate one Kou dataset through every compatible Premia candidate."""

    return validate_with_spec(path, SPEC, product_kind)


def run_product_validation_cli(product_kind):
    """Run one thin Kou product validator."""

    return run_validation_cli(SPEC, product_kind)
