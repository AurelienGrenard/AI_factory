"""Canonical equity model-product manifest for generated pricing bindings."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Binding:
    model: str
    model_display: str
    product: str
    product_type: str
    pricing_policy: str
    schedule: str
    time_kind: str
    sided: bool = True


@dataclass(frozen=True)
class RoughProductBinding:
    product: str
    product_type: str
    path_policy: str
    schedule_kind: str
    sided: bool = True


# One canonical product description drives every Gaussian-Volterra binding
# and both prepared N-factor rough-model lifts.
ROUGH_PRODUCT_BINDINGS = (
    RoughProductBinding(
        "asian_option", "AsianOption", "AsianOptionPathPolicy", "dense"
    ),
    RoughProductBinding(
        "asset_or_nothing_option", "AssetOrNothingOption",
        "AssetOrNothingOptionPathPolicy", "terminal"
    ),
    RoughProductBinding(
        "athena_autocall", "AthenaAutocall", "AthenaAutocallPathPolicy",
        "regular", False
    ),
    RoughProductBinding(
        "cliquet", "Cliquet", "CliquetPathPolicy", "regular", False
    ),
    RoughProductBinding(
        "digital_option", "DigitalOption", "DigitalOptionPathPolicy",
        "terminal"
    ),
    RoughProductBinding(
        "double_knock_out_option", "DoubleKnockOutOption",
        "DoubleKnockOutOptionPathPolicy", "dense"
    ),
    RoughProductBinding(
        "down_and_in_option", "DownAndInOption",
        "DownAndInOptionPathPolicy", "dense"
    ),
    RoughProductBinding(
        "down_and_out_option", "DownAndOutOption",
        "DownAndOutOptionPathPolicy", "dense"
    ),
    RoughProductBinding(
        "european_option", "EuropeanOption", "EuropeanOptionPathPolicy",
        "terminal"
    ),
    RoughProductBinding(
        "forward_start_option", "ForwardStartOption",
        "ForwardStartOptionPathPolicy", "calendar_2"
    ),
    RoughProductBinding(
        "gap_option", "GapOption", "GapOptionPathPolicy", "terminal"
    ),
    RoughProductBinding(
        "geometric_asian_option", "GeometricAsianOption",
        "GeometricAsianOptionPathPolicy", "dense"
    ),
    RoughProductBinding(
        "lookback_option", "LookbackOption", "LookbackOptionPathPolicy",
        "dense", False
    ),
    RoughProductBinding(
        "phoenix_autocall", "PhoenixAutocall", "PhoenixAutocallPathPolicy",
        "regular", False
    ),
    RoughProductBinding(
        "phoenix_memory_autocall", "PhoenixMemoryAutocall",
        "PhoenixMemoryAutocallPathPolicy", "regular", False
    ),
    RoughProductBinding(
        "range_accrual", "RangeAccrual", "RangeAccrualPathPolicy",
        "regular", False
    ),
    RoughProductBinding(
        "straddle", "Straddle", "StraddlePathPolicy", "terminal", False
    ),
    RoughProductBinding(
        "up_and_in_option", "UpAndInOption", "UpAndInOptionPathPolicy",
        "dense"
    ),
    RoughProductBinding(
        "up_and_out_option", "UpAndOutOption", "UpAndOutOptionPathPolicy",
        "dense"
    ),
    RoughProductBinding(
        "up_no_touch", "UpNoTouch", "UpNoTouchPathPolicy", "dense", False
    ),
    RoughProductBinding(
        "up_one_touch", "UpOneTouch", "UpOneTouchPathPolicy", "dense",
        False
    ),
)


# Schedule selection is product metadata crossed with a model's numerical
# contract.  Dense monitoring always uses fixed steps, including for models
# that also expose exact finite-horizon transitions.
MARKOVIAN_MODELS = (
    ("bates", "Bates", "fixed"),
    ("cev", "CEV", "fixed"),
    ("heston", "Heston", "fixed"),
    ("heston_3_2", "Heston 3/2", "fixed"),
    ("kou", "Kou", "exact"),
    ("merton", "Merton", "exact"),
    ("normal_inverse_gaussian", "Normal-Inverse-Gaussian", "exact"),
    ("sabr", "SABR", "fixed"),
    ("schobel_zhu", "Schobel-Zhu", "fixed"),
    ("stein_stein", "Stein-Stein", "fixed"),
    ("variance_gamma", "Variance-Gamma", "exact"),
)

ROUGH_VOLTERRA_MODELS = (
    ("rough_bergomi", "Rough-Bergomi"),
    ("rough_sabr", "Rough-SABR"),
    ("log_modulated_rough_bergomi", "Log-modulated rough-Bergomi"),
    ("rough_stein_stein", "Rough Stein-Stein"),
)

ROUGH_N_FACTOR_MODELS = (
    ("rough_heston", "Rough-Heston"),
    ("quadratic_rough_heston", "Quadratic rough-Heston"),
)

ROUGH_MODELS = ROUGH_VOLTERRA_MODELS + ROUGH_N_FACTOR_MODELS

# Black-Scholes keeps eight genuinely analytical non-American launch units.
# Its remaining
# path-dependent products are ordinary exact-transition Monte Carlo bindings
# and belong to the same generated matrix as the other Markovian models.
BLACK_SCHOLES_MONTE_CARLO_PRODUCTS = frozenset({
    "asian_option",
    "athena_autocall",
    "cliquet",
    "double_knock_out_option",
    "down_and_in_option",
    "down_and_out_option",
    "lookback_option",
    "phoenix_autocall",
    "phoenix_memory_autocall",
    "up_and_in_option",
    "up_and_out_option",
    "up_no_touch",
    "up_one_touch",
})

# These products retain their specialized analytical implementation, but the
# implementation units are code-generation assets just like the generic
# bindings.  American exercise is the sole equity-product exclusion.
BLACK_SCHOLES_CLOSED_FORM_PRODUCTS = tuple(
    product.product
    for product in ROUGH_PRODUCT_BINDINGS
    if product.product not in BLACK_SCHOLES_MONTE_CARLO_PRODUCTS
)


def _pricing_policy(path_policy: str) -> str:
    return path_policy.replace("PathPolicy", "PricingPolicy")


def _markovian_schedule(
    schedule_kind: str,
    model_time_kind: str,
) -> tuple[str, str]:
    if model_time_kind == "exact" and schedule_kind != "dense":
        schedules = {
            "terminal": "simulation::ExactTransitionTerminalSchedule",
            "regular": "simulation::ExactTransitionRegularSchedule",
            "calendar_2": "simulation::ExactTransitionCalendarSchedule",
        }
        return schedules[schedule_kind], "exact"
    schedules = {
        "terminal": "simulation::FixedStepTerminalSchedule",
        "dense": "simulation::FixedStepDenseSchedule",
        "regular": "simulation::FixedStepRegularSchedule",
        "calendar_2": "simulation::FixedStepCalendarSchedule",
    }
    return schedules[schedule_kind], "fixed"


BINDINGS = tuple(
    Binding(
        model=model,
        model_display=display,
        product=product.product,
        product_type=product.product_type,
        pricing_policy=_pricing_policy(product.path_policy),
        schedule=_markovian_schedule(product.schedule_kind, time_kind)[0],
        time_kind=_markovian_schedule(product.schedule_kind, time_kind)[1],
        sided=product.sided,
    )
    for model, display, time_kind in MARKOVIAN_MODELS
    for product in ROUGH_PRODUCT_BINDINGS
) + tuple(
    Binding(
        model="black_scholes",
        model_display="Black-Scholes",
        product=product.product,
        product_type=product.product_type,
        pricing_policy=_pricing_policy(product.path_policy),
        schedule=_markovian_schedule(product.schedule_kind, "exact")[0],
        time_kind=_markovian_schedule(product.schedule_kind, "exact")[1],
        sided=product.sided,
    )
    for product in ROUGH_PRODUCT_BINDINGS
    if product.product in BLACK_SCHOLES_MONTE_CARLO_PRODUCTS
)
