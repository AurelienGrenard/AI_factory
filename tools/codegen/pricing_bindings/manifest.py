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


@dataclass(frozen=True)
class PriceVariant:
    name: str
    product: str
    product_dataset_folder: str
    product_dataset_id: str
    product_loader: str
    side: str | None = None
    legacy_url_name: str | None = None
    threads_per_block: int | None = None
    side_aware_loader: bool = False
    analytical_steps_per_day: int | None = None


@dataclass(frozen=True)
class ModelRecipeSpec:
    name: str
    display: str
    backend: str
    numerical_method: str
    legacy_url_name: str | None = None
    threads_per_block: int = 512


@dataclass(frozen=True)
class AmericanRecipeSpec:
    model: str
    model_display: str
    time_kind: str
    numerical_method: str
    regression_basis: str
    basis_state: tuple[str, ...]
    basis_normalization: tuple[str, ...]
    basis_functions: tuple[str, ...]
    regression_precision: str = "FP64 normal equations and Cholesky on GPU"


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


# Catalog publication is deliberately distinct from launcher support.  The
# directional barriers expose call/put launchers, while the current public
# catalog publishes only the conventional up-call and down-put variants.
PRICE_VARIANTS = (
    PriceVariant(
        "asian_calls", "asian_option", "asian_options", "asian_options_01",
        "load_asian_options", "call", "AsianCall"
    ),
    PriceVariant(
        "asian_puts", "asian_option", "asian_options", "asian_options_01",
        "load_asian_options", "put"
    ),
    PriceVariant(
        "asset_or_nothing_calls", "asset_or_nothing_option",
        "asset_or_nothing_options", "asset_or_nothing_options_01",
        "load_asset_or_nothing_options", "call"
    ),
    PriceVariant(
        "asset_or_nothing_puts", "asset_or_nothing_option",
        "asset_or_nothing_options", "asset_or_nothing_options_01",
        "load_asset_or_nothing_options", "put"
    ),
    PriceVariant(
        "athena_autocalls", "athena_autocall", "athena_autocalls",
        "athena_autocalls_01", "load_athena_autocalls", None,
        "AthenaAutocall"
    ),
    PriceVariant(
        "cliquets", "cliquet", "cliquets", "cliquets_01",
        "load_cliquets", None, "Cliquet", threads_per_block=256
    ),
    PriceVariant(
        "digital_calls", "digital_option", "digital_options",
        "digital_options_01", "load_digital_options", "call"
    ),
    PriceVariant(
        "digital_puts", "digital_option", "digital_options",
        "digital_options_01", "load_digital_options", "put"
    ),
    PriceVariant(
        "double_knock_out_calls", "double_knock_out_option",
        "double_knock_out_options", "double_knock_out_options_01",
        "load_double_knock_out_options", "call", "DoubleKnockOutCall"
    ),
    PriceVariant(
        "double_knock_out_puts", "double_knock_out_option",
        "double_knock_out_options", "double_knock_out_options_01",
        "load_double_knock_out_options", "put"
    ),
    PriceVariant(
        "down_and_in_puts", "down_and_in_option", "down_and_in_options",
        "down_and_in_options_01", "load_down_and_in_options", "put"
    ),
    PriceVariant(
        "down_and_out_puts", "down_and_out_option", "down_and_out_options",
        "down_and_out_options_01", "load_down_and_out_options", "put"
    ),
    PriceVariant(
        "european_calls", "european_option", "european_options",
        "european_options_01", "load_european_options", "call",
        "EuropeanCall"
    ),
    PriceVariant(
        "european_puts", "european_option", "european_options",
        "european_options_01", "load_european_options", "put"
    ),
    PriceVariant(
        "forward_start_calls", "forward_start_option",
        "forward_start_options", "forward_start_options_01",
        "load_forward_start_options", "call", "ForwardStartCall"
    ),
    PriceVariant(
        "forward_start_puts", "forward_start_option",
        "forward_start_options", "forward_start_options_01",
        "load_forward_start_options", "put"
    ),
    PriceVariant(
        "gap_calls", "gap_option", "gap_options", "gap_call_options_01",
        "load_gap_options", "call", side_aware_loader=True
    ),
    PriceVariant(
        "gap_puts", "gap_option", "gap_options", "gap_put_options_01",
        "load_gap_options", "put", side_aware_loader=True
    ),
    PriceVariant(
        "geometric_asian_calls", "geometric_asian_option",
        "geometric_asian_options", "geometric_asian_options_01",
        "load_geometric_asian_options", "call", "GeometricAsianCall",
        analytical_steps_per_day=2
    ),
    PriceVariant(
        "geometric_asian_puts", "geometric_asian_option",
        "geometric_asian_options", "geometric_asian_options_01",
        "load_geometric_asian_options", "put", analytical_steps_per_day=2
    ),
    PriceVariant(
        "lookback_options", "lookback_option", "lookback_options",
        "lookback_options_01", "load_lookback_options", None,
        "LookbackOption"
    ),
    PriceVariant(
        "phoenix_autocalls", "phoenix_autocall", "phoenix_autocalls",
        "phoenix_autocalls_01", "load_phoenix_autocalls", None,
        "PhoenixAutocall"
    ),
    PriceVariant(
        "phoenix_memory_autocalls", "phoenix_memory_autocall",
        "phoenix_memory_autocalls", "phoenix_memory_autocalls_01",
        "load_phoenix_memory_autocalls", None, "PhoenixMemoryAutocall"
    ),
    PriceVariant(
        "range_accruals", "range_accrual", "range_accruals",
        "range_accruals_01", "load_range_accruals", None, "RangeAccrual"
    ),
    PriceVariant(
        "straddles", "straddle", "straddles", "straddles_01",
        "load_straddles"
    ),
    PriceVariant(
        "up_and_in_calls", "up_and_in_option", "up_and_in_options",
        "up_and_in_options_01", "load_up_and_in_options", "call",
        "UpAndInCall"
    ),
    PriceVariant(
        "up_and_out_calls", "up_and_out_option", "up_and_out_options",
        "up_and_out_options_01", "load_up_and_out_options", "call",
        "UpAndOutCall"
    ),
    PriceVariant(
        "up_no_touches", "up_no_touch", "up_no_touches",
        "up_no_touches_01", "load_up_no_touches", None, "UpNoTouch"
    ),
    PriceVariant(
        "up_one_touches", "up_one_touch", "up_one_touches",
        "up_one_touches_01", "load_up_one_touches", None, "UpOneTouch"
    ),
)


MODEL_RECIPE_SPECS = (
    ModelRecipeSpec(
        "bates", "Bates", "markovian",
        "Andersen QE-M with compound-Poisson lognormal jumps", "Bates"
    ),
    ModelRecipeSpec(
        "black_scholes", "Black-Scholes", "markovian",
        "Exact Gaussian log-price transitions", "BlackScholes"
    ),
    ModelRecipeSpec(
        "cev", "CEV", "markovian", "absorbed Milstein", "CEV"
    ),
    ModelRecipeSpec(
        "heston", "Heston", "markovian", "Andersen QE-M", "Heston"
    ),
    ModelRecipeSpec(
        "heston_3_2", "Heston 3/2", "markovian",
        "full-truncation Euler 3/2 variance"
    ),
    ModelRecipeSpec(
        "kou", "Kou", "markovian", "Exact Kou increments", "Kou"
    ),
    ModelRecipeSpec(
        "merton", "Merton", "markovian", "Exact Merton increments",
        "Merton"
    ),
    ModelRecipeSpec(
        "normal_inverse_gaussian", "Normal-Inverse-Gaussian", "markovian",
        "Exact inverse-Gaussian subordination", "NormalInverseGaussian"
    ),
    ModelRecipeSpec(
        "sabr", "SABR", "markovian", "Lamperti SABR Euler"
    ),
    ModelRecipeSpec(
        "schobel_zhu", "Schobel-Zhu", "markovian",
        "exact OU factor with log-spot Euler", "SchobelZhu"
    ),
    ModelRecipeSpec(
        "stein_stein", "Stein-Stein", "markovian",
        "exact OU volatility with log-spot Euler"
    ),
    ModelRecipeSpec(
        "variance_gamma", "Variance-Gamma", "markovian",
        "Exact Gamma subordination", "VarianceGamma"
    ),
    ModelRecipeSpec(
        "rough_bergomi", "Rough-Bergomi", "volterra",
        "Bennedsen-Lunde-Pakkanen hybrid FFT (kappa=1)"
    ),
    ModelRecipeSpec(
        "rough_sabr", "Rough-SABR", "volterra",
        "Bennedsen-Lunde-Pakkanen hybrid FFT with Lamperti spot"
    ),
    ModelRecipeSpec(
        "log_modulated_rough_bergomi", "Log-modulated rough-Bergomi",
        "volterra", "log-modulated hybrid FFT (kappa=1)"
    ),
    ModelRecipeSpec(
        "rough_stein_stein", "Rough Stein-Stein", "volterra",
        "fractional-resolvent hybrid FFT"
    ),
    ModelRecipeSpec(
        "rough_heston", "Rough-Heston", "n_factor",
        "7-factor Markovian lift", threads_per_block=256
    ),
    ModelRecipeSpec(
        "quadratic_rough_heston", "Quadratic rough-Heston", "n_factor",
        "7-factor Markovian lift", threads_per_block=256
    ),
)


AMERICAN_RECIPE_SPECS = (
    AmericanRecipeSpec(
        "bates",
        "Bates",
        "fixed",
        "Andersen QE-M with compound-Poisson lognormal jumps",
        "Two-factor Laguerre degree 2",
        ("spot", "instantaneous_variance"),
        ("spot / strike", "variance / theta"),
        (
            "1",
            "L1(spot / strike)",
            "L2(spot / strike)",
            "variance / theta",
            "(variance / theta)^2",
            "L1(spot / strike) * variance / theta",
        ),
    ),
    AmericanRecipeSpec(
        "heston",
        "Heston",
        "fixed",
        "Andersen QE-M",
        "Two-factor Laguerre degree 2",
        ("spot", "instantaneous_variance"),
        ("spot / strike", "variance / theta"),
        (
            "1",
            "L1(spot / strike)",
            "L2(spot / strike)",
            "variance / theta",
            "(variance / theta)^2",
            "L1(spot / strike) * variance / theta",
        ),
    ),
    AmericanRecipeSpec(
        "normal_inverse_gaussian",
        "Normal-Inverse-Gaussian",
        "exact",
        "Exact inverse-Gaussian subordination",
        "Spot and log-moneyness six-term basis",
        ("spot", "log_moneyness"),
        ("spot / strike", "log(spot / strike)"),
        (
            "1",
            "L1(spot / strike)",
            "L2(spot / strike)",
            "log(spot / strike)",
            "log(spot / strike)^2",
            "L1(spot / strike) * log(spot / strike)",
        ),
    ),
    AmericanRecipeSpec(
        "variance_gamma",
        "Variance-Gamma",
        "exact",
        "Exact Gamma subordination",
        "Spot and log-moneyness six-term basis",
        ("spot", "log_moneyness"),
        ("spot / strike", "log(spot / strike)"),
        (
            "1",
            "L1(spot / strike)",
            "L2(spot / strike)",
            "log(spot / strike)",
            "log(spot / strike)^2",
            "L1(spot / strike) * log(spot / strike)",
        ),
    ),
)
