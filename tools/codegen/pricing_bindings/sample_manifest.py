"""Typed manifest for model-only sample bindings and catalog recipes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SampleModelSpec:
    name: str
    display: str
    asset_class: str
    family: str
    backend: str
    time_kind: str
    observation: str
    outputs: tuple[str, ...]
    parameters: tuple[tuple[str, str], ...]
    uniforms: tuple[tuple[str, float, float], ...]
    constructor: str
    derived: str = ""
    acceptance: str = "true"
    kernel: str | None = None
    extra_dynamics_include: str = ""
    pricing_numerical_method: str = ""
    legacy_url_name: str | None = None
    threads_per_block: int = 512
    supported_architectures: tuple[str, ...] = ("sm75", "sm86", "sm89")
    analytics_contract: str = "none"

    @property
    def source_folder(self) -> str:
        if self.asset_class == "fixed_income":
            return f"fixed_income/{self.name}"
        return f"equity/{self.family}/{self.name}"


def u(name: str, minimum: float, maximum: float) -> tuple[str, float, float]:
    return (name, minimum, maximum)


EQUITY_COMMON = (
    u("spot", 1.0, 1.0),
    u("risk_free_rate", 0.001, 0.08),
    u("dividend_yield", 0.0, 0.06),
)


SAMPLE_MODELS = (
    SampleModelSpec(
        "bates", "Bates", "equity", "markovian", "markovian", "fixed",
        "spot_state:variance", ("spot", "variance"),
        (("spot", "spot"), ("risk_free_rate", "risk_free_rate"),
         ("dividend_yield", "dividend_yield"),
         ("initial_variance", "initial_variance"), ("kappa", "kappa"),
         ("theta", "theta"), ("gamma", "gamma"), ("rho", "rho"),
         ("jump_intensity", "jump_intensity"),
         ("jump_log_mean", "jump_log_mean"),
         ("jump_log_volatility", "jump_log_volatility")),
        EQUITY_COMMON + (u("initial_variance", .01, .12), u("kappa", .5, 4.),
        u("theta", .01, .15), u("rho", -.95, -.25),
        u("jump_intensity", .02, 1.), u("jump_log_mean", -.25, .05),
        u("jump_log_volatility", .05, .35)),
        "{spot, risk_free_rate, dividend_yield, initial_variance, kappa, "
        "theta, gamma, rho, jump_intensity, jump_log_mean, "
        "jump_log_volatility}",
        "const float gamma = uniform({std::max(std::sqrt(kappa * theta / 5.0f), 0.1f), std::min(std::sqrt(12.0f * kappa * theta), 0.8f)}, uniforms);",
        pricing_numerical_method=(
            "Andersen QE-M with compound-Poisson lognormal jumps"
        ),
        legacy_url_name="Bates",
    ),
    SampleModelSpec(
        "black_scholes", "Black-Scholes", "equity", "markovian",
        "markovian", "exact", "spot", ("spot",),
        (("spot", "spot"), ("risk_free_rate", "risk_free_rate"),
         ("dividend_yield", "dividend_yield"), ("volatility", "volatility")),
        EQUITY_COMMON + (u("volatility", .08, .45),),
        "{spot, risk_free_rate, dividend_yield, volatility}",
        pricing_numerical_method="Exact Gaussian log-price transitions",
        legacy_url_name="BlackScholes",
        analytics_contract="Black-Scholes analytical provider",
    ),
    SampleModelSpec(
        "cev", "CEV", "equity", "markovian", "markovian", "fixed",
        "spot", ("spot",),
        (("spot", "spot"), ("risk_free_rate", "risk_free_rate"),
         ("dividend_yield", "dividend_yield"), ("sigma", "sigma"),
         ("beta", "beta")),
        EQUITY_COMMON + (u("sigma", .08, .45), u("beta", .55, .95)),
        "{spot, risk_free_rate, dividend_yield, sigma, beta}",
        pricing_numerical_method="absorbed Milstein",
        legacy_url_name="CEV",
    ),
    SampleModelSpec(
        "heston", "Heston", "equity", "markovian", "markovian", "fixed",
        "spot_state:variance", ("spot", "variance"),
        (("spot", "spot"), ("risk_free_rate", "risk_free_rate"),
         ("dividend_yield", "dividend_yield"),
         ("initial_variance", "initial_variance"), ("kappa", "kappa"),
         ("theta", "theta"), ("gamma", "gamma"), ("rho", "rho")),
        EQUITY_COMMON + (u("initial_variance", .01, .12), u("kappa", .5, 4.),
        u("theta", .01, .15), u("rho", -.95, -.25)),
        "{spot, risk_free_rate, dividend_yield, initial_variance, kappa, theta, gamma, rho}",
        "const float gamma = uniform({std::max(std::sqrt(kappa * theta / 5.0f), 0.08f), std::min(std::sqrt(12.0f * kappa * theta), 0.8f)}, uniforms);",
        pricing_numerical_method="Andersen QE-M",
        legacy_url_name="Heston",
    ),
    SampleModelSpec(
        "heston_3_2", "Heston 3/2", "equity", "markovian", "markovian",
        "fixed", "spot_state:reciprocal_variance",
        ("spot", "reciprocal_variance"),
        (("spot", "spot"), ("risk_free_rate", "risk_free_rate"),
         ("dividend_yield", "dividend_yield"),
         ("initial_variance", "initial_variance"),
         ("mean_reversion", "mean_reversion"),
         ("long_run_variance", "long_run_variance"),
         ("volatility_of_variance", "volatility_of_variance"),
         ("rho", "rho")),
        EQUITY_COMMON + (u("initial_variance", .01, .09),
        u("mean_reversion", 5., 40.), u("long_run_variance", .015, .09),
        u("volatility_of_variance", 1., 8.), u("rho", -.95, -.10)),
        "{spot, risk_free_rate, dividend_yield, initial_variance, "
        "mean_reversion, long_run_variance, volatility_of_variance, rho}",
        pricing_numerical_method="full-truncation Euler 3/2 variance",
    ),
    SampleModelSpec(
        "kou", "Kou", "equity", "markovian", "markovian", "exact",
        "spot", ("spot",),
        (("spot", "spot"), ("risk_free_rate", "risk_free_rate"),
         ("dividend_yield", "dividend_yield"), ("volatility", "volatility"),
         ("jump_intensity", "jump_intensity"),
         ("up_probability", "up_probability"),
         ("positive_jump_rate", "positive_jump_rate"),
         ("negative_jump_rate", "negative_jump_rate")),
        EQUITY_COMMON + (u("volatility", .08, .45), u("jump_intensity", .02, 1.),
        u("up_probability", .2, .7), u("positive_jump_rate", 3., 20.),
        u("negative_jump_rate", 2., 20.)),
        "{spot, risk_free_rate, dividend_yield, volatility, jump_intensity, "
        "up_probability, positive_jump_rate, negative_jump_rate}",
        pricing_numerical_method="Exact Kou increments",
        legacy_url_name="Kou",
    ),
    SampleModelSpec(
        "merton", "Merton", "equity", "markovian", "markovian", "exact",
        "spot", ("spot",),
        (("spot", "spot"), ("risk_free_rate", "risk_free_rate"),
         ("dividend_yield", "dividend_yield"), ("volatility", "volatility"),
         ("jump_intensity", "jump_intensity"),
         ("jump_log_mean", "jump_log_mean"),
         ("jump_log_volatility", "jump_log_volatility")),
        EQUITY_COMMON + (u("volatility", .08, .45), u("jump_intensity", .02, 1.),
        u("jump_log_mean", -.2, .08), u("jump_log_volatility", .03, .30)),
        "{spot, risk_free_rate, dividend_yield, volatility, jump_intensity, "
        "jump_log_mean, jump_log_volatility}",
        pricing_numerical_method="Exact Merton increments",
        legacy_url_name="Merton",
    ),
    SampleModelSpec(
        "normal_inverse_gaussian", "Normal-Inverse-Gaussian", "equity",
        "markovian", "markovian", "exact", "spot", ("spot",),
        (("spot", "spot"), ("risk_free_rate", "risk_free_rate"),
         ("dividend_yield", "dividend_yield"), ("alpha", "alpha"),
         ("beta", "beta"), ("delta", "delta")),
        (u("risk_free_rate", .001, .08), u("dividend_yield", 0., .06),
         u("alpha", 4., 25.), u("skew_ratio", -.75, .05),
         u("target_volatility", .10, .55)),
        "{1.0f, risk_free_rate, dividend_yield, alpha, beta, delta}",
        "const float beta = skew_ratio * alpha; const float gamma = std::sqrt(alpha * alpha - beta * beta); const float delta = target_volatility * target_volatility * gamma * gamma * gamma / (alpha * alpha);",
        "alpha > std::max(std::fabs(beta + 1.0f), std::fabs(beta + 2.0f)) + 0.05f",
        pricing_numerical_method="Exact inverse-Gaussian subordination",
        legacy_url_name="NormalInverseGaussian",
    ),
    SampleModelSpec(
        "sabr", "SABR", "equity", "markovian", "markovian", "fixed",
        "spot_state:alpha", ("spot", "alpha"),
        (("spot", "spot"), ("risk_free_rate", "risk_free_rate"),
         ("dividend_yield", "dividend_yield"),
         ("initial_volatility", "initial_volatility"),
         ("volatility_of_volatility", "volatility_of_volatility"),
         ("rho", "rho"), ("beta", "beta")),
        (u("spot", .25, 4.), u("risk_free_rate", .001, .08),
         u("dividend_yield", 0., .06), u("initial_volatility", .08, .50),
         u("volatility_of_volatility", .10, 2.), u("rho", -.90, .20),
         u("beta", .30, 1.)),
        "{spot, risk_free_rate, dividend_yield, initial_volatility, "
        "volatility_of_volatility, rho, beta}",
        pricing_numerical_method="Lamperti SABR Euler",
    ),
    SampleModelSpec(
        "schobel_zhu", "Schobel-Zhu", "equity", "markovian", "markovian",
        "fixed", "spot_state:volatility", ("spot", "volatility"),
        (("spot", "spot"), ("risk_free_rate", "risk_free_rate"),
         ("dividend_yield", "dividend_yield"),
         ("initial_volatility", "initial_volatility"),
         ("mean_reversion", "mean_reversion"),
         ("long_run_volatility", "long_run_volatility"),
         ("volatility_of_volatility", "volatility_of_volatility"),
         ("correlation", "correlation")),
        EQUITY_COMMON + (u("initial_volatility", .08, .45),
        u("mean_reversion", .30, 5.), u("long_run_volatility", .08, .45),
        u("volatility_of_volatility", .03, .60), u("correlation", -.90, .30)),
        "{spot, risk_free_rate, dividend_yield, initial_volatility, "
        "mean_reversion, long_run_volatility, volatility_of_volatility, correlation}",
        pricing_numerical_method="exact OU factor with log-spot Euler",
        legacy_url_name="SchobelZhu",
    ),
    SampleModelSpec(
        "stein_stein", "Stein-Stein", "equity", "markovian", "markovian",
        "fixed", "spot_state:volatility", ("spot", "volatility"),
        (("spot", "spot"), ("risk_free_rate", "risk_free_rate"),
         ("dividend_yield", "dividend_yield"),
         ("initial_volatility", "initial_volatility"),
         ("mean_reversion", "mean_reversion"),
         ("volatility_of_volatility", "volatility_of_volatility"),
         ("rho", "rho")),
        EQUITY_COMMON + (u("initial_volatility", .10, .35),
        u("mean_reversion", .5, 8.), u("volatility_of_volatility", .05, .50)),
        "{spot, risk_free_rate, dividend_yield, initial_volatility, "
        "mean_reversion, volatility_of_volatility, 0.0f}",
        pricing_numerical_method="exact OU volatility with log-spot Euler",
    ),
    SampleModelSpec(
        "variance_gamma", "Variance-Gamma", "equity", "markovian",
        "markovian", "exact", "spot", ("spot",),
        (("spot", "spot"), ("risk_free_rate", "risk_free_rate"),
         ("dividend_yield", "dividend_yield"), ("sigma", "sigma"),
         ("nu", "nu"), ("theta", "theta")),
        EQUITY_COMMON + (u("sigma", .08, .45), u("nu", .03, .50),
        u("theta", -.35, .15)),
        "{spot, risk_free_rate, dividend_yield, sigma, nu, theta}",
        "const float first_moment = 1.0f - theta * nu - 0.5f * sigma * sigma * nu; const float second_moment = 1.0f - 2.0f * theta * nu - 2.0f * sigma * sigma * nu;",
        "first_moment > 0.05f && second_moment > 0.05f",
        pricing_numerical_method="Exact Gamma subordination",
        legacy_url_name="VarianceGamma",
    ),
    SampleModelSpec(
        "rough_bergomi", "Rough-Bergomi", "equity", "rough", "volterra",
        "fixed", "spot", ("spot",),
        (("spot", "spot"), ("risk_free_rate", "risk_free_rate"),
         ("dividend_yield", "dividend_yield"), ("xi_0", "xi_0"),
         ("eta", "eta"), ("hurst_exponent", "hurst_exponent"),
         ("rho", "rho")),
        (u("spot", 1., 1.), u("risk_free_rate", .001, .08),
         u("dividend_yield", 0., .06), u("xi_0", .04, .04),
         u("eta", .5, 3.), u("hurst_exponent", .03, .25), u("rho", -.95, -.30)),
        "{spot, risk_free_rate, dividend_yield, xi_0, eta, hurst_exponent, rho}",
        kernel="volterra::FractionalHybridKernelPolicy",
        pricing_numerical_method=(
            "Bennedsen-Lunde-Pakkanen hybrid FFT (kappa=1)"
        ),
    ),
    SampleModelSpec(
        "rough_sabr", "Rough-SABR", "equity", "rough", "volterra", "fixed",
        "spot", ("spot",),
        (("spot", "spot"), ("risk_free_rate", "risk_free_rate"),
         ("dividend_yield", "dividend_yield"), ("xi_0", "xi_0"),
         ("eta", "eta"), ("hurst_exponent", "hurst_exponent"),
         ("rho", "rho"), ("beta", "beta")),
        (u("spot", .25, 4.), u("risk_free_rate", .001, .08),
         u("dividend_yield", 0., .06), u("xi_0", .04, .04),
         u("eta", .5, 3.), u("hurst_exponent", .03, .25),
         u("rho", -.95, -.30), u("beta", .70, 1.)),
        "{spot, risk_free_rate, dividend_yield, xi_0, eta, hurst_exponent, rho, beta}",
        kernel="volterra::FractionalHybridKernelPolicy",
        pricing_numerical_method=(
            "Bennedsen-Lunde-Pakkanen hybrid FFT with Lamperti spot"
        ),
    ),
    SampleModelSpec(
        "log_modulated_rough_bergomi", "Log-modulated rough-Bergomi", "equity",
        "rough", "volterra", "fixed", "spot", ("spot",),
        (("spot", "spot"), ("risk_free_rate", "risk_free_rate"),
         ("dividend_yield", "dividend_yield"), ("xi_0", "xi_0"),
         ("eta", "eta"), ("hurst_exponent", "hurst_exponent"),
         ("rho", "rho"), ("log_modulation_scale", "log_modulation_scale"),
         ("log_modulation_power", "log_modulation_power")),
        EQUITY_COMMON + (u("xi_0", .01, .09), u("eta", .5, 3.),
        u("hurst_exponent", .01, .20), u("rho", -.95, -.20),
        u("log_modulation_scale", .03, .30), u("log_modulation_power", 1.20, 4.)),
        "{spot, risk_free_rate, dividend_yield, xi_0, eta, hurst_exponent, rho, "
        "log_modulation_scale, log_modulation_power}",
        kernel="volterra::LogModulatedHybridKernelPolicy",
        pricing_numerical_method="log-modulated hybrid FFT (kappa=1)",
    ),
    SampleModelSpec(
        "rough_stein_stein", "Rough Stein-Stein", "equity", "rough",
        "volterra", "fixed", "spot", ("spot",),
        (("spot", "spot"), ("risk_free_rate", "risk_free_rate"),
         ("dividend_yield", "dividend_yield"),
         ("volatility_level", "volatility_level"),
         ("mean_reversion", "mean_reversion"),
         ("volatility_of_volatility", "volatility_of_volatility"),
         ("hurst_exponent", "hurst_exponent"), ("rho", "rho")),
        EQUITY_COMMON + (u("volatility_level", .10, .35),
        u("mean_reversion", .2, 4.), u("volatility_of_volatility", .05, .50),
        u("hurst_exponent", .03, .25), u("rho", -.90, .10)),
        "{spot, risk_free_rate, dividend_yield, volatility_level, mean_reversion, "
        "volatility_of_volatility, hurst_exponent, rho}",
        kernel="volterra::FractionalResolventHybridKernelPolicy",
        pricing_numerical_method="fractional-resolvent hybrid FFT",
    ),
    SampleModelSpec(
        "rough_heston", "Rough-Heston", "equity", "rough", "n_factor",
        "fixed", "spot", ("spot",),
        (("spot", "spot"), ("risk_free_rate", "risk_free_rate"),
         ("dividend_yield", "dividend_yield"),
         ("initial_variance", "initial_variance"),
         ("mean_reversion", "mean_reversion"),
         ("variance_drift", "variance_drift"),
         ("volatility_of_variance", "volatility_of_variance"),
         ("hurst_exponent", "hurst_exponent"), ("rho", "rho")),
        EQUITY_COMMON + (u("initial_variance", .01, .12),
        u("mean_reversion", .5, 4.), u("long_run_variance", .01, .15),
        u("hurst_exponent", .03, .25), u("rho", -.95, -.25)),
        "{spot, risk_free_rate, dividend_yield, initial_variance, mean_reversion, "
        "variance_drift, volatility_of_variance, hurst_exponent, rho}",
        "const float variance_drift = mean_reversion * long_run_variance; const float volatility_of_variance = uniform({std::max(std::sqrt(variance_drift / 5.0f), 0.08f), std::min(std::sqrt(12.0f * variance_drift), 0.8f)}, uniforms);",
        pricing_numerical_method="7-factor Markovian lift",
        threads_per_block=256,
    ),
    SampleModelSpec(
        "quadratic_rough_heston", "Quadratic rough-Heston", "equity", "rough",
        "n_factor", "fixed", "spot", ("spot",),
        (("spot", "spot"), ("risk_free_rate", "risk_free_rate"),
         ("dividend_yield", "dividend_yield"),
         ("initial_feedback", "initial_feedback"),
         ("quadratic_scale", "quadratic_scale"),
         ("quadratic_shift", "quadratic_shift"),
         ("variance_floor", "variance_floor"),
         ("feedback_rate", "feedback_rate"),
         ("feedback_volatility", "feedback_volatility"),
         ("hurst_exponent", "hurst_exponent")),
        EQUITY_COMMON + (u("initial_feedback", .03, .20),
        u("quadratic_scale", .10, .80), u("quadratic_shift", .02, .20),
        u("variance_floor", .0005, .02), u("feedback_rate", .30, 3.),
        u("feedback_volatility", .30, 2.), u("hurst_exponent", .01, .20)),
        "{spot, risk_free_rate, dividend_yield, initial_feedback, quadratic_scale, "
        "quadratic_shift, variance_floor, feedback_rate, feedback_volatility, "
        "hurst_exponent}",
        pricing_numerical_method="7-factor Markovian lift",
        threads_per_block=256,
    ),
    SampleModelSpec(
        "cir", "CIR", "fixed_income", "", "markovian", "exact", "state",
        ("state",),
        (("mean_reversion", "process.mean_reversion"),
         ("long_term_mean", "process.long_term_mean"),
         ("volatility", "process.volatility"), ("initial_state", "initial_state")),
        (u("mean_reversion", .03, 1.), u("long_term_mean", .001, .08),
         u("initial_state", .001, .08)),
        "{{mean_reversion, long_term_mean, volatility}, initial_state}",
        "const float volatility = uniform({std::max(std::sqrt(mean_reversion * long_term_mean / 5.0f), 0.005f), std::min(std::sqrt(12.0f * mean_reversion * long_term_mean), 0.30f)}, uniforms);",
        pricing_numerical_method="exact noncentral-chi-square transition",
        analytics_contract="standalone CIR affine provider",
    ),
    SampleModelSpec(
        "g2", "G2", "fixed_income", "", "markovian", "exact",
        "two_state:state_x:state_y", ("state_x", "state_y"),
        (("mean_reversion_x", "process.mean_reversion_x"),
         ("volatility_x", "process.volatility_x"),
         ("mean_reversion_y", "process.mean_reversion_y"),
         ("volatility_y", "process.volatility_y"),
         ("correlation", "process.correlation"),
         ("initial_state_x", "initial_state.state_x"),
         ("initial_state_y", "initial_state.state_y")),
        (u("mean_reversion_x", .03, .35), u("mean_reversion_y", .10, 1.),
         u("volatility_x", .0025, .018), u("volatility_y", .0015, .012),
         u("correlation", -.75, .25), u("initial_state_x", .001, .05),
         u("initial_state_y", .001, .03)),
        "{{mean_reversion_x, volatility_x, mean_reversion_y, volatility_y, "
        "correlation}, {initial_state_x, initial_state_y}}",
        pricing_numerical_method="exact correlated Gaussian transition",
        analytics_contract="standalone two-factor affine provider",
    ),
    SampleModelSpec(
        "g2_plus_plus", "G2++", "fixed_income", "", "markovian", "exact",
        "two_state:state_x:state_y", ("state_x", "state_y"),
        (("mean_reversion_x", "process.mean_reversion_x"),
         ("volatility_x", "process.volatility_x"),
         ("mean_reversion_y", "process.mean_reversion_y"),
         ("volatility_y", "process.volatility_y"),
         ("correlation", "process.correlation")),
        (u("mean_reversion_x", .03, .35), u("mean_reversion_y", .10, 1.),
         u("volatility_x", .0025, .018), u("volatility_y", .0015, .012),
         u("correlation", -.75, .25)),
        "{{mean_reversion_x, volatility_x, mean_reversion_y, volatility_y, correlation}}",
        extra_dynamics_include="#include \"model/fixed_income/g2/dynamics_impl.cuh\"\n",
        pricing_numerical_method="exact correlated Gaussian transition",
        analytics_contract="curve-fitted two-factor affine provider",
    ),
    SampleModelSpec(
        "hull_white", "Hull-White", "fixed_income", "", "markovian", "exact",
        "state", ("state",),
        (("mean_reversion", "mean_reversion"), ("volatility", "volatility")),
        (u("mean_reversion", .03, 1.), u("stationary_volatility", .0025, .025)),
        "{mean_reversion, volatility}",
        "const float volatility = stationary_volatility * std::sqrt(2.0f * mean_reversion);",
        extra_dynamics_include="#include \"model/fixed_income/ornstein_uhlenbeck/dynamics_impl.cuh\"\n",
        pricing_numerical_method="exact OU transition",
        analytics_contract="curve-fitted one-factor affine provider",
    ),
    SampleModelSpec(
        "ornstein_uhlenbeck", "Ornstein-Uhlenbeck", "fixed_income", "",
        "markovian", "exact", "state", ("state",),
        (("mean_reversion", "process.mean_reversion"),
         ("volatility", "process.volatility"), ("initial_state", "initial_state")),
        (u("mean_reversion", .03, 1.), u("stationary_volatility", .0025, .025),
         u("initial_state", .001, .08)),
        "{{mean_reversion, volatility}, initial_state}",
        "const float volatility = stationary_volatility * std::sqrt(2.0f * mean_reversion);",
        pricing_numerical_method="exact OU transition",
        analytics_contract="standalone one-factor affine provider",
    ),
    SampleModelSpec(
        "vasicek", "Vasicek", "fixed_income", "", "markovian", "exact",
        "state", ("state",),
        (("mean_reversion", "process.mean_reversion"),
         ("long_term_mean", "process.long_term_mean"),
         ("volatility", "process.volatility"), ("initial_state", "initial_state")),
        (u("mean_reversion", .03, 1.), u("long_term_mean", .001, .08),
         u("stationary_volatility", .0025, .025), u("initial_state", .001, .08)),
        "{{mean_reversion, long_term_mean, volatility}, initial_state}",
        "const float volatility = stationary_volatility * std::sqrt(2.0f * mean_reversion);",
        pricing_numerical_method="exact Gaussian mean-reverting transition",
        analytics_contract="standalone one-factor affine provider",
    ),
)

SAMPLE_MODEL_BY_NAME = {model.name: model for model in SAMPLE_MODELS}


def validate_model_contracts(models) -> None:
    names = [model.name for model in models]
    if len(names) != len(set(names)):
        raise ValueError("duplicate canonical model contract")
    for model in models:
        if not model.pricing_numerical_method:
            raise ValueError(
                f"model lacks a numerical method: {model.name}"
            )
        if not model.observation or not model.outputs:
            raise ValueError(
                f"model lacks state/observable contracts: {model.name}"
            )
        if (
            not model.supported_architectures
            or len(model.supported_architectures)
                != len(set(model.supported_architectures))
        ):
            raise ValueError(
                f"model lacks unique architecture contracts: {model.name}"
            )


validate_model_contracts(SAMPLE_MODELS)
assert len(SAMPLE_MODELS) == 24
