// Validate every one-factor Jamshidian launcher against independent FP64 data.
#include "common/check_cuda.cuh"
#include "model/fixed_income/cir/european_swaption.cuh"
#include "model/fixed_income/hull_white/nelson_siegel/european_swaption.cuh"
#include "model/fixed_income/hull_white/svensson/european_swaption.cuh"
#include "model/fixed_income/vasicek/european_swaption.cuh"
#include "product/european_swaption/schedule.cuh"

// Instantiate both schedule-view paths directly in this test translation unit.
#include "model/fixed_income/cir/analytics_impl.cuh"
#include "model/fixed_income/hull_white/nelson_siegel/analytics_impl.cuh"
#include "model/fixed_income/hull_white/svensson/analytics_impl.cuh"
#include "model/fixed_income/ornstein_uhlenbeck/analytics_impl.cuh"
#include "model/fixed_income/vasicek/analytics_impl.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using CirModel = ai_factory::workbench::model::fixed_income::cir::ModelParameters;
using HullWhiteModel =
    ai_factory::workbench::model::fixed_income::hull_white::ModelParameters;
using NelsonSiegelCurve =
    ai_factory::workbench::curve::nelson_siegel::NelsonSiegelParameters;
using OrnsteinUhlenbeckModel = ai_factory::workbench::model::fixed_income::
    ornstein_uhlenbeck::ModelParameters;
using Product =
    ai_factory::workbench::product::RegularEuropeanSwaptionParameters;
using ExplicitProduct =
    ai_factory::workbench::product::ExplicitEuropeanSwaptionParameters;
using SvenssonCurve =
    ai_factory::workbench::curve::svensson::SvenssonParameters;
using VasicekModel =
    ai_factory::workbench::model::fixed_income::vasicek::ModelParameters;

template<typename Model>
using RegularTwoInputLauncher = void (*)(
    const Model*, std::size_t, const Product*, std::size_t, ai_factory::workbench::PriceConstruction,
    std::size_t, std::size_t, std::size_t, float, unsigned int,
    std::size_t, float*
);

template<typename Curve>
using RegularHullWhiteLauncher = void (*)(
    const HullWhiteModel*, std::size_t, const Curve*, std::size_t,
    const Product*, std::size_t, ai_factory::workbench::PriceConstruction, std::size_t, std::size_t,
    std::size_t, float, unsigned int, std::size_t, float*
);

constexpr double kDayFraction = 1.0 / 252.0;

struct ProductFixtures {
    std::vector<Product> products;
};

// Stop immediately with one readable invariant name.
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<
    ai_factory::workbench::SwaptionSide Side,
    typename ProductParameters,
    typename ScheduleSource
>
__global__ void cir_scalar_swaption_reference_kernel(
    const CirModel* models,
    const ProductParameters* products,
    ScheduleSource schedule_source,
    float time_day_fraction,
    float* prices
) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) return;
    namespace cir =
        ai_factory::workbench::model::fixed_income::cir;
    namespace product = ai_factory::workbench::product;
    const auto schedule = product::make_european_swaption_schedule_view(
        products[0],
        schedule_source,
        time_day_fraction
    );
    const float exercise_time =
        static_cast<float>(products[0].exercise_time_days)
            * time_day_fraction;
    prices[0] = products[0].notional * cir::european_swaption_price<Side>(
        models[0],
        models[0].initial_state,
        0.0f,
        exercise_time,
        products[0].strike,
        schedule
    );
}

// Compare regular reconstruction with the same explicitly materialized leg.
__global__ void schedule_view_equivalence_kernel(float* differences) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) return;

    namespace cir = ai_factory::workbench::model::fixed_income::cir;
    namespace hw_ns =
        ai_factory::workbench::model::fixed_income::hull_white::nelson_siegel;
    namespace hw_sv =
        ai_factory::workbench::model::fixed_income::hull_white::svensson;
    namespace ou =
        ai_factory::workbench::model::fixed_income::ornstein_uhlenbeck;
    namespace product = ai_factory::workbench::product;
    namespace vasicek = ai_factory::workbench::model::fixed_income::vasicek;

    constexpr float day_fraction = 1.0f / 252.0f;
    constexpr float exercise_time = 1.0f;
    constexpr float strike = 0.035f;
    constexpr std::uint32_t payment_days[] = {
        378U, 504U, 630U, 756U,
    };
    constexpr float accrual_fractions[] = {
        0.5f, 0.5f, 0.5f, 0.5f,
    };
    const product::RegularEuropeanSwaptionScheduleView regular = {
        1.5f, 0.5f, 0.5f, 4U,
    };
    const product::ExplicitEuropeanSwaptionScheduleView explicit_schedule = {
        payment_days, accrual_fractions, 4U, day_fraction,
    };

    const OrnsteinUhlenbeckModel ou_model = {
        {0.10f, 0.010f}, 0.030f,
    };
    differences[0] = fabsf(
        ou::european_payer_swaption_price(
            ou_model, ou_model.initial_state, 0.0f,
            exercise_time, strike, regular
        )
        - ou::european_payer_swaption_price(
            ou_model, ou_model.initial_state, 0.0f,
            exercise_time, strike, explicit_schedule
        )
    );
    differences[1] = fabsf(
        ou::european_receiver_swaption_price(
            ou_model, ou_model.initial_state, 0.0f,
            exercise_time, strike, regular
        )
        - ou::european_receiver_swaption_price(
            ou_model, ou_model.initial_state, 0.0f,
            exercise_time, strike, explicit_schedule
        )
    );

    const VasicekModel vasicek_model = {
        {0.10f, 0.030f, 0.010f}, 0.025f,
    };
    differences[2] = fabsf(
        ai_factory::workbench::model::fixed_income::vasicek::european_payer_swaption_price(
            vasicek_model, vasicek_model.initial_state, 0.0f,
            exercise_time, strike, regular
        )
        - ai_factory::workbench::model::fixed_income::vasicek::european_payer_swaption_price(
            vasicek_model, vasicek_model.initial_state, 0.0f,
            exercise_time, strike, explicit_schedule
        )
    );
    differences[3] = fabsf(
        ai_factory::workbench::model::fixed_income::vasicek::european_receiver_swaption_price(
            vasicek_model, vasicek_model.initial_state, 0.0f,
            exercise_time, strike, regular
        )
        - ai_factory::workbench::model::fixed_income::vasicek::european_receiver_swaption_price(
            vasicek_model, vasicek_model.initial_state, 0.0f,
            exercise_time, strike, explicit_schedule
        )
    );

    const CirModel cir_model = {
        {0.60f, 0.040f, 0.15f}, 0.030f,
    };
    differences[4] = fabsf(
        ai_factory::workbench::model::fixed_income::cir::european_payer_swaption_price(
            cir_model, cir_model.initial_state, 0.0f,
            exercise_time, strike, regular
        )
        - ai_factory::workbench::model::fixed_income::cir::european_payer_swaption_price(
            cir_model, cir_model.initial_state, 0.0f,
            exercise_time, strike, explicit_schedule
        )
    );
    differences[5] = fabsf(
        ai_factory::workbench::model::fixed_income::cir::european_receiver_swaption_price(
            cir_model, cir_model.initial_state, 0.0f,
            exercise_time, strike, regular
        )
        - ai_factory::workbench::model::fixed_income::cir::european_receiver_swaption_price(
            cir_model, cir_model.initial_state, 0.0f,
            exercise_time, strike, explicit_schedule
        )
    );

    const HullWhiteModel hull_white_model = {0.10f, 0.010f};
    const NelsonSiegelCurve nelson_siegel_curve = {
        0.030f, -0.010f, 0.015f, 1.50f,
    };
    const auto nelson_siegel_model = hw_ns::compose_fitted_model(
        hull_white_model, nelson_siegel_curve
    );
    differences[6] = fabsf(
        hw_ns::european_payer_swaption_price(
            nelson_siegel_model, 0.0f, 0.0f,
            exercise_time, strike, regular
        )
        - hw_ns::european_payer_swaption_price(
            nelson_siegel_model, 0.0f, 0.0f,
            exercise_time, strike, explicit_schedule
        )
    );
    differences[7] = fabsf(
        hw_ns::european_receiver_swaption_price(
            nelson_siegel_model, 0.0f, 0.0f,
            exercise_time, strike, regular
        )
        - hw_ns::european_receiver_swaption_price(
            nelson_siegel_model, 0.0f, 0.0f,
            exercise_time, strike, explicit_schedule
        )
    );

    const SvenssonCurve svensson_curve = {
        0.030f, -0.010f, 0.015f, -0.005f, 1.50f, 4.00f,
    };
    const auto svensson_model = hw_sv::compose_fitted_model(
        hull_white_model, svensson_curve
    );
    differences[8] = fabsf(
        hw_sv::european_payer_swaption_price(
            svensson_model, 0.0f, 0.0f,
            exercise_time, strike, regular
        )
        - hw_sv::european_payer_swaption_price(
            svensson_model, 0.0f, 0.0f,
            exercise_time, strike, explicit_schedule
        )
    );
    differences[9] = fabsf(
        hw_sv::european_receiver_swaption_price(
            svensson_model, 0.0f, 0.0f,
            exercise_time, strike, regular
        )
        - hw_sv::european_receiver_swaption_price(
            svensson_model, 0.0f, 0.0f,
            exercise_time, strike, explicit_schedule
        )
    );
}

// Require both representations to feed identical analytical formulas.
void check_schedule_view_equivalence() {
    constexpr std::size_t difference_count = 10U;
    float* device_differences = nullptr;
    ai_factory::workbench::check_cuda(
        cudaMalloc(
            &device_differences, difference_count * sizeof(float)
        ),
        "Swaption schedule equivalence cudaMalloc"
    );
    schedule_view_equivalence_kernel<<<1U, 1U>>>(device_differences);
    ai_factory::workbench::check_cuda(
        cudaGetLastError(),
        "Swaption schedule equivalence kernel"
    );
    std::array<float, difference_count> differences{};
    ai_factory::workbench::check_cuda(
        cudaMemcpy(
            differences.data(),
            device_differences,
            difference_count * sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "Swaption schedule equivalence cudaMemcpy"
    );
    ai_factory::workbench::check_cuda(
        cudaFree(device_differences),
        "Swaption schedule equivalence cudaFree"
    );
    for (const float difference : differences) {
        require(
            std::isfinite(difference) && difference <= 1.0e-6f,
            "Regular and explicit swaption schedules disagree"
        );
    }
}

// Return the affine loading B(delta) of a Gaussian one-factor model.
double integral_loading(double mean_reversion, double delta) {
    return -std::expm1(-mean_reversion * delta) / mean_reversion;
}

// Return the variance of the future integrated centered OU factor.
double integral_variance(
    double mean_reversion,
    double volatility,
    double delta
) {
    const double bracket =
        delta
        + 2.0 * std::expm1(-mean_reversion * delta) / mean_reversion
        - std::expm1(-2.0 * mean_reversion * delta)
            / (2.0 * mean_reversion);
    return volatility * volatility * bracket
        / (mean_reversion * mean_reversion);
}

// Evaluate a Vasicek conditional zero-coupon in FP64.
double vasicek_zero_coupon(
    const VasicekModel& model,
    double state,
    double valuation_time,
    double maturity
) {
    const double delta = maturity - valuation_time;
    const double loading = integral_loading(
        model.process.mean_reversion, delta
    );
    const double mean_increment =
        model.process.long_term_mean * (delta - loading);
    return std::exp(
        -loading * state - mean_increment
        + 0.5 * integral_variance(
            model.process.mean_reversion,
            model.process.volatility,
            delta
        )
    );
}

// Price one Vasicek call (+1) or put (-1) on a zero-coupon in FP64.
double vasicek_bond_option(
    const VasicekModel& model,
    double option_sign,
    double option_expiry,
    double bond_maturity,
    double strike
) {
    const double expiry_bond = vasicek_zero_coupon(
        model, model.initial_state, 0.0, option_expiry
    );
    const double underlying_bond = vasicek_zero_coupon(
        model, model.initial_state, 0.0, bond_maturity
    );
    const double a = model.process.mean_reversion;
    const double volatility = model.process.volatility
        * integral_loading(a, bond_maturity - option_expiry)
        * std::sqrt(-std::expm1(-2.0 * a * option_expiry) / (2.0 * a));
    if (volatility <= 1.0e-14) {
        return std::max(
            option_sign * (underlying_bond - strike * expiry_bond), 0.0
        );
    }
    const double d1 =
        std::log(underlying_bond / (strike * expiry_bond)) / volatility
        + 0.5 * volatility;
    const double d2 = d1 - volatility;
    const auto normal_cdf = [](double value) {
        return 0.5 * std::erfc(-value / std::sqrt(2.0));
    };
    return option_sign
        * (
            underlying_bond * normal_cdf(option_sign * d1)
            - strike * expiry_bond * normal_cdf(option_sign * d2)
        );
}

// Evaluate the two parametric initial curves in FP64.
double curve_log_discount(
    const NelsonSiegelCurve& curve,
    double maturity
) {
    if (maturity == 0.0) return 0.0;
    const double x = maturity / static_cast<double>(curve.tau);
    const double loading = -std::expm1(-x) / x;
    const double zero_rate = static_cast<double>(curve.beta0)
        + static_cast<double>(curve.beta1) * loading
        + static_cast<double>(curve.beta2)
            * (loading - std::exp(-x));
    return -maturity * zero_rate;
}

double curve_log_discount(const SvenssonCurve& curve, double maturity) {
    if (maturity == 0.0) return 0.0;
    const double x1 = maturity / static_cast<double>(curve.tau1);
    const double x2 = maturity / static_cast<double>(curve.tau2);
    const double loading1 = -std::expm1(-x1) / x1;
    const double loading2 = -std::expm1(-x2) / x2;
    const double zero_rate = static_cast<double>(curve.beta0)
        + static_cast<double>(curve.beta1) * loading1
        + static_cast<double>(curve.beta2)
            * (loading1 - std::exp(-x1))
        + static_cast<double>(curve.beta3)
            * (loading2 - std::exp(-x2));
    return -maturity * zero_rate;
}

// Integrate the Hull-White deterministic shift independently in FP64.
template<typename Curve>
double hull_white_shift_integral(
    const HullWhiteModel& model,
    const Curve& curve,
    double start,
    double end
) {
    const double a = model.mean_reversion;
    const double delta = end - start;
    const double forward_integral =
        curve_log_discount(curve, start)
        - curve_log_discount(curve, end);
    if (start == 0.0) {
        return forward_integral
            + 0.5 * integral_variance(a, model.volatility, end);
    }
    const double convexity_integral =
        model.volatility * model.volatility / (2.0 * a * a)
        * (
            delta
            - 2.0 * std::exp(-a * start)
                * (-std::expm1(-a * delta)) / a
            + std::exp(-2.0 * a * start)
                * (-std::expm1(-2.0 * a * delta)) / (2.0 * a)
        );
    return forward_integral + convexity_integral;
}

// Evaluate one fitted Hull-White conditional zero-coupon in FP64.
template<typename Curve>
double hull_white_zero_coupon(
    const HullWhiteModel& model,
    const Curve& curve,
    double state,
    double valuation_time,
    double maturity
) {
    if (valuation_time == 0.0)
        return std::exp(curve_log_discount(curve, maturity));
    const double delta = maturity - valuation_time;
    const double log_a = -hull_white_shift_integral(
        model, curve, valuation_time, maturity
    ) + 0.5 * integral_variance(
        model.mean_reversion, model.volatility, delta
    );
    return std::exp(
        log_a - integral_loading(model.mean_reversion, delta) * state
    );
}

// Price one fitted Hull-White bond option in FP64.
template<typename Curve>
double hull_white_bond_option(
    const HullWhiteModel& model,
    const Curve& curve,
    double option_sign,
    double option_expiry,
    double bond_maturity,
    double strike
) {
    const double expiry_bond =
        std::exp(curve_log_discount(curve, option_expiry));
    const double underlying_bond =
        std::exp(curve_log_discount(curve, bond_maturity));
    const double a = model.mean_reversion;
    const double volatility = model.volatility
        * integral_loading(a, bond_maturity - option_expiry)
        * std::sqrt(-std::expm1(-2.0 * a * option_expiry) / (2.0 * a));
    if (volatility <= 1.0e-14) {
        return std::max(
            option_sign * (underlying_bond - strike * expiry_bond), 0.0
        );
    }
    const double d1 =
        std::log(underlying_bond / (strike * expiry_bond)) / volatility
        + 0.5 * volatility;
    const double d2 = d1 - volatility;
    const auto normal_cdf = [](double value) {
        return 0.5 * std::erfc(-value / std::sqrt(2.0));
    };
    return option_sign
        * (
            underlying_bond * normal_cdf(option_sign * d1)
            - strike * expiry_bond * normal_cdf(option_sign * d2)
        );
}

// Solve and apply one generic one-factor Jamshidian decomposition in FP64.
template<typename ConditionalBond, typename BondOption>
double jamshidian_price(
    const Product& product,
    bool payer,
    ConditionalBond conditional_bond,
    BondOption bond_option
) {
    const double exercise_time = product.exercise_time_days * kDayFraction;
    const auto coupon_bond = [&](double state) {
        double value = 0.0;
        for (std::uint32_t payment = 0U;
             payment < product.payment_count;
            ++payment) {
            const double coefficient =
                product.strike * product.accrual_fraction
                + (payment + 1U == product.payment_count ? 1.0 : 0.0);
            const std::uint32_t payment_time_days = product.exercise_time_days
                + (payment + 1U) * product.payment_interval_days;
            value += coefficient * conditional_bond(
                state,
                exercise_time,
                payment_time_days * kDayFraction
            );
        }
        return value;
    };

    double lower = -0.25;
    double upper = 0.25;
    double width = 0.25;
    for (unsigned int expansion = 0U;
         expansion < 64U && coupon_bond(lower) < 1.0;
         ++expansion) {
        width *= 2.0;
        lower -= width;
    }
    width = 0.25;
    for (unsigned int expansion = 0U;
         expansion < 64U && coupon_bond(upper) > 1.0;
         ++expansion) {
        width *= 2.0;
        upper += width;
    }
    require(
        coupon_bond(lower) >= 1.0 && coupon_bond(upper) <= 1.0,
        "FP64 Jamshidian test failed to bracket the state boundary"
    );
    for (unsigned int iteration = 0U; iteration < 120U; ++iteration) {
        const double middle = 0.5 * (lower + upper);
        if (coupon_bond(middle) > 1.0)
            lower = middle;
        else
            upper = middle;
    }
    const double boundary = 0.5 * (lower + upper);

    double price = 0.0;
    for (std::uint32_t payment = 0U;
         payment < product.payment_count;
        ++payment) {
        const double coefficient =
            product.strike * product.accrual_fraction
            + (payment + 1U == product.payment_count ? 1.0 : 0.0);
        const std::uint32_t payment_day = product.exercise_time_days
            + (payment + 1U) * product.payment_interval_days;
        const double payment_time =
            payment_day * kDayFraction;
        const double bond_strike = conditional_bond(
            boundary, exercise_time, payment_time
        );
        price += coefficient * bond_option(
            payer ? -1.0 : 1.0,
            exercise_time,
            payment_time,
            bond_strike
        );
    }
    return product.notional * price;
}

// Build one regular fixed-leg schedule.
void append_product(
    ProductFixtures& fixtures,
    float notional,
    float strike,
    std::uint32_t exercise_time_days,
    std::uint32_t payment_interval_days,
    std::uint32_t payment_count,
    float accrual_fraction
) {
    require(
        payment_interval_days > 0U
            && payment_count > 0U
            && accrual_fraction > 0.0f,
        "Invalid European swaption fixture schedule"
    );
    Product product{};
    product.notional = notional;
    product.strike = strike;
    product.accrual_fraction = accrual_fraction;
    product.exercise_time_days = exercise_time_days;
    product.payment_interval_days = payment_interval_days;
    product.payment_count = payment_count;
    fixtures.products.push_back(product);
}

std::uint32_t maximum_payment_count(const ProductFixtures& fixtures) {
    std::uint32_t maximum = 0U;
    for (const Product& product : fixtures.products) {
        maximum = std::max(maximum, product.payment_count);
    }
    return maximum;
}

float read_device_price(float* device_price, const char* operation) {
    float price = 0.0f;
    ai_factory::workbench::check_cuda(
        cudaMemcpy(
            &price,
            device_price,
            sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        operation
    );
    return price;
}

// Compare cooperative CIR pricing with the scalar formula on a long leg.
void check_cir_long_schedule() {
    using ai_factory::workbench::SwaptionSide;
    namespace cir =
        ai_factory::workbench::model::fixed_income::cir;
    namespace product = ai_factory::workbench::product;

    const CirModel model = {
        {0.60f, 0.040f, 0.15f},
        0.030f,
    };
    Product contract{};
    contract.notional = 1.0f;
    contract.strike = 0.035f;
    contract.accrual_fraction = 1.0f / 12.0f;
    contract.exercise_time_days = 1260U;
    contract.payment_interval_days = 21U;
    contract.payment_count = 600U;

    CirModel* device_model = nullptr;
    Product* device_product = nullptr;
    float* device_price = nullptr;
    try {
        ai_factory::workbench::check_cuda(
            cudaMalloc(&device_model, sizeof(CirModel)),
            "Long CIR swaption test cudaMalloc model"
        );
        ai_factory::workbench::check_cuda(
            cudaMalloc(&device_product, sizeof(Product)),
            "Long CIR swaption test cudaMalloc product"
        );
        ai_factory::workbench::check_cuda(
            cudaMalloc(&device_price, sizeof(float)),
            "Long CIR swaption test cudaMalloc price"
        );
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                device_model,
                &model,
                sizeof(CirModel),
                cudaMemcpyHostToDevice
            ),
            "Long CIR swaption test cudaMemcpy model"
        );
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                device_product,
                &contract,
                sizeof(Product),
                cudaMemcpyHostToDevice
            ),
            "Long CIR swaption test cudaMemcpy product"
        );

        const auto scalar_price = [&]<SwaptionSide Side>() {
            cir_scalar_swaption_reference_kernel<Side><<<1U, 1U>>>(
                device_model,
                device_product,
                product::RegularEuropeanSwaptionScheduleSource{},
                static_cast<float>(kDayFraction),
                device_price
            );
            ai_factory::workbench::check_cuda(
                cudaGetLastError(),
                "Long CIR scalar swaption reference kernel"
            );
            return read_device_price(
                device_price,
                "Long CIR scalar swaption reference copy"
            );
        };
        const auto cooperative_price = [&]<SwaptionSide Side>(
            std::uint32_t maximum
        ) {
            cir::launch_cir_european_swaption_cuda<Side>(
                device_model,
                1U,
                device_product,
                1U,
                ai_factory::workbench::PriceConstruction::Aligned,
                1U,
                0U,
                1U,
                static_cast<float>(kDayFraction),
                128U,
                1U,
                device_price,
                maximum
            );
            return read_device_price(
                device_price,
                "Long CIR cooperative swaption copy"
            );
        };

        const float scalar_payer =
            scalar_price.template operator()<SwaptionSide::payer>();
        const float scalar_receiver =
            scalar_price.template operator()<SwaptionSide::receiver>();
        const float cooperative_payer =
            cooperative_price.template operator()<SwaptionSide::payer>(600U);
        const float cooperative_receiver =
            cooperative_price.template operator()<SwaptionSide::receiver>(
                600U
            );
        require(
            std::isfinite(cooperative_payer)
                && std::fabs(cooperative_payer - scalar_payer) <= 2.0e-5f,
            "Long CIR cooperative payer differs from the scalar formula"
        );
        require(
            std::isfinite(cooperative_receiver)
                && std::fabs(cooperative_receiver - scalar_receiver)
                    <= 2.0e-5f,
            "Long CIR cooperative receiver differs from the scalar formula"
        );

        const float understated_capacity =
            cooperative_price.template operator()<SwaptionSide::payer>(599U);
        require(
            std::isnan(understated_capacity),
            "CIR cooperative pricing accepted an understated payment maximum"
        );

        const float scalar_fallback =
            cooperative_price.template operator()<SwaptionSide::payer>(
                std::numeric_limits<std::uint32_t>::max()
            );
        require(
            std::isfinite(scalar_fallback)
                && std::fabs(scalar_fallback - scalar_payer) <= 1.0e-6f,
            "CIR scalar fallback differs from the scalar formula"
        );
    } catch (...) {
        if (device_model != nullptr) cudaFree(device_model);
        if (device_product != nullptr) cudaFree(device_product);
        if (device_price != nullptr) cudaFree(device_price);
        throw;
    }
    ai_factory::workbench::check_cuda(
        cudaFree(device_model), "Long CIR swaption test cudaFree model"
    );
    ai_factory::workbench::check_cuda(
        cudaFree(device_product), "Long CIR swaption test cudaFree product"
    );
    ai_factory::workbench::check_cuda(
        cudaFree(device_price), "Long CIR swaption test cudaFree price"
    );
}

// Exercise the same cooperative body through an arbitrary schedule pool.
void check_cir_explicit_schedule() {
    using ai_factory::workbench::SwaptionSide;
    namespace cir =
        ai_factory::workbench::model::fixed_income::cir;
    namespace product = ai_factory::workbench::product;

    const CirModel model = {
        {0.60f, 0.040f, 0.15f},
        0.030f,
    };
    const ExplicitProduct contract = {
        1.25f,
        0.0325f,
        252U,
        4U,
        0U,
    };
    constexpr std::array<std::uint32_t, 4U> payment_times_days = {
        315U, 420U, 546U, 756U,
    };
    constexpr std::array<float, 4U> accrual_fractions = {
        0.25f, 0.40f, 0.50f, 0.75f,
    };

    CirModel* device_model = nullptr;
    ExplicitProduct* device_product = nullptr;
    std::uint32_t* device_payment_times_days = nullptr;
    float* device_accrual_fractions = nullptr;
    float* device_price = nullptr;
    try {
        ai_factory::workbench::check_cuda(
            cudaMalloc(&device_model, sizeof(CirModel)),
            "Explicit CIR swaption test cudaMalloc model"
        );
        ai_factory::workbench::check_cuda(
            cudaMalloc(&device_product, sizeof(ExplicitProduct)),
            "Explicit CIR swaption test cudaMalloc product"
        );
        ai_factory::workbench::check_cuda(
            cudaMalloc(
                &device_payment_times_days,
                payment_times_days.size() * sizeof(std::uint32_t)
            ),
            "Explicit CIR swaption test cudaMalloc payment times"
        );
        ai_factory::workbench::check_cuda(
            cudaMalloc(
                &device_accrual_fractions,
                accrual_fractions.size() * sizeof(float)
            ),
            "Explicit CIR swaption test cudaMalloc accrual fractions"
        );
        ai_factory::workbench::check_cuda(
            cudaMalloc(&device_price, sizeof(float)),
            "Explicit CIR swaption test cudaMalloc price"
        );
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                device_model,
                &model,
                sizeof(CirModel),
                cudaMemcpyHostToDevice
            ),
            "Explicit CIR swaption test cudaMemcpy model"
        );
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                device_product,
                &contract,
                sizeof(ExplicitProduct),
                cudaMemcpyHostToDevice
            ),
            "Explicit CIR swaption test cudaMemcpy product"
        );
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                device_payment_times_days,
                payment_times_days.data(),
                payment_times_days.size() * sizeof(std::uint32_t),
                cudaMemcpyHostToDevice
            ),
            "Explicit CIR swaption test cudaMemcpy payment times"
        );
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                device_accrual_fractions,
                accrual_fractions.data(),
                accrual_fractions.size() * sizeof(float),
                cudaMemcpyHostToDevice
            ),
            "Explicit CIR swaption test cudaMemcpy accrual fractions"
        );

        const product::ExplicitEuropeanSwaptionScheduleSource source{
            device_payment_times_days,
            device_accrual_fractions,
            payment_times_days.size(),
        };
        const auto scalar_price = [&]<SwaptionSide Side>() {
            cir_scalar_swaption_reference_kernel<Side><<<1U, 1U>>>(
                device_model,
                device_product,
                source,
                static_cast<float>(kDayFraction),
                device_price
            );
            ai_factory::workbench::check_cuda(
                cudaGetLastError(),
                "Explicit CIR scalar swaption reference kernel"
            );
            return read_device_price(
                device_price,
                "Explicit CIR scalar swaption reference copy"
            );
        };
        const auto cooperative_price = [&]<SwaptionSide Side>() {
            cir::launch_cir_european_swaption_cuda<Side>(
                device_model,
                1U,
                device_product,
                device_payment_times_days,
                device_accrual_fractions,
                payment_times_days.size(),
                1U,
                ai_factory::workbench::PriceConstruction::Aligned,
                1U,
                0U,
                1U,
                static_cast<float>(kDayFraction),
                128U,
                1U,
                device_price,
                4U
            );
            return read_device_price(
                device_price,
                "Explicit CIR cooperative swaption copy"
            );
        };

        for (const bool payer : {true, false}) {
            const float scalar = payer
                ? scalar_price.template operator()<SwaptionSide::payer>()
                : scalar_price.template operator()<SwaptionSide::receiver>();
            const float cooperative = payer
                ? cooperative_price.template operator()<SwaptionSide::payer>()
                : cooperative_price.template operator()<
                    SwaptionSide::receiver
                >();
            require(
                std::isfinite(cooperative)
                    && std::fabs(cooperative - scalar) <= 2.0e-6f,
                "Explicit CIR cooperative swaption differs from scalar"
            );
        }
    } catch (...) {
        if (device_model != nullptr) cudaFree(device_model);
        if (device_product != nullptr) cudaFree(device_product);
        if (device_payment_times_days != nullptr) {
            cudaFree(device_payment_times_days);
        }
        if (device_accrual_fractions != nullptr)
            cudaFree(device_accrual_fractions);
        if (device_price != nullptr) cudaFree(device_price);
        throw;
    }
    ai_factory::workbench::check_cuda(
        cudaFree(device_model), "Explicit CIR swaption test cudaFree model"
    );
    ai_factory::workbench::check_cuda(
        cudaFree(device_product), "Explicit CIR swaption test cudaFree product"
    );
    ai_factory::workbench::check_cuda(
        cudaFree(device_payment_times_days),
        "Explicit CIR swaption test cudaFree payment times"
    );
    ai_factory::workbench::check_cuda(
        cudaFree(device_accrual_fractions),
        "Explicit CIR swaption test cudaFree accrual fractions"
    );
    ai_factory::workbench::check_cuda(
        cudaFree(device_price), "Explicit CIR swaption test cudaFree price"
    );
}

// Exercise aligned and model-product Cartesian indexing for one launcher.
template<typename Model, typename Launcher, typename Expected>
void check_two_input_launcher(
    const std::vector<Model>& models,
    const ProductFixtures& fixtures,
    Launcher launcher,
    Expected expected,
    double tolerance,
    const char* mismatch_message
) {
    const std::vector<Product>& products = fixtures.products;
    constexpr std::size_t aligned_count = 3U;
    constexpr std::size_t cartesian_count = 6U;
    Model* device_models = nullptr;
    Product* device_products = nullptr;
    float* device_prices = nullptr;
    try {
        ai_factory::workbench::check_cuda(
            cudaMalloc(&device_models, aligned_count * sizeof(Model)),
            "Swaption test cudaMalloc models"
        );
        ai_factory::workbench::check_cuda(
            cudaMalloc(&device_products, aligned_count * sizeof(Product)),
            "Swaption test cudaMalloc products"
        );
        ai_factory::workbench::check_cuda(
            cudaMalloc(&device_prices, cartesian_count * sizeof(float)),
            "Swaption test cudaMalloc prices"
        );
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                device_models,
                models.data(),
                aligned_count * sizeof(Model),
                cudaMemcpyHostToDevice
            ),
            "Swaption test cudaMemcpy models"
        );
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                device_products,
                products.data(),
                aligned_count * sizeof(Product),
                cudaMemcpyHostToDevice
            ),
            "Swaption test cudaMemcpy products"
        );
        launcher(
            device_models,
            aligned_count,
            device_products,
            aligned_count,
            ai_factory::workbench::PriceConstruction::Aligned,
            aligned_count,
            0U,
            aligned_count,
            static_cast<float>(kDayFraction),
            32U,
            1U,
            device_prices
        );
        std::vector<float> prices(aligned_count);
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                prices.data(),
                device_prices,
                aligned_count * sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "Swaption test cudaMemcpy aligned prices"
        );
        for (std::size_t row = 0U; row < aligned_count; ++row) {
            require(
                std::isfinite(prices[row]) && prices[row] >= 0.0f
                    && std::fabs(
                        static_cast<double>(prices[row])
                        - expected(row, row)
                    ) < tolerance,
                mismatch_message
            );
        }

        launcher(
            device_models,
            2U,
            device_products,
            products.size(),
            ai_factory::workbench::PriceConstruction::CartesianProduct,
            cartesian_count,
            0U,
            2U,
            static_cast<float>(kDayFraction),
            32U,
            1U,
            device_prices
        );
        launcher(
            device_models,
            2U,
            device_products,
            products.size(),
            ai_factory::workbench::PriceConstruction::CartesianProduct,
            cartesian_count,
            2U,
            cartesian_count - 2U,
            static_cast<float>(kDayFraction),
            32U,
            1U,
            device_prices
        );
        prices.resize(cartesian_count);
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                prices.data(),
                device_prices,
                cartesian_count * sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "Swaption test cudaMemcpy Cartesian prices"
        );
        for (std::size_t row = 0U; row < cartesian_count; ++row) {
            const std::size_t model_index = row / products.size();
            const std::size_t product_index = row % products.size();
            require(
                std::isfinite(prices[row]) && prices[row] >= 0.0f
                    && std::fabs(
                        static_cast<double>(prices[row])
                        - expected(model_index, product_index)
                    ) < tolerance,
                mismatch_message
            );
        }
    } catch (...) {
        if (device_models != nullptr) cudaFree(device_models);
        if (device_products != nullptr) cudaFree(device_products);
        if (device_prices != nullptr) cudaFree(device_prices);
        throw;
    }

    ai_factory::workbench::check_cuda(
        cudaFree(device_models), "Swaption test cudaFree models"
    );
    ai_factory::workbench::check_cuda(
        cudaFree(device_products), "Swaption test cudaFree products"
    );
    ai_factory::workbench::check_cuda(
        cudaFree(device_prices), "Swaption test cudaFree prices"
    );
}

// Exercise aligned and three-input Cartesian Hull-White indexing.
template<typename Curve, typename Launcher, typename Expected>
void check_hull_white_launcher(
    const std::vector<HullWhiteModel>& models,
    const std::vector<Curve>& curves,
    const ProductFixtures& fixtures,
    Launcher launcher,
    Expected expected,
    const char* mismatch_message
) {
    const std::vector<Product>& products = fixtures.products;
    constexpr std::size_t aligned_count = 3U;
    constexpr std::size_t cartesian_count = 12U;
    HullWhiteModel* device_models = nullptr;
    Curve* device_curves = nullptr;
    Product* device_products = nullptr;
    float* device_prices = nullptr;
    try {
        ai_factory::workbench::check_cuda(
            cudaMalloc(
                &device_models, aligned_count * sizeof(HullWhiteModel)
            ),
            "Hull-White swaption test cudaMalloc models"
        );
        ai_factory::workbench::check_cuda(
            cudaMalloc(&device_curves, aligned_count * sizeof(Curve)),
            "Hull-White swaption test cudaMalloc curves"
        );
        ai_factory::workbench::check_cuda(
            cudaMalloc(&device_products, aligned_count * sizeof(Product)),
            "Hull-White swaption test cudaMalloc products"
        );
        ai_factory::workbench::check_cuda(
            cudaMalloc(&device_prices, cartesian_count * sizeof(float)),
            "Hull-White swaption test cudaMalloc prices"
        );
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                device_models,
                models.data(),
                aligned_count * sizeof(HullWhiteModel),
                cudaMemcpyHostToDevice
            ),
            "Hull-White swaption test cudaMemcpy models"
        );
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                device_curves,
                curves.data(),
                aligned_count * sizeof(Curve),
                cudaMemcpyHostToDevice
            ),
            "Hull-White swaption test cudaMemcpy curves"
        );
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                device_products,
                products.data(),
                aligned_count * sizeof(Product),
                cudaMemcpyHostToDevice
            ),
            "Hull-White swaption test cudaMemcpy products"
        );
        launcher(
            device_models,
            aligned_count,
            device_curves,
            aligned_count,
            device_products,
            aligned_count,
            ai_factory::workbench::PriceConstruction::Aligned,
            aligned_count,
            0U,
            aligned_count,
            static_cast<float>(kDayFraction),
            32U,
            1U,
            device_prices
        );
        std::vector<float> prices(aligned_count);
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                prices.data(),
                device_prices,
                aligned_count * sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "Hull-White swaption test cudaMemcpy aligned prices"
        );
        for (std::size_t row = 0U; row < aligned_count; ++row) {
            require(
                std::isfinite(prices[row]) && prices[row] >= 0.0f
                    && std::fabs(
                        static_cast<double>(prices[row])
                        - expected(row, row, row)
                    ) < 8.0e-5,
                mismatch_message
            );
        }

        launcher(
            device_models,
            2U,
            device_curves,
            2U,
            device_products,
            products.size(),
            ai_factory::workbench::PriceConstruction::CartesianProduct,
            cartesian_count,
            0U,
            5U,
            static_cast<float>(kDayFraction),
            32U,
            1U,
            device_prices
        );
        launcher(
            device_models,
            2U,
            device_curves,
            2U,
            device_products,
            products.size(),
            ai_factory::workbench::PriceConstruction::CartesianProduct,
            cartesian_count,
            5U,
            cartesian_count - 5U,
            static_cast<float>(kDayFraction),
            32U,
            1U,
            device_prices
        );
        prices.resize(cartesian_count);
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                prices.data(),
                device_prices,
                cartesian_count * sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "Hull-White swaption test cudaMemcpy Cartesian prices"
        );
        for (std::size_t row = 0U; row < cartesian_count; ++row) {
            const std::size_t curve_product_count =
                2U * products.size();
            const std::size_t model_index = row / curve_product_count;
            const std::size_t remainder = row % curve_product_count;
            const std::size_t curve_index = remainder / products.size();
            const std::size_t product_index = remainder % products.size();
            require(
                std::isfinite(prices[row]) && prices[row] >= 0.0f
                    && std::fabs(
                        static_cast<double>(prices[row])
                        - expected(
                            model_index, curve_index, product_index
                        )
                    ) < 8.0e-5,
                mismatch_message
            );
        }
    } catch (...) {
        if (device_models != nullptr) cudaFree(device_models);
        if (device_curves != nullptr) cudaFree(device_curves);
        if (device_products != nullptr) cudaFree(device_products);
        if (device_prices != nullptr) cudaFree(device_prices);
        throw;
    }

    ai_factory::workbench::check_cuda(
        cudaFree(device_models),
        "Hull-White swaption test cudaFree models"
    );
    ai_factory::workbench::check_cuda(
        cudaFree(device_curves),
        "Hull-White swaption test cudaFree curves"
    );
    ai_factory::workbench::check_cuda(
        cudaFree(device_products),
        "Hull-White swaption test cudaFree products"
    );
    ai_factory::workbench::check_cuda(
        cudaFree(device_prices),
        "Hull-White swaption test cudaFree prices"
    );
}

}  // namespace

// Validate payer/receiver prices and every supported construction mode.
int main() {
    using namespace ai_factory::workbench;
    namespace cir = model::fixed_income::cir;
    namespace hw_ns = model::fixed_income::hull_white::nelson_siegel;
    namespace hw_sv = model::fixed_income::hull_white::svensson;
    namespace vasicek = model::fixed_income::vasicek;

    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) {
        return 77;
    }
    check_cuda(availability, "One-factor swaption test cudaGetDeviceCount");
    check_schedule_view_equivalence();
    check_cir_long_schedule();
    check_cir_explicit_schedule();

    ProductFixtures fixtures;
    append_product(
        fixtures,
        1.0f,
        0.025f,
        252U,
        126U,
        4U,
        0.50f
    );
    append_product(
        fixtures,
        2.0f,
        0.040f,
        504U,
        63U,
        8U,
        0.25f
    );
    append_product(
        fixtures,
        1.5f,
        0.060f,
        126U,
        126U,
        3U,
        0.50f
    );
    const std::vector<Product>& products = fixtures.products;

    const std::vector<VasicekModel> vasicek_models = {
        {{0.10f, 0.030f, 0.010f}, 0.025f},
        {{0.25f, 0.045f, 0.015f}, 0.040f},
        {{0.50f, 0.020f, 0.000f}, 0.030f},
    };
    for (const bool payer : {true, false}) {
        const auto expected = [&](std::size_t model_index,
                                  std::size_t product_index) {
            const VasicekModel& model = vasicek_models[model_index];
            return jamshidian_price(
                products[product_index],
                payer,
                [&](double state, double start, double maturity) {
                    return vasicek_zero_coupon(
                        model, state, start, maturity
                    );
                },
                [&](double sign,
                    double expiry,
                    double maturity,
                    double strike) {
                    return vasicek_bond_option(
                        model, sign, expiry, maturity, strike
                    );
                }
            );
        };
        if (payer) {
            check_two_input_launcher(
                vasicek_models,
                fixtures,
                static_cast<RegularTwoInputLauncher<VasicekModel>>(
                    ai_factory::workbench::model::fixed_income::vasicek::launch_vasicek_european_swaption_cuda<
                        SwaptionSide::payer
                    >
                ),
                expected,
                8.0e-5,
                "Vasicek payer swaption differs from the FP64 formula"
            );
        } else {
            check_two_input_launcher(
                vasicek_models,
                fixtures,
                static_cast<RegularTwoInputLauncher<VasicekModel>>(
                    ai_factory::workbench::model::fixed_income::vasicek::launch_vasicek_european_swaption_cuda<
                        SwaptionSide::receiver
                    >
                ),
                expected,
                8.0e-5,
                "Vasicek receiver swaption differs from the FP64 formula"
            );
        }
    }

    const std::vector<CirModel> cir_models = {
        {{0.60f, 0.040f, 0.15f}, 0.030f},
        {{0.05f, 0.015f, 0.12f}, 0.001f},
        {{1.50f, 0.100f, 0.05f}, 0.200f},
    };
    // Independent SciPy non-central-chi-square references, model-major.
    constexpr std::array<std::array<double, 3U>, 3U> cir_payer = {{
        {{0.022361778620097937, 0.016354956799095842,
          0.0003968727256370917}},
        {{0.0000757560659568985, 0.00025737436900818771,
          0.0000000004929704200662665}},
        {{0.127998685581259, 0.172636131436912,
          0.11498483574412403}},
    }};
    constexpr std::array<std::array<double, 3U>, 3U> cir_receiver = {{
        {{0.0008450000119689074, 0.024691234267383833,
          0.0536410645841747}},
        {{0.045321748414883775, 0.14791863982631231,
          0.13058022901212585}},
        {{0.0, 0.0, 0.0}},
    }};
    check_two_input_launcher(
        cir_models,
        fixtures,
        [maximum = maximum_payment_count(fixtures)](auto... arguments) {
            ai_factory::workbench::model::fixed_income::cir::
                launch_cir_european_swaption_cuda<SwaptionSide::payer>(
                    arguments...,
                    maximum
                );
        },
        [&](std::size_t model_index, std::size_t product_index) {
            return cir_payer[model_index][product_index];
        },
        2.0e-4,
        "CIR payer swaption differs from the independent SciPy reference"
    );
    check_two_input_launcher(
        cir_models,
        fixtures,
        [maximum = maximum_payment_count(fixtures)](auto... arguments) {
            ai_factory::workbench::model::fixed_income::cir::
                launch_cir_european_swaption_cuda<SwaptionSide::receiver>(
                    arguments...,
                    maximum
                );
        },
        [&](std::size_t model_index, std::size_t product_index) {
            return cir_receiver[model_index][product_index];
        },
        2.0e-4,
        "CIR receiver swaption differs from the independent SciPy reference"
    );

    const std::vector<HullWhiteModel> hull_white_models = {
        {0.10f, 0.010f},
        {0.25f, 0.015f},
        {0.50f, 0.000f},
    };
    const std::vector<NelsonSiegelCurve> nelson_siegel_curves = {
        {0.030f, -0.010f, 0.015f, 1.50f},
        {0.045f, -0.025f, 0.020f, 2.00f},
        {0.020f, 0.005f, -0.010f, 0.75f},
    };
    const std::vector<SvenssonCurve> svensson_curves = {
        {0.030f, -0.010f, 0.015f, -0.005f, 1.50f, 4.00f},
        {0.045f, -0.025f, 0.020f, 0.010f, 2.00f, 6.00f},
        {0.020f, 0.005f, -0.010f, 0.008f, 0.75f, 3.00f},
    };

    const auto check_fitted_curve = [&](const auto& curves,
                                        auto payer_launcher,
                                        auto receiver_launcher,
                                        const char* payer_message,
                                        const char* receiver_message) {
        for (const bool payer : {true, false}) {
            const auto expected = [&](std::size_t model_index,
                                      std::size_t curve_index,
                                      std::size_t product_index) {
                const HullWhiteModel& model =
                    hull_white_models[model_index];
                const auto& curve = curves[curve_index];
                return jamshidian_price(
                    products[product_index],
                    payer,
                    [&](double state, double start, double maturity) {
                        return hull_white_zero_coupon(
                            model, curve, state, start, maturity
                        );
                    },
                    [&](double sign,
                        double expiry,
                        double maturity,
                        double strike) {
                        return hull_white_bond_option(
                            model,
                            curve,
                            sign,
                            expiry,
                            maturity,
                            strike
                        );
                    }
                );
            };
            if (payer) {
                check_hull_white_launcher(
                    hull_white_models,
                    curves,
                    fixtures,
                    payer_launcher,
                    expected,
                    payer_message
                );
            } else {
                check_hull_white_launcher(
                    hull_white_models,
                    curves,
                    fixtures,
                    receiver_launcher,
                    expected,
                    receiver_message
                );
            }
        }
    };

    check_fitted_curve(
        nelson_siegel_curves,
        static_cast<RegularHullWhiteLauncher<NelsonSiegelCurve>>(
            hw_ns::launch_hull_white_nelson_siegel_european_swaption_cuda<
                SwaptionSide::payer
            >
        ),
        static_cast<RegularHullWhiteLauncher<NelsonSiegelCurve>>(
            hw_ns::launch_hull_white_nelson_siegel_european_swaption_cuda<
                SwaptionSide::receiver
            >
        ),
        "Hull-White Nelson-Siegel payer swaption differs from FP64",
        "Hull-White Nelson-Siegel receiver swaption differs from FP64"
    );
    check_fitted_curve(
        svensson_curves,
        static_cast<RegularHullWhiteLauncher<SvenssonCurve>>(
            hw_sv::launch_hull_white_svensson_european_swaption_cuda<
                SwaptionSide::payer
            >
        ),
        static_cast<RegularHullWhiteLauncher<SvenssonCurve>>(
            hw_sv::launch_hull_white_svensson_european_swaption_cuda<
                SwaptionSide::receiver
            >
        ),
        "Hull-White Svensson payer swaption differs from FP64",
        "Hull-White Svensson receiver swaption differs from FP64"
    );
}
