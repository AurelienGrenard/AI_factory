// Compile-time coverage of the canonical fixed-income analytics surface.
#include "common/check_cuda.cuh"
#include "common/fixed_income/analytics_concepts.cuh"
#include "common/fixed_income/cashflows.cuh"
#include "curve/nelson_siegel/term_structure_impl.cuh"
#include "curve/svensson/term_structure_impl.cuh"
#include "model/fixed_income/cir/analytics_impl.cuh"
#include "model/fixed_income/g2/analytics_impl.cuh"
#include "model/fixed_income/g2_plus_plus/nelson_siegel/analytics.cuh"
#include "model/fixed_income/g2_plus_plus/svensson/analytics.cuh"
#include "model/fixed_income/hull_white/nelson_siegel/analytics.cuh"
#include "model/fixed_income/hull_white/svensson/analytics.cuh"
#include "model/fixed_income/ornstein_uhlenbeck/analytics_impl.cuh"
#include "model/fixed_income/vasicek/analytics_impl.cuh"

#include <concepts>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

namespace fi = ai_factory::workbench::fixed_income;
namespace ou =
    ai_factory::workbench::model::fixed_income::ornstein_uhlenbeck;
namespace vasicek =
    ai_factory::workbench::model::fixed_income::vasicek;
namespace cir = ai_factory::workbench::model::fixed_income::cir;
namespace g2 = ai_factory::workbench::model::fixed_income::g2;
namespace hw_ns =
    ai_factory::workbench::model::fixed_income::hull_white::nelson_siegel;
namespace hw_sv =
    ai_factory::workbench::model::fixed_income::hull_white::svensson;
namespace g2pp_ns =
    ai_factory::workbench::model::fixed_income::g2_plus_plus::nelson_siegel;
namespace g2pp_sv =
    ai_factory::workbench::model::fixed_income::g2_plus_plus::svensson;

using Schedule = fi::FixedLegScheduleView;

#define AI_FACTORY_ASSERT_FIXED_INCOME_ANALYTICS(                         \
    prefix, parameters_type, state_type, bond_loadings_type              \
)                                                                       \
    static_assert(requires(                                              \
        const parameters_type& parameters,                              \
        const state_type& state,                                        \
        const Schedule& schedule,                                       \
        float time,                                                      \
        float state_integral,                                            \
        float maturity,                                                  \
        float option_expiry,                                             \
        float strike                                                     \
    ) {                                                                  \
        { prefix::short_rate(parameters, state, time) }                  \
            -> std::same_as<float>;                                      \
        { prefix::log_A(parameters, time, maturity) }                    \
            -> std::same_as<float>;                                      \
        { prefix::A(parameters, time, maturity) }                        \
            -> std::same_as<float>;                                      \
        { prefix::B(parameters, time, maturity) }                        \
            -> std::same_as<bond_loadings_type>;                         \
        { prefix::log_zero_coupon_bond(                                  \
            parameters, state, time, maturity                            \
        ) } -> std::same_as<float>;                                      \
        { prefix::log_discount_factor(                                   \
            parameters, state_integral, time                             \
        ) } -> std::same_as<float>;                                      \
        { prefix::discount_factor(                                       \
            parameters, state_integral, time                             \
        ) } -> std::same_as<float>;                                      \
        { prefix::zero_coupon_bond(                                      \
            parameters, state, time, maturity                            \
        ) } -> std::same_as<float>;                                      \
        { prefix::zero_coupon_bond_call_price(                           \
            parameters, state, time, option_expiry, maturity, strike     \
        ) } -> std::same_as<float>;                                      \
        { prefix::zero_coupon_bond_put_price(                            \
            parameters, state, time, option_expiry, maturity, strike     \
        ) } -> std::same_as<float>;                                      \
        { prefix::forward_rate(                                          \
            parameters, state, time, option_expiry, maturity, 0.5f       \
        ) } -> std::same_as<float>;                                      \
        { prefix::swap_rate(                                             \
            parameters, state, time, option_expiry, schedule             \
        ) } -> std::same_as<float>;                                      \
        { prefix::payer_swap_value(                                      \
            parameters, state, time, option_expiry, strike, schedule     \
        ) } -> std::same_as<float>;                                      \
    })

#define AI_FACTORY_ASSERT_JAMSHIDIAN_ANALYTICS(                          \
    prefix, parameters_type, state_type                                  \
)                                                                       \
    static_assert(requires(                                              \
        const parameters_type& parameters,                              \
        const state_type& state,                                        \
        const Schedule& schedule,                                       \
        float time,                                                      \
        float exercise_time,                                             \
        float maturity,                                                  \
        float fixed_rate                                                 \
    ) {                                                                  \
        { prefix::jamshidian_state_boundary(                             \
            parameters, exercise_time, fixed_rate, schedule              \
        ) } -> std::same_as<float>;                                      \
        { prefix::jamshidian_bond_strike(                                \
            parameters, exercise_time, maturity, state                   \
        ) } -> std::same_as<float>;                                      \
        { prefix::european_payer_swaption_price(                         \
            parameters, state, time, exercise_time, fixed_rate, schedule \
        ) } -> std::same_as<float>;                                      \
        { prefix::european_receiver_swaption_price(                      \
            parameters, state, time, exercise_time, fixed_rate, schedule \
        ) } -> std::same_as<float>;                                      \
    })

AI_FACTORY_ASSERT_FIXED_INCOME_ANALYTICS(
    ou, ou::ModelParameters, float, float
);
AI_FACTORY_ASSERT_FIXED_INCOME_ANALYTICS(
    vasicek, vasicek::ModelParameters, float, float
);
AI_FACTORY_ASSERT_FIXED_INCOME_ANALYTICS(
    cir, cir::ModelParameters, float, float
);
AI_FACTORY_ASSERT_FIXED_INCOME_ANALYTICS(
    hw_ns, hw_ns::HullWhiteFittedParameters, float, float
);
AI_FACTORY_ASSERT_FIXED_INCOME_ANALYTICS(
    hw_sv, hw_sv::HullWhiteFittedParameters, float, float
);
AI_FACTORY_ASSERT_FIXED_INCOME_ANALYTICS(
    g2, g2::ModelParameters, g2::State, g2::TwoFactorAffineBondLoadings
);
AI_FACTORY_ASSERT_FIXED_INCOME_ANALYTICS(
    g2pp_ns,
    g2pp_ns::G2PlusPlusFittedParameters,
    g2::State,
    g2::TwoFactorAffineBondLoadings
);
AI_FACTORY_ASSERT_FIXED_INCOME_ANALYTICS(
    g2pp_sv,
    g2pp_sv::G2PlusPlusFittedParameters,
    g2::State,
    g2::TwoFactorAffineBondLoadings
);

AI_FACTORY_ASSERT_JAMSHIDIAN_ANALYTICS(
    ou, ou::ModelParameters, float
);
AI_FACTORY_ASSERT_JAMSHIDIAN_ANALYTICS(
    vasicek, vasicek::ModelParameters, float
);
AI_FACTORY_ASSERT_JAMSHIDIAN_ANALYTICS(
    cir, cir::ModelParameters, float
);
AI_FACTORY_ASSERT_JAMSHIDIAN_ANALYTICS(
    hw_ns, hw_ns::HullWhiteFittedParameters, float
);
AI_FACTORY_ASSERT_JAMSHIDIAN_ANALYTICS(
    hw_sv, hw_sv::HullWhiteFittedParameters, float
);

static_assert(fi::ParametricCurveProvider<
    hw_ns::CurveAnalyticsProvider,
    ai_factory::workbench::curve::nelson_siegel::NelsonSiegelParameters
>);
static_assert(fi::ParametricCurveProvider<
    hw_sv::CurveAnalyticsProvider,
    ai_factory::workbench::curve::svensson::SvenssonParameters
>);
static_assert(fi::JamshidianAnalyticsProvider<
    hw_ns::FittedAnalyticsProvider,
    hw_ns::HullWhiteFittedParameters,
    float
>);
static_assert(fi::JamshidianAnalyticsProvider<
    hw_sv::FittedAnalyticsProvider,
    hw_sv::HullWhiteFittedParameters,
    float
>);
static_assert(fi::ZeroCouponBondProvider<
    g2pp_ns::FittedAnalyticsProvider,
    g2pp_ns::G2PlusPlusFittedParameters,
    g2::State
>);
static_assert(fi::ZeroCouponBondProvider<
    g2pp_sv::FittedAnalyticsProvider,
    g2pp_sv::G2PlusPlusFittedParameters,
    g2::State
>);
static_assert(fi::BondOptionProvider<
    g2pp_ns::FittedAnalyticsProvider,
    g2pp_ns::G2PlusPlusFittedParameters,
    g2::State
>);
static_assert(fi::BondOptionProvider<
    g2pp_sv::FittedAnalyticsProvider,
    g2pp_sv::G2PlusPlusFittedParameters,
    g2::State
>);

struct AnalyticsMatrixRow {
    float short_rate_value;
    float A_log_identity_error;
    float affine_bond_identity_error;
    float log_bond_identity_error;
    float path_discount_identity_error;
    float forward_rate_identity_error;
    float swap_rate_identity_error;
    float payer_swap_value_identity_error;
    float bond_option_parity_error;
};

__device__ __forceinline__ float affine_state_loading(
    float loadings,
    float state
) {
    return loadings * state;
}

__device__ __forceinline__ float affine_state_loading(
    const g2::TwoFactorAffineBondLoadings& loadings,
    const g2::State& state
) {
    return fmaf(
        loadings.state_x,
        state.state_x,
        loadings.state_y * state.state_y
    );
}

#define AI_FACTORY_DEFINE_ANALYTICS_FACADE(                              \
    name, prefix, parameters_type, state_type                            \
)                                                                       \
    struct name {                                                        \
        using Parameters = parameters_type;                             \
        using State = state_type;                                       \
        __device__ __forceinline__ static float short_rate(              \
            const Parameters& parameters, const State& state, float time \
        ) { return prefix::short_rate(parameters, state, time); }         \
        __device__ __forceinline__ static float log_A(                   \
            const Parameters& parameters, float time, float maturity     \
        ) { return prefix::log_A(parameters, time, maturity); }           \
        __device__ __forceinline__ static float A(                       \
            const Parameters& parameters, float time, float maturity     \
        ) { return prefix::A(parameters, time, maturity); }               \
        __device__ __forceinline__ static auto B(                        \
            const Parameters& parameters, float time, float maturity     \
        ) { return prefix::B(parameters, time, maturity); }               \
        __device__ __forceinline__ static float log_zero_coupon_bond(    \
            const Parameters& parameters, const State& state,            \
            float time, float maturity                                   \
        ) {                                                              \
            return prefix::log_zero_coupon_bond(                         \
                parameters, state, time, maturity                        \
            );                                                           \
        }                                                                \
        __device__ __forceinline__ static float log_discount_factor(     \
            const Parameters& parameters, float integral, float time     \
        ) {                                                              \
            return prefix::log_discount_factor(                          \
                parameters, integral, time                               \
            );                                                           \
        }                                                                \
        __device__ __forceinline__ static float discount_factor(         \
            const Parameters& parameters, float integral, float time     \
        ) {                                                              \
            return prefix::discount_factor(parameters, integral, time);  \
        }                                                                \
        __device__ __forceinline__ static float zero_coupon_bond(        \
            const Parameters& parameters, const State& state,            \
            float time, float maturity                                   \
        ) {                                                              \
            return prefix::zero_coupon_bond(                             \
                parameters, state, time, maturity                        \
            );                                                           \
        }                                                                \
        __device__ __forceinline__ static float forward_rate(            \
            const Parameters& parameters, const State& state,            \
            float time, float start_time, float end_time, float accrual  \
        ) {                                                              \
            return prefix::forward_rate(                                 \
                parameters, state, time, start_time, end_time, accrual   \
            );                                                           \
        }                                                                \
        template<typename ScheduleView>                                  \
        __device__ __forceinline__ static float swap_rate(               \
            const Parameters& parameters, const State& state,            \
            float time, float start_time, const ScheduleView& schedule   \
        ) {                                                              \
            return prefix::swap_rate(                                    \
                parameters, state, time, start_time, schedule            \
            );                                                           \
        }                                                                \
        template<typename ScheduleView>                                  \
        __device__ __forceinline__ static float payer_swap_value(        \
            const Parameters& parameters, const State& state,            \
            float time, float start_time, float fixed_rate,              \
            const ScheduleView& schedule                                 \
        ) {                                                              \
            return prefix::payer_swap_value(                             \
                parameters, state, time, start_time, fixed_rate, schedule\
            );                                                           \
        }                                                                \
        __device__ __forceinline__ static float bond_call(               \
            const Parameters& parameters, const State& state,            \
            float time, float expiry, float maturity, float strike       \
        ) {                                                              \
            return prefix::zero_coupon_bond_call_price(                  \
                parameters, state, time, expiry, maturity, strike        \
            );                                                           \
        }                                                                \
        __device__ __forceinline__ static float bond_put(                \
            const Parameters& parameters, const State& state,            \
            float time, float expiry, float maturity, float strike       \
        ) {                                                              \
            return prefix::zero_coupon_bond_put_price(                   \
                parameters, state, time, expiry, maturity, strike        \
            );                                                           \
        }                                                                \
    }

AI_FACTORY_DEFINE_ANALYTICS_FACADE(
    OuAnalytics, ou, ou::ModelParameters, float
);
AI_FACTORY_DEFINE_ANALYTICS_FACADE(
    VasicekAnalytics, vasicek, vasicek::ModelParameters, float
);
AI_FACTORY_DEFINE_ANALYTICS_FACADE(
    CirAnalytics, cir, cir::ModelParameters, float
);
AI_FACTORY_DEFINE_ANALYTICS_FACADE(
    HullWhiteNelsonSiegelAnalytics,
    hw_ns,
    hw_ns::HullWhiteFittedParameters,
    float
);
AI_FACTORY_DEFINE_ANALYTICS_FACADE(
    HullWhiteSvenssonAnalytics,
    hw_sv,
    hw_sv::HullWhiteFittedParameters,
    float
);
AI_FACTORY_DEFINE_ANALYTICS_FACADE(
    G2Analytics, g2, g2::ModelParameters, g2::State
);
AI_FACTORY_DEFINE_ANALYTICS_FACADE(
    G2PlusPlusNelsonSiegelAnalytics,
    g2pp_ns,
    g2pp_ns::G2PlusPlusFittedParameters,
    g2::State
);
AI_FACTORY_DEFINE_ANALYTICS_FACADE(
    G2PlusPlusSvenssonAnalytics,
    g2pp_sv,
    g2pp_sv::G2PlusPlusFittedParameters,
    g2::State
);

template<typename Analytics>
__device__ __forceinline__ AnalyticsMatrixRow evaluate_analytics_matrix_row(
    const typename Analytics::Parameters& parameters,
    const typename Analytics::State& state,
    const Schedule& schedule
) {
    constexpr float valuation_time = 0.25f;
    constexpr float option_expiry = 0.75f;
    constexpr float forward_end_time = 1.0f;
    constexpr float bond_maturity = 2.0f;
    constexpr float accrual_fraction = 0.25f;
    constexpr float fixed_rate = 0.04f;
    constexpr float bond_strike = 0.92f;
    constexpr float state_integral = 0.05f;

    const float log_A_value = Analytics::log_A(
        parameters, valuation_time, bond_maturity
    );
    const float A_value = Analytics::A(
        parameters, valuation_time, bond_maturity
    );
    const auto loadings = Analytics::B(
        parameters, valuation_time, bond_maturity
    );
    const float log_bond = Analytics::log_zero_coupon_bond(
        parameters, state, valuation_time, bond_maturity
    );
    const float bond = Analytics::zero_coupon_bond(
        parameters, state, valuation_time, bond_maturity
    );
    const float log_path_discount = Analytics::log_discount_factor(
        parameters, state_integral, option_expiry
    );
    const float path_discount = Analytics::discount_factor(
        parameters, state_integral, option_expiry
    );
    const float forward_start_bond = Analytics::zero_coupon_bond(
        parameters, state, valuation_time, option_expiry
    );
    const float forward_end_bond = Analytics::zero_coupon_bond(
        parameters, state, valuation_time, forward_end_time
    );
    const float forward = Analytics::forward_rate(
        parameters,
        state,
        valuation_time,
        option_expiry,
        forward_end_time,
        accrual_fraction
    );

    double annuity = 0.0;
    for (std::uint32_t payment = 0U;
         payment < schedule.payment_count();
         ++payment) {
        annuity += static_cast<double>(schedule.accrual_fraction(payment))
            * static_cast<double>(Analytics::zero_coupon_bond(
                parameters,
                state,
                valuation_time,
                schedule.payment_time(payment)
            ));
    }
    const float swap_end_bond = Analytics::zero_coupon_bond(
        parameters,
        state,
        valuation_time,
        schedule.payment_time(schedule.payment_count() - 1U)
    );
    const float swap_numerator = forward_start_bond - swap_end_bond;
    const float swap = Analytics::swap_rate(
        parameters, state, valuation_time, option_expiry, schedule
    );
    const float payer_value = Analytics::payer_swap_value(
        parameters,
        state,
        valuation_time,
        option_expiry,
        fixed_rate,
        schedule
    );
    const float call = Analytics::bond_call(
        parameters,
        state,
        valuation_time,
        option_expiry,
        bond_maturity,
        bond_strike
    );
    const float put = Analytics::bond_put(
        parameters,
        state,
        valuation_time,
        option_expiry,
        bond_maturity,
        bond_strike
    );

    return {
        Analytics::short_rate(parameters, state, valuation_time),
        fabsf(expf(log_A_value) - A_value),
        fabsf(
            expf(
                log_A_value - affine_state_loading(loadings, state)
            ) - bond
        ),
        fabsf(expf(log_bond) - bond),
        fabsf(expf(log_path_discount) - path_discount),
        fabsf(
            forward - (forward_start_bond / forward_end_bond - 1.0f)
                / accrual_fraction
        ),
        fabsf(swap - swap_numerator / static_cast<float>(annuity)),
        fabsf(
            payer_value
            - (swap_numerator - fixed_rate * static_cast<float>(annuity))
        ),
        fabsf(
            call - put
            - (bond - bond_strike * forward_start_bond)
        ),
    };
}

__global__ void fixed_income_analytics_matrix_kernel(
    AnalyticsMatrixRow* rows
) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) return;

    constexpr float payment_times[] = {1.0f, 1.5f, 2.0f};
    constexpr float accrual_fractions[] = {0.25f, 0.5f, 0.5f};
    const Schedule schedule{payment_times, accrual_fractions, 3U};

    const ou::ModelParameters ou_parameters{{0.15f, 0.01f}, 0.03f};
    const vasicek::ModelParameters vasicek_parameters{
        {0.15f, 0.04f, 0.01f}, 0.03f,
    };
    const cir::ModelParameters cir_parameters{
        {0.60f, 0.04f, 0.15f}, 0.03f,
    };
    const ai_factory::workbench::model::fixed_income::hull_white::ModelParameters
        hull_white_parameters{0.15f, 0.01f};
    const ai_factory::workbench::curve::nelson_siegel::NelsonSiegelParameters
        nelson_siegel_curve{0.03f, -0.01f, 0.02f, 2.0f};
    const ai_factory::workbench::curve::svensson::SvenssonParameters
        svensson_curve{0.03f, -0.01f, 0.02f, -0.005f, 2.0f, 5.0f};
    const g2::ModelParameters g2_parameters{
        {0.15f, 0.01f, 0.70f, 0.008f, -0.40f},
        {0.01f, -0.005f},
    };
    const ai_factory::workbench::model::fixed_income::g2_plus_plus::ModelParameters
        g2_plus_plus_parameters{{
            0.15f, 0.01f, 0.70f, 0.008f, -0.40f,
        }};

    rows[0] = evaluate_analytics_matrix_row<OuAnalytics>(
        ou_parameters, 0.02f, schedule
    );
    rows[1] = evaluate_analytics_matrix_row<VasicekAnalytics>(
        vasicek_parameters, 0.02f, schedule
    );
    rows[2] = evaluate_analytics_matrix_row<CirAnalytics>(
        cir_parameters, 0.025f, schedule
    );
    rows[3] = evaluate_analytics_matrix_row<
        HullWhiteNelsonSiegelAnalytics
    >(
        hw_ns::compose_fitted_model(
            hull_white_parameters, nelson_siegel_curve
        ),
        0.0f,
        schedule
    );
    rows[4] = evaluate_analytics_matrix_row<HullWhiteSvenssonAnalytics>(
        hw_sv::compose_fitted_model(
            hull_white_parameters, svensson_curve
        ),
        0.0f,
        schedule
    );
    rows[5] = evaluate_analytics_matrix_row<G2Analytics>(
        g2_parameters, g2_parameters.initial_state, schedule
    );
    rows[6] = evaluate_analytics_matrix_row<
        G2PlusPlusNelsonSiegelAnalytics
    >(
        g2pp_ns::compose_fitted_model(
            g2_plus_plus_parameters, nelson_siegel_curve
        ),
        g2::State{0.0f, 0.0f},
        schedule
    );
    rows[7] = evaluate_analytics_matrix_row<
        G2PlusPlusSvenssonAnalytics
    >(
        g2pp_sv::compose_fitted_model(
            g2_plus_plus_parameters, svensson_curve
        ),
        g2::State{0.0f, 0.0f},
        schedule
    );
}

#undef AI_FACTORY_ASSERT_JAMSHIDIAN_ANALYTICS
#undef AI_FACTORY_ASSERT_FIXED_INCOME_ANALYTICS
#undef AI_FACTORY_DEFINE_ANALYTICS_FACADE

}  // namespace

int main() {
    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) {
        return 77;
    }
    ai_factory::workbench::check_cuda(
        availability, "analytics contract cudaGetDeviceCount"
    );

    constexpr std::size_t row_count = 8U;
    AnalyticsMatrixRow* device_rows = nullptr;
    ai_factory::workbench::check_cuda(
        cudaMalloc(&device_rows, row_count * sizeof(AnalyticsMatrixRow)),
        "analytics contract row allocation"
    );
    fixed_income_analytics_matrix_kernel<<<1U, 1U>>>(device_rows);
    ai_factory::workbench::check_cuda(
        cudaGetLastError(), "analytics contract kernel"
    );
    AnalyticsMatrixRow rows[row_count]{};
    ai_factory::workbench::check_cuda(
        cudaMemcpy(
            rows,
            device_rows,
            sizeof(rows),
            cudaMemcpyDeviceToHost
        ),
        "analytics contract row copy"
    );
    ai_factory::workbench::check_cuda(
        cudaFree(device_rows), "analytics contract row free"
    );

    constexpr const char* model_names[row_count] = {
        "OU", "Vasicek", "CIR", "Hull-White Nelson-Siegel",
        "Hull-White Svensson", "G2", "G2++ Nelson-Siegel",
        "G2++ Svensson",
    };
    for (std::size_t row = 0U; row < row_count; ++row) {
        if (!std::isfinite(rows[row].short_rate_value)) {
            throw std::runtime_error(
                std::string(model_names[row]) + " short rate is not finite"
            );
        }
        const float errors[] = {
            rows[row].A_log_identity_error,
            rows[row].affine_bond_identity_error,
            rows[row].log_bond_identity_error,
            rows[row].path_discount_identity_error,
            rows[row].forward_rate_identity_error,
            rows[row].swap_rate_identity_error,
            rows[row].payer_swap_value_identity_error,
            rows[row].bond_option_parity_error,
        };
        for (float error : errors) {
            if (!std::isfinite(error) || error > 3.0e-5f) {
                throw std::runtime_error(
                    std::string(model_names[row])
                    + " analytics matrix identity failed"
                );
            }
        }
    }
    return 0;
}
