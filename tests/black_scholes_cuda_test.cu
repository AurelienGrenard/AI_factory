// Verify exact Black-Scholes dynamics and closed-form product identities.
#include "common/check_cuda.cuh"
#include "common/equity/handlers.cuh"
#include "common/simulation/path_simulation.cuh"
#include "model/equity/black_scholes/asset_or_nothing_option.cuh"
#include "model/equity/black_scholes/analytics.cu"
#include "model/equity/black_scholes/digital_option.cuh"
#include "model/equity/black_scholes/dynamics.cu"
#include "model/equity/black_scholes/european_option.cuh"
#include "model/equity/black_scholes/forward_start_option.cuh"
#include "model/equity/black_scholes/gap_option.cuh"
#include "model/equity/black_scholes/geometric_asian_option.cuh"
#include "model/equity/black_scholes/range_accrual.cuh"
#include "model/equity/black_scholes/straddle.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace {

using ai_factory::workbench::model::equity::black_scholes::ModelParameters;

struct DynamicsResults {
    float prepared_drift;
    float prepared_standard_deviation;
    float transitioned_log_spot;
    float terminal_first;
    float terminal_replay;
    float first_time;
    float second_time;
};

struct AnalyticsResults {
    float d1;
    float d2;
    float total_volatility;
    float zero_volatility_call;
    float zero_volatility_put;
    float interval_probabilities[3];
};

__global__ void exercise_dynamics_kernel(DynamicsResults* output) {
    using namespace ai_factory::workbench;
    const ModelParameters parameters = {
        1.25f, 0.04f, 0.01f, 0.20f,
    };
    const ai_factory::workbench::model::equity::black_scholes::PreparedModel prepared =
        ai_factory::workbench::model::equity::black_scholes::prepare_model(parameters);
    const ai_factory::workbench::model::equity::black_scholes::PreparedTransition quarter =
        ai_factory::workbench::model::equity::black_scholes::prepare_transition(prepared, 0.25f);
    ai_factory::workbench::model::equity::black_scholes::State transitioned =
        ai_factory::workbench::model::equity::black_scholes::initial_state(prepared);
    ai_factory::workbench::model::equity::black_scholes::one_step_transition(quarter, 0.4f, transitioned);
    const philox::PhiloxKey key = philox::make_key(900000001ULL);
    const float terminal_first =
        simulation::simulate_exact_transition_terminal<
            ai_factory::workbench::model::equity::black_scholes::DynamicsPolicy
        >(prepared, quarter, key, 17U).log_spot;
    const float terminal_replay =
        simulation::simulate_exact_transition_terminal<
            ai_factory::workbench::model::equity::black_scholes::DynamicsPolicy
        >(prepared, quarter, key, 17U).log_spot;
    const ai_factory::workbench::model::equity::black_scholes::PreparedTransition two_transitions[2] = {
        quarter, quarter,
    };
    float first_spot = 0.0f;
    equity::SpotObservationWriter<ai_factory::workbench::model::equity::black_scholes::DynamicsPolicy> writer{
        &first_spot, 1U, 1U,
    };
    const ai_factory::workbench::model::equity::black_scholes::State second_state =
        simulation::simulate_exact_transition_calendar<
            ai_factory::workbench::model::equity::black_scholes::DynamicsPolicy
        >(
            prepared, two_transitions, 2U, key, 23U, writer
        );
    *output = {
        quarter.drift,
        quarter.diffusion_standard_deviation,
        transitioned.log_spot,
        terminal_first,
        terminal_replay,
        first_spot,
        expf(second_state.log_spot),
    };
}

__global__ void exercise_analytics_kernel(AnalyticsResults* output) {
    using namespace ai_factory::workbench;
    namespace bs = model::equity::black_scholes;

    constexpr float maturity_years = 1.25f;
    const ModelParameters model = {1.1f, 0.04f, 0.01f, 0.23f};
    const bs::BlackScholesAnalyticsContext analytics =
        bs::prepare_analytics(model);
    const DiscountedLognormalOptionValues values =
        bs::prepare_vanilla_option_values(
            analytics, 0.95f, maturity_years
        );

    const ModelParameters deterministic_model = {
        1.2f, 0.02f, 0.01f, 0.0f,
    };
    const DiscountedLognormalOptionValues deterministic_values =
        bs::prepare_vanilla_option_values(
            bs::prepare_analytics(deterministic_model), 1.0f, 1.0f
        );

    constexpr float spots[3] = {0.75f, 1.0f, 1.4f};
    AnalyticsResults results{
        values.d1,
        values.d2,
        model.volatility * sqrtf(maturity_years),
        discounted_lognormal_option_price(deterministic_values, 1.0f),
        discounted_lognormal_option_price(deterministic_values, -1.0f),
        {},
    };
    for (std::uint32_t index = 0U; index < 3U; ++index) {
        ModelParameters interval_model = model;
        interval_model.spot = spots[index];
        results.interval_probabilities[index] =
            bs::lognormal_interval_probability(
                bs::prepare_analytics(interval_model),
                0.8f,
                1.2f,
                0.75f
            );
    }
    *output = results;
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool close(float lhs, float rhs, float tolerance = 3.0e-6f) {
    return std::fabs(lhs - rhs) <= tolerance;
}

double normal_cdf(double value) {
    return 0.5 * std::erfc(-value / std::sqrt(2.0));
}

double black_price(
    double discounted_underlying,
    double discounted_strike,
    double total_volatility,
    double option_sign
) {
    if (total_volatility <= 1.0e-14) {
        return std::max(
            option_sign * (
                discounted_underlying - discounted_strike
            ),
            0.0
        );
    }
    const double d1 = std::log(
        discounted_underlying / discounted_strike
    ) / total_volatility + 0.5 * total_volatility;
    const double d2 = d1 - total_volatility;
    return option_sign * (
        discounted_underlying * normal_cdf(option_sign * d1)
        - discounted_strike * normal_cdf(option_sign * d2)
    );
}

double interval_probability(
    const ModelParameters& model,
    double lower_level,
    double upper_level,
    double observation_time_years
) {
    const double variance = static_cast<double>(model.volatility)
        * static_cast<double>(model.volatility);
    const double mean = std::log(static_cast<double>(model.spot))
        + (static_cast<double>(model.risk_free_rate)
            - static_cast<double>(model.dividend_yield)
            - 0.5 * variance) * observation_time_years;
    const double standard_deviation =
        static_cast<double>(model.volatility)
        * std::sqrt(observation_time_years);
    if (standard_deviation <= 1.0e-14) {
        return mean > std::log(lower_level)
            && mean < std::log(upper_level) ? 1.0 : 0.0;
    }
    return normal_cdf((std::log(upper_level) - mean) / standard_deviation)
        - normal_cdf((std::log(lower_level) - mean) / standard_deviation);
}

double range_accrual_price(
    const ModelParameters& model,
    const ai_factory::workbench::product::RangeAccrualParameters& product
) {
    constexpr double day_fraction = 1.0 / 252.0;
    const double maturity_years =
        static_cast<double>(product.maturity) * day_fraction;
    const double observation_interval_years =
        static_cast<double>(product.observation_interval) * day_fraction;
    const std::uint32_t observation_count =
        product.maturity / product.observation_interval;
    double probability_sum = 0.0;
    for (std::uint32_t observation = 1U;
         observation <= observation_count;
         ++observation) {
        probability_sum += interval_probability(
            model,
            product.lower_barrier,
            product.upper_barrier,
            static_cast<double>(observation)
                * observation_interval_years
        );
    }
    return std::exp(-static_cast<double>(model.risk_free_rate)
        * maturity_years) * (
            1.0 + static_cast<double>(product.coupon_rate)
                * observation_interval_years * probability_sum
        );
}

template <typename Product, typename Launcher>
float price_one(
    const ModelParameters& model,
    const Product& product,
    Launcher launch
) {
    using namespace ai_factory::workbench;
    ModelParameters* device_model = nullptr;
    Product* device_product = nullptr;
    float* device_price = nullptr;
    check_cuda(cudaMalloc(&device_model, sizeof(model)), "BS test model allocation");
    check_cuda(cudaMalloc(&device_product, sizeof(product)), "BS test product allocation");
    check_cuda(cudaMalloc(&device_price, sizeof(float)), "BS test price allocation");
    check_cuda(cudaMemcpy(
        device_model, &model, sizeof(model), cudaMemcpyHostToDevice
    ), "BS test model copy");
    check_cuda(cudaMemcpy(
        device_product, &product, sizeof(product), cudaMemcpyHostToDevice
    ), "BS test product copy");
    launch(
        device_model, 1U, device_product, 1U, false, 1U, 0U, 1U,
        1.0f / 252.0f, 32U, 1U, device_price
    );
    check_cuda(cudaDeviceSynchronize(), "BS analytical kernel synchronize");
    float price = 0.0f;
    check_cuda(cudaMemcpy(
        &price, device_price, sizeof(price), cudaMemcpyDeviceToHost
    ), "BS test price copy");
    check_cuda(cudaFree(device_model), "BS test model free");
    check_cuda(cudaFree(device_product), "BS test product free");
    check_cuda(cudaFree(device_price), "BS test price free");
    return price;
}

float geometric_price_one(
    const ModelParameters& model,
    const ai_factory::workbench::product::GeometricAsianOptionParameters& contract,
    ai_factory::workbench::OptionSide side
) {
    using namespace ai_factory::workbench;
    ModelParameters* device_model = nullptr;
    product::GeometricAsianOptionParameters* device_product = nullptr;
    float* device_price = nullptr;
    check_cuda(cudaMalloc(&device_model, sizeof(model)), "BS geometric model allocation");
    check_cuda(cudaMalloc(&device_product, sizeof(contract)), "BS geometric product allocation");
    check_cuda(cudaMalloc(&device_price, sizeof(float)), "BS geometric price allocation");
    check_cuda(cudaMemcpy(device_model, &model, sizeof(model), cudaMemcpyHostToDevice), "BS geometric model copy");
    check_cuda(cudaMemcpy(device_product, &contract, sizeof(contract), cudaMemcpyHostToDevice), "BS geometric product copy");
    if (side == OptionSide::call) {
        ai_factory::workbench::model::equity::black_scholes::launch_black_scholes_geometric_asian_option_cuda<OptionSide::call>(
            device_model, 1U, device_product, 1U, false, 1U, 0U, 1U,
            1.0f / 504.0f, 2U, 32U, 1U, device_price
        );
    } else {
        ai_factory::workbench::model::equity::black_scholes::launch_black_scholes_geometric_asian_option_cuda<OptionSide::put>(
            device_model, 1U, device_product, 1U, false, 1U, 0U, 1U,
            1.0f / 504.0f, 2U, 32U, 1U, device_price
        );
    }
    check_cuda(cudaDeviceSynchronize(), "BS geometric kernel synchronize");
    float price = 0.0f;
    check_cuda(cudaMemcpy(&price, device_price, sizeof(price), cudaMemcpyDeviceToHost), "BS geometric price copy");
    check_cuda(cudaFree(device_model), "BS geometric model free");
    check_cuda(cudaFree(device_product), "BS geometric product free");
    check_cuda(cudaFree(device_price), "BS geometric price free");
    return price;
}

void require_closed_form_grid_stride_matches_direct() {
    using namespace ai_factory::workbench;
    constexpr std::size_t model_count = 2U;
    constexpr std::size_t product_count = 3U;
    constexpr std::size_t result_count = model_count * product_count;
    const ModelParameters models[model_count] = {
        {0.9f, 0.01f, 0.00f, 0.15f},
        {1.2f, 0.04f, 0.02f, 0.35f},
    };
    const product::EuropeanOptionParameters products[product_count] = {
        {0.8f, 63U},
        {1.0f, 252U},
        {1.4f, 756U},
    };

    ModelParameters* device_models = nullptr;
    product::EuropeanOptionParameters* device_products = nullptr;
    float* device_prices = nullptr;
    check_cuda(
        cudaMalloc(&device_models, sizeof(models)),
        "BS grid-stride model allocation"
    );
    check_cuda(
        cudaMalloc(&device_products, sizeof(products)),
        "BS grid-stride product allocation"
    );
    check_cuda(
        cudaMalloc(&device_prices, result_count * sizeof(float)),
        "BS grid-stride price allocation"
    );
    check_cuda(cudaMemcpy(
        device_models, models, sizeof(models), cudaMemcpyHostToDevice
    ), "BS grid-stride model copy");
    check_cuda(cudaMemcpy(
        device_products, products, sizeof(products), cudaMemcpyHostToDevice
    ), "BS grid-stride product copy");

    float direct[result_count]{};
    float grid_stride[result_count]{};
    ai_factory::workbench::model::equity::black_scholes::launch_black_scholes_european_option_cuda<
        OptionSide::call
    >(
        device_models, model_count, device_products, product_count, true,
        result_count, 0U, result_count, 1.0f / 252.0f, 32U, 1U,
        device_prices
    );
    check_cuda(cudaMemcpy(
        direct,
        device_prices,
        sizeof(direct),
        cudaMemcpyDeviceToHost
    ), "BS direct closed-form result copy");

    ai_factory::workbench::model::equity::black_scholes::launch_black_scholes_european_option_cuda<
        OptionSide::call
    >(
        device_models, model_count, device_products, product_count, true,
        result_count, 0U, result_count, 1.0f / 252.0f, 2U, 1U,
        device_prices
    );
    check_cuda(cudaMemcpy(
        grid_stride,
        device_prices,
        sizeof(grid_stride),
        cudaMemcpyDeviceToHost
    ), "BS grid-stride closed-form result copy");

    require(
        std::memcmp(direct, grid_stride, sizeof(direct)) == 0,
        "BS closed-form grid-stride results differ from direct results"
    );
    check_cuda(cudaFree(device_models), "BS grid-stride model free");
    check_cuda(cudaFree(device_products), "BS grid-stride product free");
    check_cuda(cudaFree(device_prices), "BS grid-stride price free");
}

}  // namespace

int main() {
    using namespace ai_factory::workbench;
    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) return 77;
    check_cuda(availability, "Black-Scholes test cudaGetDeviceCount");

    require_closed_form_grid_stride_matches_direct();

    DynamicsResults* device_results = nullptr;
    check_cuda(cudaMalloc(&device_results, sizeof(DynamicsResults)), "BS dynamics allocation");
    exercise_dynamics_kernel<<<1, 1>>>(device_results);
    check_cuda(cudaGetLastError(), "BS dynamics kernel");
    DynamicsResults results{};
    check_cuda(cudaMemcpy(&results, device_results, sizeof(results), cudaMemcpyDeviceToHost), "BS dynamics result copy");
    check_cuda(cudaFree(device_results), "BS dynamics result free");

    const float expected_drift = (0.04f - 0.01f - 0.5f * 0.20f * 0.20f) * 0.25f;
    const float expected_stddev = 0.20f * std::sqrt(0.25f);
    require(close(results.prepared_drift, expected_drift), "BS exact drift is incorrect");
    require(close(results.prepared_standard_deviation, expected_stddev), "BS exact variance is incorrect");
    require(close(results.transitioned_log_spot, std::log(1.25f) + expected_drift + expected_stddev * 0.4f), "BS one-step transition is incorrect");
    require(results.terminal_first == results.terminal_replay, "BS terminal simulation is not reproducible");
    require(results.second_time != results.first_time, "BS two-time simulation did not advance");

    AnalyticsResults* device_analytics_results = nullptr;
    check_cuda(
        cudaMalloc(&device_analytics_results, sizeof(AnalyticsResults)),
        "BS analytics allocation"
    );
    exercise_analytics_kernel<<<1, 1>>>(device_analytics_results);
    check_cuda(cudaGetLastError(), "BS analytics kernel");
    AnalyticsResults analytics_results{};
    check_cuda(
        cudaMemcpy(
            &analytics_results,
            device_analytics_results,
            sizeof(analytics_results),
            cudaMemcpyDeviceToHost
        ),
        "BS analytics result copy"
    );
    check_cuda(
        cudaFree(device_analytics_results),
        "BS analytics result free"
    );
    require(
        close(
            analytics_results.d2,
            analytics_results.d1 - analytics_results.total_volatility
        ),
        "BS analytics d2 identity failed"
    );
    const ModelParameters deterministic_model = {
        1.2f, 0.02f, 0.01f, 0.0f,
    };
    const double deterministic_discounted_spot =
        static_cast<double>(deterministic_model.spot)
        * std::exp(-static_cast<double>(deterministic_model.dividend_yield));
    const double deterministic_discounted_strike =
        std::exp(-static_cast<double>(deterministic_model.risk_free_rate));
    require(
        close(
            analytics_results.zero_volatility_call,
            static_cast<float>(std::max(
                deterministic_discounted_spot
                    - deterministic_discounted_strike,
                0.0
            ))
        ),
        "common lognormal zero-volatility call limit failed"
    );
    require(
        close(
            analytics_results.zero_volatility_put,
            static_cast<float>(std::max(
                deterministic_discounted_strike
                    - deterministic_discounted_spot,
                0.0
            ))
        ),
        "common lognormal zero-volatility put limit failed"
    );
    constexpr float analytics_spots[3] = {0.75f, 1.0f, 1.4f};
    for (std::size_t index = 0U; index < 3U; ++index) {
        ModelParameters interval_model = {
            analytics_spots[index], 0.04f, 0.01f, 0.23f,
        };
        require(
            close(
                analytics_results.interval_probabilities[index],
                static_cast<float>(interval_probability(
                    interval_model, 0.8, 1.2, 0.75
                )),
                5.0e-6f
            ),
            "BS direct interval probability failed"
        );
    }

    const ModelParameters model = {1.0f, 0.02f, 0.01f, 0.20f};
    const product::EuropeanOptionParameters vanilla{1.0f, 252U};
    const float call = price_one(model, vanilla, ai_factory::workbench::model::equity::black_scholes::launch_black_scholes_european_option_cuda<OptionSide::call>);
    const float put = price_one(model, vanilla, ai_factory::workbench::model::equity::black_scholes::launch_black_scholes_european_option_cuda<OptionSide::put>);
    const float discount = std::exp(-model.risk_free_rate);
    const float discounted_spot = std::exp(-model.dividend_yield);
    require(close(call - put, discounted_spot - discount), "BS put-call parity failed");

    const float straddle = price_one(model, product::StraddleParameters{1.0f, 252U}, ai_factory::workbench::model::equity::black_scholes::launch_black_scholes_straddle_cuda);
    require(close(straddle, call + put), "BS straddle identity failed");
    const float gap_call = price_one(model, product::GapOptionParameters{1.0f, 1.0f, 252U}, ai_factory::workbench::model::equity::black_scholes::launch_black_scholes_gap_option_cuda<OptionSide::call>);
    require(close(gap_call, call), "BS zero-gap call identity failed");

    const product::GapOptionParameters nontrivial_gap{1.1f, 0.9f, 252U};
    const float nontrivial_gap_call = price_one(
        model,
        nontrivial_gap,
        ai_factory::workbench::model::equity::black_scholes::launch_black_scholes_gap_option_cuda<OptionSide::call>
    );
    const float threshold_asset_call = price_one(
        model,
        product::AssetOrNothingOptionParameters{1.1f, 252U},
        ai_factory::workbench::model::equity::black_scholes::launch_black_scholes_asset_or_nothing_option_cuda<OptionSide::call>
    );
    const float threshold_cash_call = price_one(
        model,
        product::DigitalOptionParameters{1.1f, 252U, 0.9f},
        ai_factory::workbench::model::equity::black_scholes::launch_black_scholes_digital_option_cuda<OptionSide::call>
    );
    require(
        close(nontrivial_gap_call, threshold_asset_call - threshold_cash_call),
        "BS nontrivial gap call decomposition failed"
    );

    const product::DigitalOptionParameters digital{1.0f, 252U, 2.0f};
    const float digital_call = price_one(model, digital, ai_factory::workbench::model::equity::black_scholes::launch_black_scholes_digital_option_cuda<OptionSide::call>);
    const float digital_put = price_one(model, digital, ai_factory::workbench::model::equity::black_scholes::launch_black_scholes_digital_option_cuda<OptionSide::put>);
    require(close(digital_call + digital_put, 2.0f * discount), "BS digital partition failed");

    const product::AssetOrNothingOptionParameters asset{1.0f, 252U};
    const float asset_call = price_one(model, asset, ai_factory::workbench::model::equity::black_scholes::launch_black_scholes_asset_or_nothing_option_cuda<OptionSide::call>);
    const float asset_put = price_one(model, asset, ai_factory::workbench::model::equity::black_scholes::launch_black_scholes_asset_or_nothing_option_cuda<OptionSide::put>);
    require(close(asset_call + asset_put, discounted_spot), "BS asset partition failed");

    const product::ForwardStartOptionParameters forward{1.0f, 126U, 252U};
    const float forward_call = price_one(model, forward, ai_factory::workbench::model::equity::black_scholes::launch_black_scholes_forward_start_option_cuda<OptionSide::call>);
    const float forward_put = price_one(model, forward, ai_factory::workbench::model::equity::black_scholes::launch_black_scholes_forward_start_option_cuda<OptionSide::put>);
    require(close(
        forward_call - forward_put,
        discounted_spot - std::exp(-model.dividend_yield * 0.5f - model.risk_free_rate * 0.5f)
    ), "BS forward-start parity failed");

    const product::GeometricAsianOptionParameters geometric{1.0f, 252U};
    const float geometric_call = geometric_price_one(model, geometric, OptionSide::call);
    const float geometric_put = geometric_price_one(model, geometric, OptionSide::put);
    require(std::isfinite(geometric_call) && std::isfinite(geometric_put), "BS geometric prices are not finite");
    constexpr double geometric_step_count = 504.0;
    constexpr double geometric_maturity_years = 1.0;
    const double geometric_variance =
        static_cast<double>(model.volatility)
        * static_cast<double>(model.volatility);
    const double geometric_log_mean = std::log(
        static_cast<double>(model.spot)
    ) + 0.5 * (
        static_cast<double>(model.risk_free_rate)
        - static_cast<double>(model.dividend_yield)
        - 0.5 * geometric_variance
    ) * geometric_maturity_years;
    const double geometric_log_variance = geometric_variance
        * geometric_maturity_years
        * (2.0 * geometric_step_count + 1.0)
        / (6.0 * (geometric_step_count + 1.0));
    const double geometric_discount = std::exp(
        -static_cast<double>(model.risk_free_rate)
            * geometric_maturity_years
    );
    const double geometric_expected_call = black_price(
        geometric_discount * std::exp(
            geometric_log_mean + 0.5 * geometric_log_variance
        ),
        geometric_discount * static_cast<double>(geometric.strike),
        std::sqrt(geometric_log_variance),
        1.0
    );
    require(
        close(
            geometric_call,
            static_cast<float>(geometric_expected_call),
            5.0e-6f
        ),
        "BS geometric Asian FP64 reference failed"
    );

    const product::RangeAccrualParameters range{
        252U, 63U, 0.8f, 1.2f, 0.05f,
    };
    constexpr float range_spots[3] = {0.75f, 1.0f, 1.4f};
    for (float range_spot : range_spots) {
        ModelParameters range_model = model;
        range_model.spot = range_spot;
        const float range_price = price_one(
            range_model,
            range,
            ai_factory::workbench::model::equity::black_scholes::launch_black_scholes_range_accrual_cuda
        );
        require(
            close(
                range_price,
                static_cast<float>(range_accrual_price(range_model, range)),
                7.0e-6f
            ),
            "BS range-accrual FP64 reference failed"
        );
    }
}
