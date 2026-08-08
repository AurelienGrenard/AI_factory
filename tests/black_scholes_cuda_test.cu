// Verify exact Black-Scholes dynamics and closed-form product identities.
#include "common/check_cuda.cuh"
#include "model/equity/black_scholes/asset_or_nothing_option.cuh"
#include "model/equity/black_scholes/digital_option.cuh"
#include "model/equity/black_scholes/dynamics.cu"
#include "model/equity/black_scholes/european_option.cuh"
#include "model/equity/black_scholes/forward_start_option.cuh"
#include "model/equity/black_scholes/gap_option.cuh"
#include "model/equity/black_scholes/geometric_asian_option.cuh"
#include "model/equity/black_scholes/range_accrual.cuh"
#include "model/equity/black_scholes/straddle.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace {

using ai_factory::workbench::black_scholes::BlackScholesModelParameters;

struct DynamicsResults {
    float prepared_drift;
    float prepared_standard_deviation;
    float transitioned_log_spot;
    float terminal_first;
    float terminal_replay;
    float first_time;
    float second_time;
    float one_step_geometric_mean;
    float one_step_geometric_terminal;
};

__global__ void exercise_dynamics_kernel(DynamicsResults* output) {
    using namespace ai_factory::workbench;
    const BlackScholesModelParameters parameters = {
        1.25f, 0.04f, 0.01f, 0.20f,
    };
    const black_scholes::BlackScholesPreparedParameters quarter =
        black_scholes::prepare_model(parameters, 0.25f);
    black_scholes::BlackScholesState transitioned =
        black_scholes::initial_state(quarter);
    black_scholes::one_step_transition(quarter, 0.4f, transitioned);
    const philox::PhiloxKey key = philox::make_key(900000001ULL);
    const float terminal_first = black_scholes::simulate_terminal_state(
        quarter, key, 17U
    ).log_spot;
    const float terminal_replay = black_scholes::simulate_terminal_state(
        quarter, key, 17U
    ).log_spot;
    const auto two_times = black_scholes::simulate_at_two_times(
        quarter, quarter, key, 23U
    );
    const auto geometric = black_scholes::simulate_geometric_mean_state(
        quarter, key, 31U, 1U
    );
    *output = {
        quarter.drift,
        quarter.standard_deviation,
        transitioned.log_spot,
        terminal_first,
        terminal_replay,
        two_times.first_state.log_spot,
        two_times.terminal_state.log_spot,
        geometric.geometric_mean,
        geometric.terminal_state.log_spot,
    };
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool close(float lhs, float rhs, float tolerance = 3.0e-6f) {
    return std::fabs(lhs - rhs) <= tolerance;
}

template <typename Product, typename Launcher>
float price_one(
    const BlackScholesModelParameters& model,
    const Product& product,
    Launcher launch
) {
    using namespace ai_factory::workbench;
    BlackScholesModelParameters* device_model = nullptr;
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
        32U, 1U, device_price
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
    const BlackScholesModelParameters& model,
    const ai_factory::workbench::product::GeometricAsianOptionParameters& contract,
    ai_factory::workbench::OptionSide side
) {
    using namespace ai_factory::workbench;
    BlackScholesModelParameters* device_model = nullptr;
    product::GeometricAsianOptionParameters* device_product = nullptr;
    float* device_price = nullptr;
    check_cuda(cudaMalloc(&device_model, sizeof(model)), "BS geometric model allocation");
    check_cuda(cudaMalloc(&device_product, sizeof(contract)), "BS geometric product allocation");
    check_cuda(cudaMalloc(&device_price, sizeof(float)), "BS geometric price allocation");
    check_cuda(cudaMemcpy(device_model, &model, sizeof(model), cudaMemcpyHostToDevice), "BS geometric model copy");
    check_cuda(cudaMemcpy(device_product, &contract, sizeof(contract), cudaMemcpyHostToDevice), "BS geometric product copy");
    if (side == OptionSide::call) {
        black_scholes::launch_black_scholes_geometric_asian_option_cuda<OptionSide::call>(
            device_model, 1U, device_product, 1U, false, 1U, 0U, 1U,
            1.0f / 360.0f, 32U, 1U, device_price
        );
    } else {
        black_scholes::launch_black_scholes_geometric_asian_option_cuda<OptionSide::put>(
            device_model, 1U, device_product, 1U, false, 1U, 0U, 1U,
            1.0f / 360.0f, 32U, 1U, device_price
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

}  // namespace

int main() {
    using namespace ai_factory::workbench;
    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) return 77;
    check_cuda(availability, "Black-Scholes test cudaGetDeviceCount");

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
    require(close(
        results.one_step_geometric_mean,
        std::exp(0.5f * (std::log(1.25f) + results.one_step_geometric_terminal))
    ), "BS direct geometric-mean simulation is inconsistent");

    const BlackScholesModelParameters model = {1.0f, 0.02f, 0.01f, 0.20f};
    const product::EuropeanOptionParameters vanilla{1.0f, 1.0f};
    const float call = price_one(model, vanilla, black_scholes::launch_black_scholes_european_option_cuda<OptionSide::call>);
    const float put = price_one(model, vanilla, black_scholes::launch_black_scholes_european_option_cuda<OptionSide::put>);
    const float discount = std::exp(-model.risk_free_rate);
    const float discounted_spot = std::exp(-model.dividend_yield);
    require(close(call - put, discounted_spot - discount), "BS put-call parity failed");

    const float straddle = price_one(model, product::StraddleParameters{1.0f, 1.0f}, black_scholes::launch_black_scholes_straddle_cuda);
    require(close(straddle, call + put), "BS straddle identity failed");
    const float gap_call = price_one(model, product::GapOptionParameters{1.0f, 1.0f, 1.0f}, black_scholes::launch_black_scholes_gap_option_cuda<OptionSide::call>);
    require(close(gap_call, call), "BS zero-gap call identity failed");

    const product::DigitalOptionParameters digital{1.0f, 1.0f, 2.0f};
    const float digital_call = price_one(model, digital, black_scholes::launch_black_scholes_digital_option_cuda<OptionSide::call>);
    const float digital_put = price_one(model, digital, black_scholes::launch_black_scholes_digital_option_cuda<OptionSide::put>);
    require(close(digital_call + digital_put, 2.0f * discount), "BS digital partition failed");

    const product::AssetOrNothingOptionParameters asset{1.0f, 1.0f};
    const float asset_call = price_one(model, asset, black_scholes::launch_black_scholes_asset_or_nothing_option_cuda<OptionSide::call>);
    const float asset_put = price_one(model, asset, black_scholes::launch_black_scholes_asset_or_nothing_option_cuda<OptionSide::put>);
    require(close(asset_call + asset_put, discounted_spot), "BS asset partition failed");

    const product::ForwardStartOptionParameters forward{1.0f, 0.5f, 1.0f};
    const float forward_call = price_one(model, forward, black_scholes::launch_black_scholes_forward_start_option_cuda<OptionSide::call>);
    const float forward_put = price_one(model, forward, black_scholes::launch_black_scholes_forward_start_option_cuda<OptionSide::put>);
    require(close(
        forward_call - forward_put,
        discounted_spot - std::exp(-model.dividend_yield * 0.5f - model.risk_free_rate * 0.5f)
    ), "BS forward-start parity failed");

    const product::GeometricAsianOptionParameters geometric{1.0f, 1.0f};
    const float geometric_call = geometric_price_one(model, geometric, OptionSide::call);
    const float geometric_put = geometric_price_one(model, geometric, OptionSide::put);
    require(std::isfinite(geometric_call) && std::isfinite(geometric_put), "BS geometric prices are not finite");

    const product::RangeAccrualParameters range{1.0f, 0.25f, 0.8f, 1.2f, 0.05f};
    const float range_price = price_one(model, range, black_scholes::launch_black_scholes_range_accrual_cuda);
    require(range_price >= discount && range_price <= discount * 1.05f, "BS range-accrual bounds failed");
}
