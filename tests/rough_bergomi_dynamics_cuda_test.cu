// Verify rough-Bergomi hybrid preparation, transition, and Philox replay.
#include "common/check_cuda.cuh"
#include "model/equity/rough_bergomi/dataset.hpp"
#include "model/equity/rough_bergomi/dynamics.cu"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace {

constexpr std::uint32_t kStepCount = 8U;

struct RoughBergomiDynamicsResults {
    ai_factory::workbench::model::equity::rough_bergomi::RoughBergomiPreparedParameters model;
    float first_far_weight;
    float first_log_variance_correction;
    float explicit_log_spot;
    float explicit_variance;
    float explicit_brownian_increment;
    float explicit_second_log_spot;
    float explicit_second_variance;
    float explicit_second_brownian_increment;
    float terminal_first;
    float terminal_replay;
    float two_time_terminal;
    float arithmetic_mean;
    float geometric_mean;
    float maximum_spot;
};

__global__ void exercise_rough_bergomi_dynamics_kernel(
    float* history_storage,
    RoughBergomiDynamicsResults* output
) {
    using namespace ai_factory::workbench;
    using namespace ai_factory::workbench::model::equity::rough_bergomi;

    __shared__ RoughBergomiPreparedParameters model;
    __shared__ float far_weights[kStepCount];
    __shared__ float log_variance_corrections[kStepCount];
    if (threadIdx.x == 0U) {
        const RoughBergomiModelParameters parameters = {
            1.0f, 0.03f, 0.01f, 0.04f, 1.7f, 0.10f, -0.70f,
        };
        model = prepare_model(parameters, 1.0f, kStepCount);
    }
    __syncthreads();
    prepare_hybrid_grid(
        model,
        kStepCount,
        far_weights,
        log_variance_corrections,
        threadIdx.x,
        blockDim.x
    );
    __syncthreads();
    if (threadIdx.x != 0U) return;

    const RoughBergomiHybridGridView grid = {
        far_weights,
        log_variance_corrections,
    };
    const RoughBergomiHistoryView history = {history_storage, 1U};

    RoughBergomiState explicit_state = initial_state(model);
    one_step_transition(
        model,
        grid,
        history,
        0U,
        -0.35f,
        0.45f,
        0.20f,
        explicit_state
    );
    const float explicit_log_spot = explicit_state.log_spot;
    const float explicit_variance = explicit_state.variance;
    const float explicit_brownian_increment = history_storage[0];
    one_step_transition(
        model,
        grid,
        history,
        1U,
        0.25f,
        -0.15f,
        0.40f,
        explicit_state
    );
    const float explicit_second_log_spot = explicit_state.log_spot;
    const float explicit_second_variance = explicit_state.variance;
    const float explicit_second_brownian_increment = history_storage[1];

    const philox::PhiloxKey key = philox::make_key(900001001ULL);
    const RoughBergomiState terminal_first = simulate_terminal_state(
        model, grid, history, key, 17U, kStepCount
    );
    const RoughBergomiState terminal_replay = simulate_terminal_state(
        model, grid, history, key, 17U, kStepCount
    );
    const RoughBergomiTwoTimePathResult two_time = simulate_at_two_times(
        model, grid, history, key, 17U, 3U, kStepCount
    );
    const RoughBergomiMeanPathResult mean = simulate_mean_state(
        model, grid, history, key, 19U, kStepCount
    );
    const RoughBergomiGeometricMeanPathResult geometric =
        simulate_geometric_mean_state(
            model, grid, history, key, 23U, kStepCount
        );
    const RoughBergomiMaximumPathResult maximum = simulate_maximum_state(
        model, grid, history, key, 29U, kStepCount
    );

    *output = {
        model,
        far_weights[0],
        log_variance_corrections[0],
        explicit_log_spot,
        explicit_variance,
        explicit_brownian_increment,
        explicit_second_log_spot,
        explicit_second_variance,
        explicit_second_brownian_increment,
        terminal_first.log_spot,
        terminal_replay.log_spot,
        logf(two_time.terminal_spot),
        mean.arithmetic_mean,
        geometric.geometric_mean,
        maximum.maximum_spot,
    };
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool close(float left, float right, float tolerance = 5.0e-6f) {
    return std::fabs(left - right) <= tolerance;
}

}  // namespace

int main() {
    const auto models =
        ai_factory::workbench::model::equity::rough_bergomi::load_models(
            "datasets/model/equity/rough_bergomi/parameters/rough_bergomi_01.json"
        );
    require(
        models.size() == 1000U,
        "rough-Bergomi model dataset does not contain 1000 rows"
    );
    require(
        std::all_of(
            models.begin(),
            models.end(),
            [](const auto& model) {
                return model.spot == 1.0f && model.xi_0 == 0.04f;
            }
        ),
        "rough-Bergomi dataset does not keep spot and xi_0 fixed"
    );

    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) {
        return 77;
    }
    ai_factory::workbench::check_cuda(
        availability, "rough-Bergomi dynamics test cudaGetDeviceCount"
    );

    float* device_history = nullptr;
    RoughBergomiDynamicsResults* device_results = nullptr;
    ai_factory::workbench::check_cuda(
        cudaMalloc(&device_history, kStepCount * sizeof(float)),
        "rough-Bergomi dynamics test history cudaMalloc"
    );
    ai_factory::workbench::check_cuda(
        cudaMalloc(&device_results, sizeof(RoughBergomiDynamicsResults)),
        "rough-Bergomi dynamics test results cudaMalloc"
    );
    exercise_rough_bergomi_dynamics_kernel<<<1, 32>>>(
        device_history, device_results
    );
    ai_factory::workbench::check_cuda(
        cudaGetLastError(), "rough-Bergomi dynamics test kernel launch"
    );

    RoughBergomiDynamicsResults results{};
    ai_factory::workbench::check_cuda(
        cudaMemcpy(
            &results,
            device_results,
            sizeof(results),
            cudaMemcpyDeviceToHost
        ),
        "rough-Bergomi dynamics test cudaMemcpy"
    );
    ai_factory::workbench::check_cuda(
        cudaFree(device_results),
        "rough-Bergomi dynamics test results cudaFree"
    );
    ai_factory::workbench::check_cuda(
        cudaFree(device_history),
        "rough-Bergomi dynamics test history cudaFree"
    );

    constexpr float dt = 1.0f / static_cast<float>(kStepCount);
    constexpr float h = 0.10f;
    constexpr float alpha = h - 0.5f;
    constexpr float alpha_plus_one = alpha + 1.0f;
    constexpr float eta = 1.7f;
    constexpr float rho = -0.70f;
    const float expected_first_weight = std::pow(dt, alpha)
        * (std::pow(2.0f, alpha_plus_one) - 1.0f)
        / alpha_plus_one;
    const float expected_first_correction = std::log(0.04f)
        - 0.5f * eta * eta * std::pow(dt, 2.0f * h);
    require(
        close(results.first_far_weight, expected_first_weight),
        "rough-Bergomi optimal far-cell weight is incorrect"
    );
    require(
        close(
            results.first_log_variance_correction,
            expected_first_correction
        ),
        "rough-Bergomi log-variance correction is incorrect"
    );

    constexpr float rough_normal = -0.35f;
    constexpr float singular_normal = 0.45f;
    constexpr float spot_normal = 0.20f;
    const float delta_w = std::sqrt(dt) * rough_normal;
    const float delta_w_perp = std::sqrt(dt) * spot_normal;
    const float spot_increment = rho * delta_w
        + std::sqrt(1.0f - rho * rho) * delta_w_perp;
    const float expected_log_spot = (0.03f - 0.01f) * dt
        - 0.5f * 0.04f * dt
        + std::sqrt(0.04f) * spot_increment;
    const float singular_integral =
        results.model.singular_driver_loading * rough_normal
        + results.model.singular_independent_loading * singular_normal;
    const float rough_driver = std::sqrt(2.0f * h) * singular_integral;
    const float expected_variance = std::exp(
        expected_first_correction + eta * rough_driver
    );
    require(
        close(results.explicit_brownian_increment, delta_w),
        "rough-Bergomi history does not store the driving Brownian increment"
    );
    require(
        close(results.explicit_log_spot, expected_log_spot),
        "rough-Bergomi explicit log-spot transition is incorrect"
    );
    require(
        close(results.explicit_variance, expected_variance),
        "rough-Bergomi explicit variance transition is incorrect"
    );

    constexpr float second_rough_normal = 0.25f;
    constexpr float second_singular_normal = -0.15f;
    constexpr float second_spot_normal = 0.40f;
    const float second_delta_w = std::sqrt(dt) * second_rough_normal;
    const float second_delta_w_perp = std::sqrt(dt) * second_spot_normal;
    const float second_spot_increment = rho * second_delta_w
        + std::sqrt(1.0f - rho * rho) * second_delta_w_perp;
    const float expected_second_log_spot = expected_log_spot
        + (0.03f - 0.01f) * dt
        - 0.5f * expected_variance * dt
        + std::sqrt(expected_variance) * second_spot_increment;
    const float second_singular_integral =
        results.model.singular_driver_loading * second_rough_normal
        + results.model.singular_independent_loading
            * second_singular_normal;
    const float second_rough_driver = std::sqrt(2.0f * h)
        * (second_singular_integral + expected_first_weight * delta_w);
    const float second_correction = std::log(0.04f)
        - 0.5f * eta * eta * std::pow(2.0f * dt, 2.0f * h);
    const float expected_second_variance = std::exp(
        second_correction + eta * second_rough_driver
    );
    require(
        close(results.explicit_second_brownian_increment, second_delta_w),
        "rough-Bergomi second Brownian increment is not stored correctly"
    );
    require(
        close(results.explicit_second_log_spot, expected_second_log_spot),
        "rough-Bergomi second log-spot transition is incorrect"
    );
    require(
        close(results.explicit_second_variance, expected_second_variance),
        "rough-Bergomi far-history convolution is incorrect"
    );
    require(
        results.terminal_first == results.terminal_replay
            && results.terminal_first == results.two_time_terminal,
        "rough-Bergomi path replay or continuous two-time history is incorrect"
    );
    require(
        std::isfinite(results.arithmetic_mean)
            && results.arithmetic_mean > 0.0f
            && std::isfinite(results.geometric_mean)
            && results.geometric_mean > 0.0f
            && std::isfinite(results.maximum_spot)
            && results.maximum_spot >= 1.0f,
        "rough-Bergomi path summaries are not finite and positive"
    );
    return 0;
}
