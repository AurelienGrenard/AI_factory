// Validate the market-neutral dynamics contracts on every non-rough equity model.
#include "common/check_cuda.cuh"
#include "tests/common/dynamics_contract.cuh"

// This translation unit intentionally includes every factorized equity
// implementation. It therefore also detects missing include guards and
// duplicate private implementation symbols.
#include "model/equity/bates/dynamics.cu"
#include "model/equity/black_scholes/dynamics.cu"
#include "model/equity/cev/dynamics.cu"
#include "model/equity/heston/dynamics.cu"
#include "model/equity/kou/dynamics.cu"
#include "model/equity/merton/dynamics.cu"
#include "model/equity/normal_inverse_gaussian/dynamics.cu"
#include "model/equity/schobel_zhu/dynamics.cu"
#include "model/equity/variance_gamma/dynamics.cu"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace {

namespace workbench = ai_factory::workbench;
namespace equity_model = workbench::model::equity;

struct LogSpotInspector {
    template<typename State>
    __device__ __forceinline__ static bool finite(const State& state) {
        return isfinite(state.log_spot);
    }
};

struct SpotInspector {
    template<typename State>
    __device__ __forceinline__ static bool finite(const State& state) {
        return isfinite(state.spot);
    }
};

struct SpotVarianceInspector {
    template<typename State>
    __device__ __forceinline__ static bool finite(const State& state) {
        return isfinite(state.log_spot) && isfinite(state.variance);
    }
};

struct SpotVolatilityInspector {
    template<typename State>
    __device__ __forceinline__ static bool finite(const State& state) {
        return isfinite(state.log_spot) && isfinite(state.volatility);
    }
};

template<typename Dynamics, typename Inspector>
__device__ __forceinline__ std::uint32_t fixed_contract_result(
    const typename Dynamics::Parameters& parameters,
    float delta_t,
    workbench::philox::PhiloxKey key,
    std::size_t path
) {
    return workbench::test::test_fixed_step_dynamics_contract<
        Dynamics,
        Inspector
    >(parameters, delta_t, 4U, key, path);
}

template<typename Dynamics, typename Inspector>
requires workbench::simulation::ExactTransitionDynamicsPolicy<Dynamics>
__device__ __forceinline__ std::uint32_t exact_contract_result(
    const typename Dynamics::Parameters& parameters,
    float delta_t,
    workbench::philox::PhiloxKey key,
    std::size_t path
) {
    const std::uint32_t fixed = fixed_contract_result<Dynamics, Inspector>(
        parameters, delta_t, key, path
    );
    const std::uint32_t exact =
        workbench::test::test_exact_transition_dynamics_contract<
            Dynamics,
            Inspector
        >(parameters, delta_t, key, path);
    const bool representation_parity =
        workbench::test::test_exact_fixed_step_parity<
            Dynamics,
            Inspector
        >(parameters, delta_t, key, path);
    return fixed
        | (exact << 6U)
        | (representation_parity ? (1U << 10U) : 0U);
}

__global__ void equity_dynamics_policy_contract_kernel(
    std::uint32_t* results
) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) return;

    constexpr float delta_t = 0.125f;
    constexpr std::size_t path = 17U;
    const workbench::philox::PhiloxKey key =
        workbench::philox::make_key(0x123456789abcdef0ULL);

    results[0] = exact_contract_result<
        ai_factory::workbench::model::equity::black_scholes::DynamicsPolicy,
        LogSpotInspector
    >({1.0f, 0.03f, 0.01f, 0.20f}, delta_t, key, path);
    results[1] = fixed_contract_result<
        ai_factory::workbench::model::equity::cev::DynamicsPolicy,
        SpotInspector
    >({1.0f, 0.03f, 0.01f, 0.25f, 0.75f}, delta_t, key, path);
    results[2] = fixed_contract_result<
        ai_factory::workbench::model::equity::heston::DynamicsPolicy,
        SpotVarianceInspector
    >(
        {1.0f, 0.03f, 0.01f, 0.04f, 1.4f, 0.04f, 0.32f, -0.65f},
        delta_t,
        key,
        path
    );
    results[3] = fixed_contract_result<
        ai_factory::workbench::model::equity::bates::DynamicsPolicy,
        SpotVarianceInspector
    >(
        {
            1.0f, 0.03f, 0.01f, 0.04f, 1.4f, 0.04f, 0.32f, -0.65f,
            0.8f, -0.12f, 0.24f,
        },
        delta_t,
        key,
        path
    );
    results[4] = exact_contract_result<
        ai_factory::workbench::model::equity::merton::DynamicsPolicy,
        LogSpotInspector
    >(
        {1.0f, 0.03f, 0.01f, 0.20f, 0.7f, -0.12f, 0.25f},
        delta_t,
        key,
        path
    );
    results[5] = exact_contract_result<
        ai_factory::workbench::model::equity::kou::DynamicsPolicy,
        LogSpotInspector
    >(
        {1.0f, 0.03f, 0.01f, 0.20f, 0.7f, 0.35f, 8.0f, 10.0f},
        delta_t,
        key,
        path
    );
    results[6] = exact_contract_result<
        ai_factory::workbench::model::equity::variance_gamma::DynamicsPolicy,
        LogSpotInspector
    >({1.0f, 0.03f, 0.01f, 0.20f, 0.20f, -0.10f}, delta_t, key, path);
    results[7] = exact_contract_result<
        ai_factory::workbench::model::equity::normal_inverse_gaussian::DynamicsPolicy,
        LogSpotInspector
    >({1.0f, 0.03f, 0.01f, 5.0f, -1.0f, 0.20f}, delta_t, key, path);
    results[8] = fixed_contract_result<
        ai_factory::workbench::model::equity::schobel_zhu::DynamicsPolicy,
        SpotVolatilityInspector
    >(
        {1.0f, 0.03f, 0.01f, 0.22f, 1.4f, 0.20f, 0.35f, -0.60f},
        delta_t,
        key,
        path
    );
}

}  // namespace

int main() {
    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) {
        return 77;
    }
    workbench::check_cuda(
        availability,
        "Equity dynamics policy test cudaGetDeviceCount"
    );

    constexpr std::size_t kResultCount = 9U;
    std::uint32_t* device_results = nullptr;
    workbench::check_cuda(
        cudaMalloc(&device_results, kResultCount * sizeof(std::uint32_t)),
        "Equity dynamics policy test cudaMalloc"
    );
    equity_dynamics_policy_contract_kernel<<<1U, 1U>>>(device_results);
    workbench::check_cuda(
        cudaGetLastError(),
        "Equity dynamics policy test kernel launch"
    );
    workbench::check_cuda(
        cudaDeviceSynchronize(),
        "Equity dynamics policy test synchronization"
    );

    std::uint32_t results[kResultCount]{};
    workbench::check_cuda(
        cudaMemcpy(
            results,
            device_results,
            sizeof(results),
            cudaMemcpyDeviceToHost
        ),
        "Equity dynamics policy test cudaMemcpy"
    );
    workbench::check_cuda(
        cudaFree(device_results),
        "Equity dynamics policy test cudaFree"
    );

    constexpr std::uint32_t kFixedExpected = (1U << 6U) - 1U;
    constexpr std::uint32_t kExactExpected = (1U << 11U) - 1U;
    for (std::size_t result = 0U; result < kResultCount; ++result) {
        const bool exact = result == 0U
            || (result >= 4U && result <= 7U);
        const std::uint32_t expected = exact
            ? kExactExpected
            : kFixedExpected;
        if (results[result] != expected) {
            throw std::runtime_error(
                "An equity dynamics policy contract failed."
            );
        }
    }
    return 0;
}
