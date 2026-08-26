// Validate the market-neutral dynamics contracts on fixed-income models.
#include "common/check_cuda.cuh"
#include "tests/common/dynamics_contract.cuh"

#include "model/fixed_income/cir/dynamics.cu"
#include "model/fixed_income/g2/dynamics.cu"
#include "model/fixed_income/ornstein_uhlenbeck/dynamics.cu"
#include "model/fixed_income/vasicek/dynamics.cu"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace {

namespace workbench = ai_factory::workbench;
namespace cir = workbench::model::fixed_income::cir;
namespace g2 = workbench::model::fixed_income::g2;
namespace ou = workbench::model::fixed_income::ornstein_uhlenbeck;
namespace vasicek = workbench::model::fixed_income::vasicek;

struct ScalarInspector {
    __device__ __forceinline__ static bool finite(float state) {
        return isfinite(state);
    }
};

struct JointScalarInspector {
    template<typename State>
    __device__ __forceinline__ static bool finite(const State& state) {
        return isfinite(state.state) && isfinite(state.state_integral);
    }
};

struct G2Inspector {
    __device__ __forceinline__ static bool finite(const ai_factory::workbench::model::fixed_income::g2::State& state) {
        return isfinite(state.state_x) && isfinite(state.state_y);
    }
};

struct JointG2Inspector {
    __device__ __forceinline__ static bool finite(
        const ai_factory::workbench::model::fixed_income::g2::joint::State& state
    ) {
        return G2Inspector::finite(state.state)
            && isfinite(state.state_integral);
    }
};

template<typename Dynamics, typename Inspector>
__device__ __forceinline__ std::uint32_t contract_result(
    const typename Dynamics::Parameters& parameters,
    float delta_t,
    workbench::philox::PhiloxKey key,
    std::size_t path
) {
    const std::uint32_t fixed =
        workbench::test::test_fixed_step_dynamics_contract<
            Dynamics,
            Inspector
        >(parameters, delta_t, 4U, key, path);
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

template<typename Dynamics, typename Inspector>
__device__ __forceinline__ std::uint32_t fixed_step_contract_result(
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

__global__ void fixed_income_dynamics_policy_contract_kernel(
    std::uint32_t* results
) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) return;

    constexpr float delta_t = 0.125f;
    constexpr std::size_t path = 17U;
    const workbench::philox::PhiloxKey key =
        workbench::philox::make_key(0x123456789abcdef0ULL);
    const ou::ModelParameters ou_parameters = {{0.25f, 0.02f}, 0.03f};
    const ai_factory::workbench::model::fixed_income::vasicek::ModelParameters vasicek_parameters = {
        {0.25f, 0.04f, 0.02f}, 0.03f,
    };
    const ai_factory::workbench::model::fixed_income::cir::ModelParameters cir_parameters = {
        {0.70f, 0.04f, 0.16f}, 0.03f,
    };
    const ai_factory::workbench::model::fixed_income::g2::ModelParameters g2_parameters = {
        {0.10f, 0.01f, 0.30f, 0.015f, -0.40f},
        {0.02f, -0.01f},
    };

    results[0] = contract_result<ou::DynamicsPolicy, ScalarInspector>(
        ou_parameters,
        delta_t,
        key,
        path
    );
    results[1] = contract_result<
        ou::joint::DynamicsPolicy,
        JointScalarInspector
    >(
        ou_parameters,
        delta_t,
        key,
        path
    );
    results[2] = contract_result<ai_factory::workbench::model::fixed_income::vasicek::DynamicsPolicy, ScalarInspector>(
        vasicek_parameters,
        delta_t,
        key,
        path
    );
    results[3] = contract_result<
        ai_factory::workbench::model::fixed_income::vasicek::joint::DynamicsPolicy,
        JointScalarInspector
    >(
        vasicek_parameters,
        delta_t,
        key,
        path
    );
    results[4] = contract_result<ai_factory::workbench::model::fixed_income::cir::DynamicsPolicy, ScalarInspector>(
        cir_parameters,
        delta_t,
        key,
        path
    );
    results[5] = fixed_step_contract_result<
        ai_factory::workbench::model::fixed_income::cir::joint::DynamicsPolicy,
        JointScalarInspector
    >(
        cir_parameters,
        delta_t,
        key,
        path
    );
    results[6] = contract_result<ai_factory::workbench::model::fixed_income::g2::DynamicsPolicy, G2Inspector>(
        g2_parameters,
        delta_t,
        key,
        path
    );
    results[7] = contract_result<
        ai_factory::workbench::model::fixed_income::g2::joint::DynamicsPolicy,
        JointG2Inspector
    >(
        g2_parameters,
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
        "Fixed-income dynamics policy test cudaGetDeviceCount"
    );

    constexpr std::size_t kResultCount = 8U;
    std::uint32_t* device_results = nullptr;
    workbench::check_cuda(
        cudaMalloc(&device_results, kResultCount * sizeof(std::uint32_t)),
        "Fixed-income dynamics policy test cudaMalloc"
    );
    fixed_income_dynamics_policy_contract_kernel<<<1U, 1U>>>(device_results);
    workbench::check_cuda(
        cudaGetLastError(),
        "Fixed-income dynamics policy test kernel launch"
    );
    workbench::check_cuda(
        cudaDeviceSynchronize(),
        "Fixed-income dynamics policy test synchronization"
    );

    std::uint32_t results[kResultCount]{};
    workbench::check_cuda(
        cudaMemcpy(
            results,
            device_results,
            sizeof(results),
            cudaMemcpyDeviceToHost
        ),
        "Fixed-income dynamics policy test cudaMemcpy"
    );
    workbench::check_cuda(
        cudaFree(device_results),
        "Fixed-income dynamics policy test cudaFree"
    );

    constexpr std::uint32_t kExpected = (1U << 11U) - 1U;
    for (std::size_t result = 0U; result < kResultCount; ++result) {
        const std::uint32_t expected = result == 5U ? 63U : kExpected;
        if (results[result] != expected) {
            throw std::runtime_error(
                "A fixed-income dynamics policy contract failed."
            );
        }
    }
    return 0;
}
