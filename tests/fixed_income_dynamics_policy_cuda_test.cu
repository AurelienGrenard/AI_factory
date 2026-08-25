// Validate fixed-income dynamics policies against their legacy entry points.
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
namespace cir = workbench::model::cir;
namespace g2 = workbench::model::g2;
namespace ou = workbench::model::ornstein_uhlenbeck;
namespace vasicek = workbench::model::vasicek;

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
    __device__ __forceinline__ static bool finite(const g2::State& state) {
        return isfinite(state.state_x) && isfinite(state.state_y);
    }
};

struct JointG2Inspector {
    __device__ __forceinline__ static bool finite(
        const g2::joint::State& state
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
    std::size_t path,
    bool legacy_parity
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
        | (exact << 4U)
        | (representation_parity ? (1U << 8U) : 0U)
        | (legacy_parity ? (1U << 9U) : 0U);
}

__device__ __forceinline__ bool ou_legacy_parity(
    const ou::ModelParameters& parameters,
    float delta_t,
    workbench::philox::PhiloxKey key,
    std::size_t path
) {
    const ou::PreparedModel legacy_model = ou::prepare_model(
        parameters.process
    );
    const ou::PreparedTransition legacy_transition = ou::prepare_transition(
        legacy_model, delta_t
    );
    const float legacy = ou::simulate_terminal_state(
        legacy_model, legacy_transition, parameters.initial_state, key, path
    );
    const ou::PreparedModel model = ou::DynamicsPolicy::prepare_model(
        parameters
    );
    const float current =
        workbench::simulation::simulate_exact_transition_terminal<
            ou::DynamicsPolicy
        >(
            model,
            ou::DynamicsPolicy::prepare_transition(model, delta_t),
            key,
            path
        );
    return workbench::test::bitwise_equal(legacy, current);
}

__device__ __forceinline__ bool ou_joint_legacy_parity(
    const ou::ModelParameters& parameters,
    float delta_t,
    workbench::philox::PhiloxKey key,
    std::size_t path
) {
    const ou::PreparedModel legacy_model = ou::prepare_model(
        parameters.process
    );
    const ou::joint::PreparedTransition legacy_transition =
        ou::joint::prepare_transition(legacy_model, delta_t);
    const ou::joint::State legacy = ou::joint::simulate_terminal_state(
        legacy_model, legacy_transition, parameters.initial_state, key, path
    );
    const ou::PreparedModel model = ou::joint::DynamicsPolicy::prepare_model(
        parameters
    );
    const ou::joint::State current =
        workbench::simulation::simulate_exact_transition_terminal<
            ou::joint::DynamicsPolicy
        >(
            model,
            ou::joint::DynamicsPolicy::prepare_transition(model, delta_t),
            key,
            path
        );
    return workbench::test::bitwise_equal(legacy, current);
}

__device__ __forceinline__ bool vasicek_legacy_parity(
    const vasicek::ModelParameters& parameters,
    float delta_t,
    workbench::philox::PhiloxKey key,
    std::size_t path
) {
    const vasicek::PreparedModel legacy_model = vasicek::prepare_model(
        parameters.process
    );
    const vasicek::PreparedTransition legacy_transition =
        vasicek::prepare_transition(legacy_model, delta_t);
    const float legacy = vasicek::simulate_terminal_state(
        legacy_model, legacy_transition, parameters.initial_state, key, path
    );
    const vasicek::PreparedModel model =
        vasicek::DynamicsPolicy::prepare_model(parameters);
    const float current =
        workbench::simulation::simulate_exact_transition_terminal<
            vasicek::DynamicsPolicy
        >(
            model,
            vasicek::DynamicsPolicy::prepare_transition(model, delta_t),
            key,
            path
        );
    return workbench::test::bitwise_equal(legacy, current);
}

__device__ __forceinline__ bool vasicek_joint_legacy_parity(
    const vasicek::ModelParameters& parameters,
    float delta_t,
    workbench::philox::PhiloxKey key,
    std::size_t path
) {
    const vasicek::PreparedModel legacy_model = vasicek::prepare_model(
        parameters.process
    );
    const vasicek::joint::PreparedTransition legacy_transition =
        vasicek::joint::prepare_transition(legacy_model, delta_t);
    const vasicek::joint::State legacy =
        vasicek::joint::simulate_terminal_state(
            legacy_model,
            legacy_transition,
            parameters.initial_state,
            key,
            path
        );
    const vasicek::PreparedModel model =
        vasicek::joint::DynamicsPolicy::prepare_model(parameters);
    const vasicek::joint::State current =
        workbench::simulation::simulate_exact_transition_terminal<
            vasicek::joint::DynamicsPolicy
        >(
            model,
            vasicek::joint::DynamicsPolicy::prepare_transition(
                model, delta_t
            ),
            key,
            path
        );
    return workbench::test::bitwise_equal(legacy, current);
}

__device__ __forceinline__ bool cir_legacy_parity(
    const cir::ModelParameters& parameters,
    float delta_t,
    workbench::philox::PhiloxKey key,
    std::size_t path
) {
    const cir::PreparedModel legacy_model = cir::prepare_model(
        parameters.process
    );
    const cir::PreparedTransition legacy_transition = cir::prepare_transition(
        legacy_model, delta_t
    );
    const float legacy = cir::simulate_terminal_state(
        legacy_model, legacy_transition, parameters.initial_state, key, path
    );
    const cir::PreparedModel model = cir::DynamicsPolicy::prepare_model(
        parameters
    );
    const float current =
        workbench::simulation::simulate_exact_transition_terminal<
            cir::DynamicsPolicy
        >(
            model,
            cir::DynamicsPolicy::prepare_transition(model, delta_t),
            key,
            path
        );
    return workbench::test::bitwise_equal(legacy, current);
}

__device__ __forceinline__ bool g2_legacy_parity(
    const g2::ModelParameters& parameters,
    float delta_t,
    workbench::philox::PhiloxKey key,
    std::size_t path
) {
    const g2::PreparedModel legacy_model = g2::prepare_model(
        parameters.process
    );
    const g2::PreparedTransition legacy_transition = g2::prepare_transition(
        legacy_model, delta_t
    );
    const g2::State legacy = g2::simulate_terminal_state(
        legacy_model, legacy_transition, parameters.initial_state, key, path
    );
    const g2::PreparedModel model = g2::DynamicsPolicy::prepare_model(
        parameters
    );
    const g2::State current =
        workbench::simulation::simulate_exact_transition_terminal<
            g2::DynamicsPolicy
        >(
            model,
            g2::DynamicsPolicy::prepare_transition(model, delta_t),
            key,
            path
        );
    return workbench::test::bitwise_equal(legacy, current);
}

__device__ __forceinline__ bool g2_joint_legacy_parity(
    const g2::ModelParameters& parameters,
    float delta_t,
    workbench::philox::PhiloxKey key,
    std::size_t path
) {
    const g2::PreparedModel legacy_model = g2::prepare_model(
        parameters.process
    );
    const g2::joint::PreparedTransition legacy_transition =
        g2::joint::prepare_transition(legacy_model, delta_t);
    const g2::joint::State legacy = g2::joint::simulate_terminal_state(
        legacy_model, legacy_transition, parameters.initial_state, key, path
    );
    const g2::PreparedModel model = g2::joint::DynamicsPolicy::prepare_model(
        parameters
    );
    const g2::joint::State current =
        workbench::simulation::simulate_exact_transition_terminal<
            g2::joint::DynamicsPolicy
        >(
            model,
            g2::joint::DynamicsPolicy::prepare_transition(model, delta_t),
            key,
            path
        );
    return workbench::test::bitwise_equal(legacy, current);
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
    const vasicek::ModelParameters vasicek_parameters = {
        {0.25f, 0.04f, 0.02f}, 0.03f,
    };
    const cir::ModelParameters cir_parameters = {
        {0.70f, 0.04f, 0.16f}, 0.03f,
    };
    const g2::ModelParameters g2_parameters = {
        {0.10f, 0.01f, 0.30f, 0.015f, -0.40f},
        {0.02f, -0.01f},
    };

    results[0] = contract_result<ou::DynamicsPolicy, ScalarInspector>(
        ou_parameters,
        delta_t,
        key,
        path,
        ou_legacy_parity(ou_parameters, delta_t, key, path)
    );
    results[1] = contract_result<
        ou::joint::DynamicsPolicy,
        JointScalarInspector
    >(
        ou_parameters,
        delta_t,
        key,
        path,
        ou_joint_legacy_parity(ou_parameters, delta_t, key, path)
    );
    results[2] = contract_result<vasicek::DynamicsPolicy, ScalarInspector>(
        vasicek_parameters,
        delta_t,
        key,
        path,
        vasicek_legacy_parity(vasicek_parameters, delta_t, key, path)
    );
    results[3] = contract_result<
        vasicek::joint::DynamicsPolicy,
        JointScalarInspector
    >(
        vasicek_parameters,
        delta_t,
        key,
        path,
        vasicek_joint_legacy_parity(
            vasicek_parameters, delta_t, key, path
        )
    );
    results[4] = contract_result<cir::DynamicsPolicy, ScalarInspector>(
        cir_parameters,
        delta_t,
        key,
        path,
        cir_legacy_parity(cir_parameters, delta_t, key, path)
    );
    results[5] = contract_result<g2::DynamicsPolicy, G2Inspector>(
        g2_parameters,
        delta_t,
        key,
        path,
        g2_legacy_parity(g2_parameters, delta_t, key, path)
    );
    results[6] = contract_result<
        g2::joint::DynamicsPolicy,
        JointG2Inspector
    >(
        g2_parameters,
        delta_t,
        key,
        path,
        g2_joint_legacy_parity(g2_parameters, delta_t, key, path)
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

    constexpr std::size_t kResultCount = 7U;
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

    constexpr std::uint32_t kExpected = (1U << 10U) - 1U;
    for (std::size_t result = 0U; result < kResultCount; ++result) {
        if (results[result] != kExpected) {
            throw std::runtime_error(
                "A fixed-income dynamics policy contract failed."
            );
        }
    }
    return 0;
}
