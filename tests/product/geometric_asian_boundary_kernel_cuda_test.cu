// Check geometric Asian absorption and invalid observations through the real policy.
#include "common/check_cuda.cuh"
#include "model/equity/markovian/cev/dynamics_impl.cuh"
#include "model/equity/markovian/sabr/dynamics_impl.cuh"
#include "product/geometric_asian_option/pricing_policy.cuh"

#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
namespace wb = ai_factory::workbench;
namespace cev = wb::model::equity::cev;
namespace sabr = wb::model::equity::sabr;
using Put = wb::product::GeometricAsianOptionPathPolicy<wb::OptionSide::put>;
using Call = wb::product::GeometricAsianOptionPathPolicy<wb::OptionSide::call>;

template<typename Product, typename Dynamics>
__device__ float observe_path(int zero_from, bool invalid) {
    const typename Product::PreparedProduct terms{1.0f, 0.97f};
    auto handler = Product::make_handler(terms);
    for (int date = 0; date < 5; ++date) {
        float log_spot = date >= zero_from ? -CUDART_INF_F : logf(0.8f);
        if (invalid && date == 4) log_spot = CUDART_NAN_F;
        if (date == 0) handler.on_initial_value(log_spot);
        else handler.on_observation(date - 1, log_spot);
    }
    return Product::template finalize<Dynamics>(terms, {}, handler);
}

__global__ void boundary_probe(float* output) {
    // Initial, intermediate and final absorption, repeated zeros, and finite control.
    for (int zero_from = 0; zero_from <= 5; ++zero_from) {
        output[zero_from] = observe_path<Put, cev::DynamicsPolicy>(zero_from, false);
        output[6 + zero_from] = observe_path<Call, sabr::DynamicsPolicy>(zero_from, false);
    }
    output[12] = observe_path<Put, cev::DynamicsPolicy>(1, true);
    output[13] = observe_path<Call, sabr::DynamicsPolicy>(5, true);

    const Put::PreparedProduct terms{1.0f, 0.97f};
    const auto dynamics = cev::prepare_model({0.0001f, 0.0f, 0.0f, 1.0f, 0.5f},
                                             1.0f / 504.0f);
    auto state = cev::initial_state(dynamics);
    auto handler = Put::make_handler(terms);
    handler.on_initial_value(cev::DynamicsPolicy::log_spot(state));
    for (unsigned int step = 0; step < 2; ++step) {
        cev::one_step_transition(dynamics, 0.0f, state);
        handler.on_observation(step, cev::DynamicsPolicy::log_spot(state));
    }
    output[14] = state.spot;
    output[15] = Put::finalize<cev::DynamicsPolicy>(terms, state, handler);
    handler.on_observation(2U, CUDART_INF_F);
    output[16] = Put::finalize<cev::DynamicsPolicy>(terms, state, handler);

    const auto sabr_dynamics = sabr::DynamicsPolicy::prepare_dynamics(
        {1.0f, 0.0f, 0.0f, 0.2f, 0.3f, -0.5f, 0.5f}, 1.0f / 504.0f);
    auto sabr_state = sabr::DynamicsPolicy::initial_state(sabr_dynamics);
    sabr_state.log_spot = -CUDART_INF_F;
    sabr::DynamicsPolicy::RandomContext random(wb::philox::make_key(91827ULL), 0U);
    auto sabr_handler = Put::make_handler(terms);
    sabr_handler.on_initial_value(sabr_state.log_spot);
    for (unsigned int step = 0; step < 2; ++step) {
        sabr::DynamicsPolicy::simulate_one_step(sabr_dynamics, random, sabr_state);
        sabr_handler.on_observation(step, sabr_state.log_spot);
    }
    output[17] = Put::finalize<sabr::DynamicsPolicy>(terms, sabr_state, sabr_handler);
}
}  // namespace

int main() {
    using wb::check_cuda;
    int device_count = 0;
    const auto availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice || availability == cudaErrorInsufficientDriver
        || device_count == 0) return 77;
    check_cuda(availability, "Asian boundary device discovery");
    std::array<float, 18> actual{};
    float* device = nullptr;
    check_cuda(cudaMalloc(&device, sizeof(actual)), "Asian boundary allocation");
    boundary_probe<<<1, 1>>>(device);
    check_cuda(cudaGetLastError(), "Asian boundary launch");
    check_cuda(cudaMemcpy(actual.data(), device, sizeof(actual), cudaMemcpyDeviceToHost),
               "Asian boundary copy");
    check_cuda(cudaFree(device), "Asian boundary release");
    for (int index = 0; index < 12; ++index) {
        const float expected = index < 5 ? 0.97f : index == 5 ? 0.194f : 0.0f;
        if (!(std::abs(actual[index] - expected) < 1.0e-6f)) {
            throw std::runtime_error("Geometric Asian absorption or finite control failed");
        }
    }
    if (!std::isnan(actual[12]) || !std::isnan(actual[13]) || !std::isnan(actual[16])
        || actual[14] != 0.0f || !(std::abs(actual[15] - 0.97f) < 1.0e-6f)
        || !(std::abs(actual[17] - 0.97f) < 1.0e-6f)) {
        throw std::runtime_error("Geometric Asian masked invalid data or CEV absorption");
    }
    std::cout << "Absorbed CEV put=" << actual[15] << "; finite put=" << actual[5]
              << "; NaN and +infinity rejected\n";
}
