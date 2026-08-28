// Compile-time contracts for Gaussian-Volterra hybrid path simulation.
#pragma once

#include "common/simulation/concepts.cuh"

#include <concepts>
#include <type_traits>

namespace ai_factory::workbench::volterra {

template<typename Kernel>
concept HybridKernelPolicy =
    std::is_trivially_copyable_v<typename Kernel::Parameters>
    && std::is_trivially_copyable_v<typename Kernel::PreparedKernel>
    && requires(
        const typename Kernel::Parameters& parameters,
        const typename Kernel::PreparedKernel& kernel,
        float time_step,
        float time,
        float far_convolution,
        float rough_normal,
        float singular_independent_normal,
        unsigned int lag
    ) {
        {
            Kernel::prepare(parameters, time_step)
        } -> std::same_as<typename Kernel::PreparedKernel>;
        { Kernel::far_cell_weight(kernel, lag) } -> std::same_as<float>;
        { Kernel::volterra_variance(kernel, time) } -> std::same_as<float>;
        {
            Kernel::reconstruct_volterra_value(
                kernel,
                far_convolution,
                rough_normal,
                singular_independent_normal
            )
        } -> std::same_as<float>;
    };

template<typename Path>
concept HybridPathPolicy =
    simulation::StatePolicy<Path>
    && std::is_trivially_copyable_v<typename Path::Parameters>
    && std::is_trivially_copyable_v<typename Path::PreparedModel>
    && requires(
        const typename Path::Parameters& parameters,
        const typename Path::PreparedModel& model,
        typename Path::State& state,
        float time_step,
        float volterra_value,
        float volterra_variance,
        float rough_normal,
        float independent_spot_normal
    ) {
        { Path::kUsesVolterraVariance } -> std::convertible_to<bool>;
        { Path::kernel_parameters(parameters) };
        {
            Path::prepare_model(parameters, time_step)
        } -> std::same_as<typename Path::PreparedModel>;
        { Path::initial_state(model) } -> std::same_as<typename Path::State>;
        {
            Path::advance(
                model,
                volterra_value,
                volterra_variance,
                rough_normal,
                independent_spot_normal,
                state
            )
        } -> std::same_as<void>;
    };

template<typename Path, typename Kernel>
concept HybridPathPolicyFor =
    HybridPathPolicy<Path>
    && HybridKernelPolicy<Kernel>
    && requires(const typename Path::Parameters& parameters) {
        {
            Path::kernel_parameters(parameters)
        } -> std::same_as<typename Kernel::Parameters>;
    };

}  // namespace ai_factory::workbench::volterra
