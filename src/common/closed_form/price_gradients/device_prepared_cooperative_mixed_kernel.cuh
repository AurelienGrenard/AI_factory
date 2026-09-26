// One cooperative block evaluates one selected closed-form Hessian graph.
#pragma once

#include "common/closed_form/concepts.cuh"
#include "common/closed_form/price_gradients/device_prepared_mixed_kernel.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::closed_form::price_gradients {

template<
    typename Policy,
    typename Preparation,
    typename Inputs,
    std::size_t MaximumSensitivities,
    std::size_t MaximumMixedSensitivities>
__global__ void device_prepared_cooperative_mixed_kernel(
    Inputs inputs,
    mcpg::DevicePreparedPlan plan,
    pg::DeviceSensitivityGraph graph,
    pg::LaunchConfiguration launch,
    pg::SensitivityOutputs outputs,
    pg::MixedSensitivityOutputs mixed_outputs,
    mcpg::DevicePreparedStencilOutputs<4U> stencil_outputs,
    pg::MixedSensitivityStencilOutputs mixed_stencil_outputs,
    std::uint32_t workspace_capacity
) {
    using Scenario = typename Preparation::Scenario;
    static_assert(
        sizeof(typename Policy::PreparedRow)
            <= closed_form::kMaximumSharedPreparedRowBytes,
        "Cooperative mixed PreparedRow exceeds the shared-row budget."
    );

    __shared__ Scenario central;
    __shared__ Scenario axis_nodes[4U];
    __shared__ Scenario corner;
    __shared__ typename Policy::PreparedRow prepared;
    __shared__ mixed_detail::AxisValues<MaximumSensitivities> axes;
    __shared__ pg::MixedSensitivityStencil mixed_stencil;
    __shared__ pg::MixedSensitivityValues mixed_values;
    __shared__ float central_value;
    __shared__ bool valid;
    __shared__ bool evaluate_node;
    // Keep one immutable count per coordinate. Reusing one shared scalar lets
    // thread 0 overwrite it for the next coordinate while another warp still
    // evaluates the previous loop condition.
    __shared__ std::uint8_t node_counts[MaximumSensitivities];
    extern __shared__ __align__(16) unsigned char workspace_storage[];

    for (std::size_t launch_index = blockIdx.x;
         launch_index < launch.result_count;
         launch_index += gridDim.x) {
        const std::size_t row = launch.result_offset + launch_index;
        if (threadIdx.x == 0U) {
            valid = inputs.make_central(row, plan, central);
            if (valid) {
                prepared = Policy::prepare(central);
            } else {
                preparation::record_error(
                    stencil_outputs.error,
                    preparation::invalid_central,
                    row,
                    0U
                );
            }
        }
        __syncthreads();
        if (!valid) continue;

        const float evaluated_central = Policy::evaluate(
            prepared,
            reinterpret_cast<std::byte*>(workspace_storage),
            workspace_capacity
        );
        if (threadIdx.x == 0U) {
            central_value = evaluated_central;
            outputs.prices[row] = evaluated_central;
        }
        __syncthreads();

        for (std::size_t sensitivity = 0U;
             sensitivity < plan.sensitivity_count;
             ++sensitivity) {
            if (threadIdx.x == 0U) {
                const auto use = graph.coordinate_uses[sensitivity];
                if (use == pg::SensitivityCoordinateUse::none) {
                    node_counts[sensitivity] = 0U;
                    valid = true;
                } else {
                    int error = preparation::valid;
                    valid = pg::prepare_axis_sensitivity_nodes<Preparation>(
                        central,
                        inputs.sensitivities[sensitivity],
                        use,
                        plan.time,
                        axes.stencils[sensitivity],
                        axis_nodes,
                        error
                    );
                    if (valid) {
                        axes.values[sensitivity][0U] = central_value;
                        node_counts[sensitivity] =
                            static_cast<std::uint8_t>(
                                pg::active_node_count(
                                    axes.stencils[sensitivity]
                                )
                            );
                        stencil_outputs.stencils[
                            row * plan.sensitivity_count + sensitivity
                        ] = axes.stencils[sensitivity];
                    } else {
                        preparation::record_error(
                            stencil_outputs.error,
                            error,
                            row,
                            sensitivity
                        );
                    }
                }
            }
            __syncthreads();
            if (!valid) break;

            for (std::size_t node = 1U;
                 node < node_counts[sensitivity];
                 ++node) {
                if (threadIdx.x == 0U) {
                    prepared = Policy::prepare(axis_nodes[node]);
                }
                __syncthreads();
                const float value = Policy::evaluate(
                    prepared,
                    reinterpret_cast<std::byte*>(workspace_storage),
                    workspace_capacity
                );
                if (threadIdx.x == 0U) {
                    axes.values[sensitivity][node] = value;
                }
                __syncthreads();
            }
        }
        if (!valid) continue;

        if (threadIdx.x == 0U) {
            mixed_detail::reconstruct_axes(axes, graph, row, outputs);
        }
        __syncthreads();

        for (std::size_t selected = 0U;
             selected < graph.mixed_second_count;
             ++selected) {
            const auto pair = graph.mixed_second[selected];
            if (threadIdx.x == 0U) {
                mixed_stencil = pg::make_mixed_sensitivity_stencil(
                    axes.stencils[pair.first],
                    axes.stencils[pair.second]
                );
                valid = true;
            }
            __syncthreads();

            for (std::size_t node = 0U;
                 node < mixed_stencil.node_count;
                 ++node) {
                if (threadIdx.x == 0U) {
                    const auto first_local =
                        mixed_stencil.first_local_nodes[node];
                    const auto second_local =
                        mixed_stencil.second_local_nodes[node];
                    evaluate_node = false;
                    if (first_local == 0U && second_local == 0U) {
                        mixed_values[node] = central_value;
                    } else if (first_local == 0U) {
                        mixed_values[node] =
                            axes.values[pair.second][second_local];
                    } else if (second_local == 0U) {
                        mixed_values[node] =
                            axes.values[pair.first][first_local];
                    } else {
                        valid = pg::prepare_mixed_corner_from_stencils<
                            Preparation
                        >(
                            central,
                            inputs.sensitivities[pair.first],
                            axes.stencils[pair.first],
                            first_local,
                            inputs.sensitivities[pair.second],
                            axes.stencils[pair.second],
                            second_local,
                            plan.time,
                            corner
                        );
                        if (valid) {
                            prepared = Policy::prepare(corner);
                            evaluate_node = true;
                        } else {
                            preparation::record_error(
                                stencil_outputs.error,
                                preparation::no_admissible_stencil,
                                row,
                                pair.first
                            );
                        }
                    }
                }
                __syncthreads();
                if (!valid) break;
                if (evaluate_node) {
                    const float value = Policy::evaluate(
                        prepared,
                        reinterpret_cast<std::byte*>(workspace_storage),
                        workspace_capacity
                    );
                    if (threadIdx.x == 0U) mixed_values[node] = value;
                }
                __syncthreads();
            }
            if (!valid) break;

            if (threadIdx.x == 0U) {
                const auto output =
                    row * graph.mixed_second_count + selected;
                mixed_stencil_outputs.stencils[output] = mixed_stencil;
                mixed_outputs.hessians[output] =
                    pg::reconstruct_mixed_sensitivity(
                        mixed_stencil, mixed_values, central_value
                    );
            }
            __syncthreads();
        }
    }
}

template<
    typename Policy,
    std::size_t MaximumSensitivities,
    std::size_t MaximumMixedSensitivities,
    typename HostPlan>
bool launch_device_prepared_cooperative_mixed(
    const HostPlan& host,
    typename HostPlan::DeviceInputs device,
    typename HostPlan::DiagonalStencilOutputs stencil_outputs,
    typename HostPlan::MixedStencilOutputs mixed_stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    pg::MixedSensitivityOutputs mixed_outputs,
    void* workspace_storage,
    std::size_t workspace_bytes,
    std::uint32_t pricing_workspace_capacity,
    const char* name,
    const char* variant
) {
    if (pricing_workspace_capacity == 0U) {
        throw std::invalid_argument(
            "Cooperative mixed pricing workspace capacity is zero."
        );
    }
    const auto function = device_prepared_cooperative_mixed_kernel<
        Policy,
        typename HostPlan::Preparation,
        typename HostPlan::DeviceInputs,
        MaximumSensitivities,
        MaximumMixedSensitivities
    >;
    const std::size_t shared =
        Policy::required_shared_memory_bytes(pricing_workspace_capacity);
    cudaFuncAttributes attributes{};
    check_cuda(cudaFuncGetAttributes(&attributes, function), name);
    if (shared
        > static_cast<std::size_t>(attributes.maxDynamicSharedSizeBytes)) {
        return false;
    }
    int active = 0;
    check_cuda(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active,
            function,
            configuration.threads_per_block,
            shared
        ),
        name
    );
    if (active == 0) return false;

    const auto graph = prepare_device_mixed_launch<
        MaximumSensitivities,
        MaximumMixedSensitivities
    >(
        host,
        device,
        stencil_outputs,
        mixed_stencil_outputs,
        configuration,
        outputs,
        mixed_outputs,
        workspace_storage,
        workspace_bytes
    );
    const std::string diagnostic_variant = std::string(variant)
        + "/cooperative/K=" + std::to_string(host.sensitivity_count())
        + "/mixed="
        + std::to_string(host.sensitivity_graph.mixed_second.size());
    report_cuda_kernel_launch_if_enabled(
        name,
        diagnostic_variant.c_str(),
        function,
        dim3(static_cast<unsigned int>(configuration.block_count)),
        dim3(configuration.threads_per_block),
        shared
    );
    function<<<configuration.block_count, configuration.threads_per_block,
               shared>>>(
        device,
        mcpg::make_device_prepared_plan(host),
        graph,
        configuration,
        outputs,
        mixed_outputs,
        stencil_outputs,
        mixed_stencil_outputs,
        pricing_workspace_capacity
    );
    check_cuda(cudaGetLastError(), name);
    return true;
}

}  // namespace ai_factory::workbench::closed_form::price_gradients
