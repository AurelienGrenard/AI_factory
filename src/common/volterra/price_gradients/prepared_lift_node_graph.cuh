// Bounded CRN node graph for host-prepared fixed-factor Volterra lifts.
#pragma once

#include "common/check_cuda.cuh"
#include "common/equity/price_gradients/terminal_device_preparation.cuh"
#include "common/monte_carlo/price_gradients/terminal_node_graph/reconstruction.cuh"
#include "common/monte_carlo/price_gradients/terminal_node_graph/workspace.cuh"
#include "common/price_gradients/device_prepared_launch.cuh"
#include "common/volterra/price_gradients/prepared_lift_row_graph.hpp"
#include "common/volterra/price_gradients/path_node_evaluation.cuh"
#include "common/volterra/price_gradients/terminal_node_policy.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ai_factory::workbench::volterra::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace mcpg = ::ai_factory::workbench::monte_carlo::price_gradients;

template<typename Dynamics, typename ProductPolicy>
struct PreparedLiftTerminalEvaluation {
    static constexpr bool kReuseCentralValue = true;
    static constexpr bool kBatchSharedPath = false;
    using NodePolicy = PreparedLiftTerminalNodePolicy<Dynamics, ProductPolicy>;

    template<typename Scenario>
    __device__ static float evaluate(
        const typename Dynamics::PreparedDynamics& dynamics,
        const Scenario& scenario,
        const typename NodePolicy::Metadata&,
        pg::TimeConfiguration,
        philox::PhiloxKey key,
        std::size_t path
    ) {
        typename Dynamics::RandomContext random(key, path);
        auto state = Dynamics::initial_state(dynamics);
        Dynamics::advance(dynamics, scenario.step_count, random, state);
        return Dynamics::spot(state);
    }
};

template<typename NodePolicy>
struct PreparedLiftGraphExecutionPlan {
    mcpg::TerminalNodeGraphConfiguration graph{};
    mcpg::TerminalNodeGraphWorkspaceLayout node_layout{};
    std::size_t row_offset = 0U;
    std::size_t prepared_offset = 0U;
    std::size_t prepared_capacity = 0U;
    std::size_t bytes = 0U;
};

template<
    typename Dynamics,
    typename ProductPolicy,
    typename Scenario,
    typename Prepared,
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities,
    typename Evaluation = PreparedLiftTerminalEvaluation<Dynamics, ProductPolicy>>
PreparedLiftGraphExecutionPlan<
    typename Evaluation::NodePolicy
> plan_prepared_lift_node_graph(
    std::size_t result_count,
    std::size_t sensitivity_count,
    std::size_t paths_per_row,
    std::size_t prepared_count,
    unsigned int reduction_threads,
    std::size_t workspace_limit = 1U << 29U
) {
    static_assert(Orders != pg::SensitivityOrders::none);
    using NodePolicy = typename Evaluation::NodePolicy;
    using Row = PreparedLiftGraphRow<Scenario, MaximumSensitivities>;
    if (!result_count || !sensitivity_count
        || sensitivity_count > MaximumSensitivities
        || paths_per_row < 2U || !prepared_count) {
        throw std::invalid_argument("Invalid prepared-lift graph dimensions.");
    }
    auto rows = std::min<std::size_t>(result_count, 16U);
    auto paths = std::min<std::size_t>(paths_per_row, 8192U);
    paths = std::max<std::size_t>(reduction_threads,
        paths / reduction_threads * reduction_threads);
    while (true) {
        mcpg::TerminalNodeGraphConfiguration graph{rows, paths, 1U};
        const auto layout = mcpg::terminal_node_graph_workspace_layout<
            NodePolicy, Orders, MaximumSensitivities
        >(sensitivity_count, reduction_threads, graph);
        auto bytes = layout.bytes;
        const auto row_offset =
            workspace_layout::append_array<Row>(
                bytes, rows, "Prepared-lift graph workspace overflow."
            );
        const auto prepared_offset =
            workspace_layout::append_array<Prepared>(
                bytes, prepared_count,
                "Prepared-lift graph workspace overflow."
            );
        if (bytes <= workspace_limit) {
            return {graph, layout, row_offset, prepared_offset, prepared_count, bytes};
        }
        if (rows > 1U) rows = (rows + 1U) / 2U;
        else if (paths > reduction_threads) {
            paths = std::max<std::size_t>(
                reduction_threads,
                (paths / 2U) / reduction_threads * reduction_threads
            );
        } else {
            throw std::invalid_argument(
                "Prepared-lift graph exceeds its workspace limit."
            );
        }
    }
}

// Public planner: populate the cache from one row per model before sizing
// device storage. Cartesian product rows then only assemble cheap descriptors.
template<
    typename Dynamics,
    typename ProductPolicy,
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities,
    typename Evaluation = PreparedLiftTerminalEvaluation<Dynamics, ProductPolicy>,
    typename HostPlan,
    typename Cache>
auto plan_prepared_lift_node_graph_for_host(
    const HostPlan& host,
    Cache& cache,
    const pg::LaunchConfiguration& launch,
    std::size_t workspace_limit = 1U << 29U
) {
    prepare_lift_model_node_cache<Orders, MaximumSensitivities>(host, cache);
    return plan_prepared_lift_node_graph<
        Dynamics, ProductPolicy,
        typename HostPlan::Preparation::Scenario,
        typename Cache::Prepared,
        Orders, MaximumSensitivities, Evaluation
    >(
        launch.result_count, host.sensitivity_count(),
        launch.paths_per_price, cache.values().size(),
        launch.threads_per_block, workspace_limit
    );
}

template<
    typename Dynamics,
    typename ProductPolicy,
    typename Preparation,
    std::size_t MaximumSensitivities,
    typename Evaluation = PreparedLiftTerminalEvaluation<Dynamics, ProductPolicy>>
__global__ void evaluate_prepared_lift_nodes_kernel(
    const PreparedLiftGraphRow<
        typename Preparation::Scenario, MaximumSensitivities
    >* rows,
    const typename Dynamics::PreparedDynamics* prepared_pool,
    std::size_t first_row,
    std::size_t row_count,
    std::size_t first_path,
    std::size_t path_count,
    std::size_t path_capacity,
    std::size_t sensitivity_count,
    pg::TimeConfiguration time,
    mcpg::TerminalNodeGraphWorkspace<
        typename Evaluation::NodePolicy
    > workspace,
    mcpg::DevicePreparedStencilOutputs<4U> stencil_outputs,
    std::uint64_t base_seed
) {
    using NodePolicy = typename Evaluation::NodePolicy;
    constexpr auto capacity = PreparedLiftGraphRow<
        typename Preparation::Scenario, MaximumSensitivities
    >::kNodeCapacity;
    const auto local_row = static_cast<std::size_t>(blockIdx.x);
    if (local_row >= row_count) return;
    const auto& row = rows[local_row];
    __shared__ typename Dynamics::PreparedDynamics shared_dynamics[capacity];
    __shared__ typename NodePolicy::Metadata shared_metadata[capacity];
    for (std::size_t node = threadIdx.x; node < row.node_count;
         node += blockDim.x) {
        shared_dynamics[node] = prepared_pool[row.prepared_indices[node]];
        shared_metadata[node] = NodePolicy::prepare_metadata(
            row.scenarios[node], time
        );
    }
    __syncthreads();
    const auto global_row = first_row + local_row;
    if (blockIdx.y == 0U && threadIdx.x == 0U) {
        workspace.row_status[local_row] = 1U;
        for (std::size_t node = 0U; node < row.node_count; ++node) {
            workspace.node_metadata[local_row * capacity + node] =
                shared_metadata[node];
        }
        for (std::size_t sensitivity = 0U;
             sensitivity < sensitivity_count; ++sensitivity) {
            workspace.node_indices[
                local_row * sensitivity_count + sensitivity
            ] = row.node_indices[sensitivity];
            stencil_outputs.stencils[
                global_row * sensitivity_count + sensitivity
            ] = row.stencils[sensitivity];
        }
    }
    const auto local_path = static_cast<std::size_t>(blockIdx.y)
        * blockDim.x + threadIdx.x;
    if (local_path >= path_count) return;
    const auto path = first_path + local_path;
    const auto key = philox::make_key(base_seed + global_row);
    auto* values = workspace.node_values
        + (local_row * path_capacity + local_path) * capacity;
    if constexpr (Evaluation::kBatchSharedPath) {
        constexpr std::size_t batch_capacity = 8U;
        std::uint16_t indices[batch_capacity]{};
        float batch_values[batch_capacity]{};
        std::size_t scan = 0U;
        while (scan < row.node_count) {
            std::size_t count = 0U;
            for (; scan < row.node_count && count < batch_capacity;
                 ++scan) {
                if (scan == 0U || row.scenarios[scan].reuse_central)
                    indices[count++] = static_cast<std::uint16_t>(scan);
            }
            if (count == 0U) continue;
            Evaluation::evaluate_batch(
                shared_dynamics[0U], row.scenarios, shared_metadata,
                indices, count, time, key, path, batch_values
            );
            for (std::size_t i = 0U; i < count; ++i)
                values[indices[i]] = batch_values[i];
        }
        for (std::size_t node = 1U; node < row.node_count; ++node) {
            if (row.scenarios[node].reuse_central) continue;
            values[node] = Evaluation::evaluate(
                shared_dynamics[node], row.scenarios[node],
                shared_metadata[node], time, key, path
            );
        }
    } else {
        float central = 0.0f;
        for (std::size_t node = 0U; node < row.node_count; ++node) {
            float value = central;
            if (node == 0U || !row.scenarios[node].reuse_central) {
                value = Evaluation::evaluate(
                    shared_dynamics[node], row.scenarios[node],
                    shared_metadata[node], time, key, path
                );
                if (node == 0U) central = value;
            }
            values[node] = value;
        }
    }
}

template<
    typename Dynamics,
    typename ProductPolicy,
    typename HostPlan,
    typename Cache,
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities,
    typename Evaluation = PreparedLiftTerminalEvaluation<Dynamics, ProductPolicy>>
void launch_prepared_lift_node_graph(
    const HostPlan& host,
    Cache& cache,
    const pg::LaunchConfiguration& launch,
    pg::SensitivityOutputs outputs,
    mcpg::DevicePreparedStencilOutputs<4U> stencil_outputs,
    const PreparedLiftGraphExecutionPlan<
        typename Evaluation::NodePolicy
    >& execution,
    void* storage,
    std::size_t storage_bytes
) {
    static_assert(Orders != pg::SensitivityOrders::none);
    using NodePolicy = typename Evaluation::NodePolicy;
    using Row = PreparedLiftGraphRow<
        typename HostPlan::Preparation::Scenario, MaximumSensitivities
    >;
    if (storage == nullptr || storage_bytes < execution.bytes
        || launch.result_offset + launch.result_count > host.result_count
        || stencil_outputs.stencils == nullptr
        || stencil_outputs.capacity
            < host.result_count * host.sensitivity_count()
        || cache.values().size() != execution.prepared_capacity
        || launch.threads_per_block !=
            execution.node_layout.capacities.thread_moments
                / (execution.graph.row_chunk_size
                    * mcpg::terminal_node_graph_output_count<Orders>(
                        host.sensitivity_count()))
        || outputs.prices == nullptr
        || outputs.price_standard_errors == nullptr
        || (pg::requests_first_v<Orders>
            && (outputs.gradients == nullptr
                || outputs.gradient_standard_errors == nullptr))
        || (pg::requests_second_v<Orders>
            && (outputs.diagonal_hessians == nullptr
                || outputs.diagonal_hessian_standard_errors == nullptr))
        || outputs.price_capacity < host.result_count
        || outputs.sensitivity_capacity
            < host.result_count * host.sensitivity_count()) {
        throw std::invalid_argument("Invalid prepared-lift graph launch.");
    }
    auto* base = static_cast<unsigned char*>(storage);
    auto* device_rows = reinterpret_cast<Row*>(base + execution.row_offset);
    auto* device_pool = reinterpret_cast<typename Cache::Prepared*>(
        base + execution.prepared_offset
    );
    const auto workspace = mcpg::make_terminal_node_graph_workspace<
        NodePolicy
    >(storage, storage_bytes, execution.node_layout);
    const auto uploaded_prepared_count = cache.values().size();
    check_cuda(cudaMemcpy(
        device_pool, cache.values().data(),
        uploaded_prepared_count * sizeof(typename Cache::Prepared),
        cudaMemcpyHostToDevice
    ), "prepared-lift graph dynamics upload");
    const auto output_count =
        mcpg::terminal_node_graph_output_count<Orders>(
            host.sensitivity_count()
        );
    for (std::size_t first = launch.result_offset;
         first < launch.result_offset + launch.result_count;
         first += execution.graph.row_chunk_size) {
        const auto row_count = std::min(
            execution.graph.row_chunk_size,
            launch.result_offset + launch.result_count - first
        );
        std::vector<Row> rows;
        rows.reserve(row_count);
        for (std::size_t row = first; row < first + row_count; ++row) {
            rows.push_back(prepare_lift_graph_row<
                Orders, MaximumSensitivities
            >(host, row, cache));
        }
        if (cache.values().size() != uploaded_prepared_count) {
            throw std::invalid_argument(
                "Prepared-lift cache changed after dynamics upload; "
                "plan it from the complete host request."
            );
        }
        check_cuda(cudaMemcpy(
            device_rows, rows.data(), row_count * sizeof(Row),
            cudaMemcpyHostToDevice
        ), "prepared-lift graph row upload");
        for (std::size_t path = 0U; path < launch.paths_per_price;
             path += execution.graph.path_chunk_size) {
            const auto count = std::min(
                execution.graph.path_chunk_size,
                launch.paths_per_price - path
            );
            const auto path_blocks = static_cast<unsigned int>(
                (count + 255U) / 256U
            );
            evaluate_prepared_lift_nodes_kernel<
                Dynamics, ProductPolicy,
                typename HostPlan::Preparation, MaximumSensitivities,
                Evaluation
            ><<<dim3(static_cast<unsigned int>(row_count), path_blocks), 256>>>(
                device_rows, device_pool, first, row_count, path, count,
                execution.graph.path_chunk_size,
                host.sensitivity_count(), host.time, workspace,
                stencil_outputs, launch.base_seed
            );
            check_cuda(cudaGetLastError(), "prepared-lift node evaluation");
            const auto accumulate = mcpg::node_graph_detail::
                accumulate_node_moments_kernel<
                    Orders, NodePolicy, MaximumSensitivities
                >;
            accumulate<<<dim3(
                static_cast<unsigned int>(row_count),
                static_cast<unsigned int>(output_count)
            ), launch.threads_per_block>>>(
                first, row_count, path, count,
                execution.graph.path_chunk_size,
                host.sensitivity_count(), workspace, stencil_outputs
            );
            check_cuda(cudaGetLastError(), "prepared-lift moment accumulation");
        }
        const auto finalize = mcpg::node_graph_detail::
            finalize_node_moments_kernel<Orders, NodePolicy>;
        finalize<<<dim3(
            static_cast<unsigned int>(row_count),
            static_cast<unsigned int>(output_count)
        ), launch.threads_per_block,
        2U * (launch.threads_per_block / 32U) * sizeof(double)>>>(
            first, row_count, host.sensitivity_count(),
            launch.paths_per_price, workspace, outputs
        );
        check_cuda(cudaGetLastError(), "prepared-lift moment finalization");
    }
}

}  // namespace ai_factory::workbench::volterra::price_gradients
