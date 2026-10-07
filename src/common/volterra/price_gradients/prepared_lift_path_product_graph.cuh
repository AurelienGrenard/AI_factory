// Path-product binding for the shared rough lift sensitivity graph.
#pragma once

#include "common/volterra/price_gradients/terminal_product_graph.cuh"
#include "common/volterra/price_gradients/path_node_evaluation.cuh"

namespace ai_factory::workbench::volterra::price_gradients {

template<
    typename Dynamics,
    typename HostPlan,
    typename CachePolicy,
    typename ProductPolicy,
    typename HybridSchedule,
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities>
struct PreparedLiftPathProductGraph {
    using Cache = CachePolicy;
    using Evaluation = PreparedLiftPathEvaluation<
        Dynamics, ProductPolicy, HybridSchedule
    >;
    using ExecutionPlan = PreparedLiftGraphExecutionPlan<
        typename Evaluation::NodePolicy
    >;

    static ExecutionPlan plan(
        const HostPlan& host,
        Cache& cache,
        const pg::LaunchConfiguration& launch,
        std::size_t workspace_limit = 1U << 29U
    ) {
        validate_rough_graph_request<Orders>(host);
        return plan_prepared_lift_node_graph_for_host<
            Dynamics, ProductPolicy, Orders, MaximumSensitivities,
            Evaluation
        >(host, cache, launch, workspace_limit);
    }

    static void launch(
        const HostPlan& host,
        Cache& cache,
        const pg::LaunchConfiguration& configuration,
        pg::SensitivityOutputs outputs,
        mcpg::DevicePreparedStencilOutputs<4U> stencils,
        const ExecutionPlan& execution,
        void* workspace,
        std::size_t workspace_bytes
    ) {
        validate_rough_graph_request<Orders>(host);
        launch_prepared_lift_node_graph<
            Dynamics, ProductPolicy, HostPlan, Cache,
            Orders, MaximumSensitivities, Evaluation
        >(host, cache, configuration, outputs, stencils,
          execution, workspace, workspace_bytes);
    }
};

}  // namespace ai_factory::workbench::volterra::price_gradients
