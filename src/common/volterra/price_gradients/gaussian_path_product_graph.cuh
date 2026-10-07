// Product-generic path sensitivity graph for Gaussian Volterra models.
#pragma once

#include "common/volterra/price_gradients/gaussian_terminal_product_graph.cuh"

namespace ai_factory::workbench::volterra::price_gradients {

template<
    typename ModelPreparation,
    typename ModelPathPolicy,
    typename ProductPreparation,
    typename ProductPolicy,
    typename HybridSchedule,
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities>
struct GaussianPathProductNodeGraph {
    using KernelPolicy = typename ModelPreparation::KernelPolicy;
    using HostPlan = TerminalProductPriceGradientPlan<
        ModelPreparation, ProductPreparation
    >;
    using Cache = GaussianFftKernelCache<KernelPolicy>;
    using Evaluation = GaussianPathEvaluation<
        KernelPolicy, ModelPathPolicy, ProductPolicy, HybridSchedule
    >;
    using ExecutionPlan = GaussianFftGraphExecutionPlan<
        typename Evaluation::NodePolicy
    >;

    static ExecutionPlan plan(
        const HostPlan& host,
        Cache& cache,
        const pg::LaunchConfiguration& launch,
        std::size_t workspace_limit = 1U << 29U
    ) {
        validate_rough_graph_request<Orders>(host);
        return plan_gaussian_fft_node_graph<
            KernelPolicy, ModelPathPolicy, ProductPolicy,
            HostPlan, Orders, MaximumSensitivities, Evaluation
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
        launch_gaussian_fft_node_graph<
            KernelPolicy, ModelPathPolicy, ProductPolicy,
            HostPlan, Orders, MaximumSensitivities, Evaluation
        >(host, cache, configuration, outputs, stencils,
          execution, workspace, workspace_bytes);
    }
};

}  // namespace ai_factory::workbench::volterra::price_gradients
