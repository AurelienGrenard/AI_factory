// Generic terminal-product binding for the Gaussian-Volterra FFT graph.
#pragma once

#include "common/volterra/price_gradients/terminal_product_graph.cuh"
#include "common/volterra/price_gradients/gaussian_fft_node_graph.cuh"

namespace ai_factory::workbench::volterra::price_gradients {

template<
    typename ModelPreparation,
    typename ModelPathPolicy,
    typename ProductPreparation,
    typename ProductPolicyT,
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities>
struct GaussianTerminalProductNodeGraph {
    using KernelPolicy = typename ModelPreparation::KernelPolicy;
    using ProductPolicy = ProductPolicyT;
    using HostPlan = TerminalProductPriceGradientPlan<
        ModelPreparation, ProductPreparation
    >;
    using Cache = GaussianFftKernelCache<KernelPolicy>;
    using ExecutionPlan = GaussianFftGraphExecutionPlan<
        PreparedLiftTerminalNodePolicy<ModelPathPolicy, ProductPolicy>
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
            HostPlan, Orders, MaximumSensitivities
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
            HostPlan, Orders, MaximumSensitivities
        >(host, cache, configuration, outputs, stencils,
          execution, workspace, workspace_bytes);
    }
};

}  // namespace ai_factory::workbench::volterra::price_gradients
