// Offline first-order rough graph datasets; node and checkpoint handling reuse
// the shared price-gradient writer and Philox launch plan.
#pragma once

#include "tools/pricing/price_gradients/generation.cuh"

#include <algorithm>
#include <optional>
#include <type_traits>
#include <utility>

namespace ai_factory::workbench::offline::pricing::price_gradients {

template<typename Graph>
class RoughGraphExecution {
public:
    static constexpr std::size_t kStencilNodeCapacity = 4U;
    static constexpr bool kHasHostStencilReplay = true;
    using Cache = typename Graph::Cache;
    using ExecutionPlan = typename Graph::ExecutionPlan;

    RoughGraphExecution(float horizon, float dt)
        : cache_(make_cache(horizon, dt)) {}

    std::size_t workspace_bytes(
        const auto& host, const pg::LaunchConfiguration& launch
    ) {
        if constexpr (requires(const Cache& cache) { cache.entries(); }) {
            // FFT spectra are prepared before each row launch. The dataset
            // writer launches one row at a time, so planning only its first
            // chunk leaves later model kernels absent from the frozen cache.
            // Plan all rows once; the graph still executes one row at a time.
            auto complete = launch;
            complete.result_offset = 0U;
            complete.result_count = host.result_count;
            plan_ = Graph::plan(host, cache_, complete);
        } else {
            plan_ = Graph::plan(host, cache_, launch);
        }
        return plan_->bytes;
    }

    void launch(
        const auto& host, const auto&,
        mcpg::DevicePreparedStencilOutputs<4U> stencils,
        const pg::LaunchConfiguration& launch,
        pg::Outputs outputs, void* workspace, std::size_t bytes
    ) {
        Graph::launch(host, cache_, launch,
                      pg::as_sensitivity_outputs(outputs), stencils,
                      *plan_, workspace, bytes);
    }

    void prepare_stencils(
        const auto& host, mcpg::DevicePreparedStencilOutputs<4U> stencils,
        const pg::LaunchConfiguration& launch,
        pg::SensitivityOutputs outputs, void* workspace, std::size_t bytes
    ) {
        // Rebuild deterministic stencils for checkpointed rows with two
        // Philox paths; the checkpoint restores their actual price moments.
        Graph::launch(host, cache_, launch, outputs, stencils,
                      *plan_, workspace, bytes);
    }

    pg::SensitivityStrategy strategy() const {
        return pg::SensitivityStrategy::node_graph;
    }
    const char* strategy_name() const { return "node_graph"; }
    const char* preparation_name() const {
        return "host_prepared_rough_sensitivity_graph";
    }

private:
    static Cache make_cache(float horizon, float dt) {
        if constexpr (std::is_constructible_v<Cache, float, float>)
            return Cache(horizon, dt);
        else
            return Cache{};
    }

    Cache cache_;
    std::optional<ExecutionPlan> plan_;
};

template<typename Graph, typename Models, typename Products, typename Prepare>
int execute_rough_graph_dataset(
    const datasets::price_gradients::Recipe& recipe,
    cuda_tuning::PricingIdentity identity, std::uint64_t seed,
    const Models& models, const Products& products, Prepare prepare,
    std::size_t paths_per_price = cuda_tuning::kProductionPathsPerPrice
) {
    if (recipe.orders != pg::SensitivityOrders::first
        || recipe.sensitivity_strategy != pg::SensitivityStrategy::node_graph) {
        std::cerr << "Rough datasets require a first-order node graph.\n";
        return 1;
    }
    const auto host = prepare(models, products, recipe.construction,
                              recipe.time, recipe.configuration);
    float horizon = 1.0f / 252.0f;
    for (const auto& product : products)
        horizon = std::max(horizon,
            static_cast<float>(product.maturity_days) / 252.0f);
    return execute_prepared_dataset<true, pg::SensitivityOrders::first>(
        recipe, identity, seed, host,
        RoughGraphExecution<Graph>(horizon, recipe.time.dt), nullptr,
        paths_per_price
    );
}

} // namespace ai_factory::workbench::offline::pricing::price_gradients
