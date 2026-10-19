// Offline first-order rough graph datasets; node and checkpoint handling reuse
// the shared price-gradient writer and Philox launch plan.
#pragma once

#include "tools/pricing/price_gradients/generation.cuh"

#include <algorithm>
#include <limits>
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
        : horizon_(horizon), dt_(dt), cache_(make_cache(horizon, dt)) {}

    std::size_t workspace_bytes(
        const auto& host, const pg::LaunchConfiguration& launch
    ) {
        if constexpr (requires(const Cache& cache) { cache.entries(); }) {
            auto complete = launch;
            complete.result_offset = 0U;
            complete.result_count = host.result_count;
            if (host.construction == PriceConstruction::CartesianProduct
                && host.models.size() > 1U) {
                // A global Cartesian cache can crowd out the convolution
                // workspace. Reserve the maximum needed by one model.
                model_span_ = host.products.size();
                std::size_t maximum_bytes = 0U;
                for (std::size_t model = 0U;
                     model < host.models.size(); ++model) {
                    Cache scoped_cache = make_cache(horizon_, dt_);
                    auto scoped_launch = complete;
                    scoped_launch.result_offset = model * model_span_;
                    scoped_launch.result_count = model_span_;
                    auto scoped_plan = Graph::plan(
                        host, scoped_cache, scoped_launch
                    );
                    maximum_bytes = std::max(
                        maximum_bytes, scoped_plan.bytes
                    );
                    if (model == 0U) {
                        cache_ = std::move(scoped_cache);
                        plan_ = std::move(scoped_plan);
                        active_model_ = 0U;
                    }
                }
                workspace_capacity_ = maximum_bytes;
                return maximum_bytes;
            }
            plan_ = Graph::plan(host, cache_, complete);
        } else {
            plan_ = Graph::plan(host, cache_, launch);
        }
        workspace_capacity_ = plan_->bytes;
        return plan_->bytes;
    }

    void launch(
        const auto& host, const auto&,
        mcpg::DevicePreparedStencilOutputs<4U> stencils,
        const pg::LaunchConfiguration& launch,
        pg::Outputs outputs, void* workspace, std::size_t bytes
    ) {
        launch_partitioned(host, stencils, launch,
                           pg::as_sensitivity_outputs(outputs),
                           workspace, bytes);
    }

    void prepare_stencils(
        const auto& host, mcpg::DevicePreparedStencilOutputs<4U> stencils,
        const pg::LaunchConfiguration& launch,
        pg::SensitivityOutputs outputs, void* workspace, std::size_t bytes
    ) {
        // Rebuild deterministic stencils for checkpointed rows with two
        // Philox paths; the checkpoint restores their actual price moments.
        launch_partitioned(host, stencils, launch, outputs,
                           workspace, bytes);
    }

    pg::SensitivityStrategy strategy() const {
        return pg::SensitivityStrategy::node_graph;
    }
    const char* strategy_name() const { return "node_graph"; }
    const char* preparation_name() const {
        return "host_prepared_rough_sensitivity_graph";
    }

private:
    void launch_partitioned(
        const auto& host,
        mcpg::DevicePreparedStencilOutputs<4U> stencils,
        const pg::LaunchConfiguration& launch,
        pg::SensitivityOutputs outputs,
        void* workspace,
        std::size_t bytes
    ) {
        if (!plan_ || bytes < workspace_capacity_)
            throw std::invalid_argument("Rough graph workspace is too small.");
        if (model_span_ == 0U) {
            Graph::launch(host, cache_, launch, outputs, stencils,
                          *plan_, workspace, bytes);
            return;
        }
        const auto end = launch.result_offset + launch.result_count;
        for (std::size_t offset = launch.result_offset;
             offset < end;) {
            const auto model = offset / model_span_;
            if (model != active_model_) {
                cache_ = make_cache(horizon_, dt_);
                auto scoped_launch = launch;
                scoped_launch.result_offset = model * model_span_;
                scoped_launch.result_count = model_span_;
                plan_ = Graph::plan(host, cache_, scoped_launch);
                if (plan_->bytes > workspace_capacity_)
                    throw std::invalid_argument(
                        "Rough graph model exceeds planned workspace."
                    );
                active_model_ = model;
            }
            auto part = launch;
            part.result_offset = offset;
            part.result_count = std::min(end - offset,
                (model + 1U) * model_span_ - offset);
            Graph::launch(host, cache_, part, outputs, stencils,
                          *plan_, workspace, bytes);
            offset += part.result_count;
        }
    }

    static Cache make_cache(float horizon, float dt) {
        if constexpr (std::is_constructible_v<Cache, float, float>)
            return Cache(horizon, dt);
        else
            return Cache{};
    }

    float horizon_;
    float dt_;
    Cache cache_;
    std::optional<ExecutionPlan> plan_;
    std::size_t model_span_ = 0U;
    std::size_t active_model_ = std::numeric_limits<std::size_t>::max();
    std::size_t workspace_capacity_ = 0U;
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
