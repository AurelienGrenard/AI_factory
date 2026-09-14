// Reuse one convolution workspace across constructed rough price-delta rows.
#pragma once
#include "tools/pricing/equity_price_delta_generation.cuh"
#include "common/volterra/hybrid_fft_price_delta_workspace.cuh"
#include "common/volterra/hybrid_schedule.cuh"
#include <algorithm>

namespace ai_factory::workbench::offline::pricing {

template<typename Schedule, typename ProductPolicy, typename ModelLoader, typename ProductLoader, typename Launcher>
int generate_volterra_price_delta_dataset(const datasets::PriceDeltaRecipe& recipe,
    cuda_tuning::PricingIdentity identity, std::uint64_t seed, const char* method,
    ModelLoader load_models, ProductLoader load_products, Launcher launch) {
    try {
        const auto models = load_models(recipe.model_input);
        const auto products = load_products(recipe.product_input);
        const auto wall_start = std::chrono::steady_clock::now();
        const auto result_count = price_row_count(
            models.size(), products.size(), recipe.construction);
        const auto plan = cuda_tuning::make_equity_price_delta_launch_plan(identity, result_count);
        const volterra::HybridTimeConfiguration time{1.f / 252.f, 1.f / 504.f};
        std::vector<std::size_t> steps;
        for (const auto& product : products) {
            const auto calendar = ProductPolicy::calendar(product);
            simulation::validate_calendar(calendar);
            steps.push_back(Schedule::execution_step_count(calendar, time));
        }
        std::vector<std::size_t> result_steps;
        result_steps.reserve(result_count);
        for (std::size_t result_index = 0; result_index < result_count; ++result_index) {
            const auto indices = decode_model_product_result_index(
                result_index, products.size(), recipe.construction);
            result_steps.push_back(steps.at(indices.product_index));
        }
        const auto maximum = *std::max_element(steps.begin(), steps.end());
        const auto workspace = volterra::plan_hybrid_fft_price_delta_workspace(
            maximum, plan.paths_per_price, plan.profile.path_chunk_size);
        cuda::DeviceBuffer<unsigned char> storage(workspace.workspace_bytes);
        return execute_equity_price_delta_dataset<true, false>(recipe, identity, seed, models, products,
            [&](const auto* host_models, const auto* device_models, std::size_t model_count,
                const auto* host_products, const auto* device_products, std::size_t product_count,
                const PriceDeltaLaunchContext& context,
                float* prices, float* price_errors, float* deltas, float* delta_errors) {
                for (std::size_t row = context.offset; row < context.offset + context.count; ++row)
                    launch(host_models, device_models, model_count, host_products, device_products,
                        product_count, recipe.construction, context.results, row, context.paths, time.day_fraction,
                        time.target_dt, result_steps[row], plan.profile.path_chunk_size,
                        storage.data(), workspace.workspace_bytes, context.seed, context.bump,
                        prices, price_errors, deltas, delta_errors);
            }, wall_start, {{"method", method}, {"step_counts", result_steps},
                            {"maximum_step_count", maximum}, {"workspace_bytes", workspace.workspace_bytes},
                            {"shared_convolution", true}});
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

}  // namespace ai_factory::workbench::offline::pricing
