// Prepare one N-factor plan per model, then reuse the paired-output runner.
#pragma once

#include "tools/pricing/equity_price_delta_generation.cuh"
#include <algorithm>

namespace ai_factory::workbench::offline::pricing {

template<typename ModelLoader, typename ProductLoader, typename Preparation, typename Launcher>
int generate_prepared_price_delta_dataset(const datasets::PriceDeltaRecipe& recipe,
    cuda_tuning::PricingIdentity identity, std::uint64_t seed,
    std::size_t factor_count, const char* method,
    ModelLoader load_models, ProductLoader load_products, Preparation prepare, Launcher launch) {
    try {
        const auto models = load_models(recipe.model_input);
        const auto products = load_products(recipe.product_input);
        const auto wall_start = std::chrono::steady_clock::now();
        if (models.empty() || models.size() != products.size())
            throw std::invalid_argument("N-factor price-delta requires non-empty aligned inputs.");
        float horizon = 1.0f / 252.0f;
        for (const auto& product : products)
            horizon = std::max(horizon, static_cast<float>(product.maturity_days) * (1.0f / 252.0f));
        const auto prepared = prepare(models, horizon);
        if (prepared.size() != models.size())
            throw std::invalid_argument("N-factor preparation must return one row per model.");
        using Prepared = typename decltype(prepared)::value_type;
        cuda::DeviceBuffer<Prepared> device_prepared(prepared.size());
        device_prepared.copy_from(prepared.data());
        return execute_equity_price_delta_dataset<true, false>(
            recipe, identity, seed, models, products,
            [&](const auto* host_models, const auto* device_models, std::size_t model_count,
                const auto* host_products, const auto* device_products, std::size_t product_count,
                const PriceDeltaLaunchContext& context,
                float* prices, float* price_errors, float* deltas, float* delta_errors) {
                launch(host_models, device_models, model_count, device_prepared.data(), model_count,
                    host_products, device_products, product_count, context,
                    prices, price_errors, deltas, delta_errors);
            }, wall_start, {{"method", method}, {"factor_count", factor_count},
                            {"approximation_horizon", horizon},
                            {"approximation_horizon_rule", "maximum product maturity in years"},
                            {"coefficient_precision", "host FP64, device FP32"}});
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

}  // namespace ai_factory::workbench::offline::pricing
