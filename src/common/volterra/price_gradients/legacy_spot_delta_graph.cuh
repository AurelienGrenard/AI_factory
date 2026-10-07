// Compatibility launchers for the historical rough price/spot-delta API.
// All path simulation and paired moments are delegated to the rough node graph.
#pragma once

#include "common/check_cuda.cuh"
#include "common/equity/price_delta/spot_bump.cuh"
#include "common/price_gradients/configuration.hpp"
#include "common/price_gradients/sensitivity_outputs.cuh"
#include "common/result_index.cuh"
#include "common/volterra/hybrid_fft_price_delta_workspace.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace ai_factory::workbench::volterra::price_gradients {

namespace delta_pg = ::ai_factory::workbench::price_gradients;
namespace delta_mcpg = ::ai_factory::workbench::monte_carlo::price_gradients;
namespace legacy_delta = ::ai_factory::workbench::equity::price_delta;

class LegacyGraphStorage {
public:
    LegacyGraphStorage() = default;
    LegacyGraphStorage(const LegacyGraphStorage&) = delete;
    LegacyGraphStorage& operator=(const LegacyGraphStorage&) = delete;
    ~LegacyGraphStorage() { if (data_) cudaFree(data_); }

    void* ensure(std::size_t bytes) {
        if (bytes > capacity_) {
            void* replacement = nullptr;
            check_cuda(cudaMalloc(&replacement, bytes),
                       "rough delta graph allocation");
            if (data_) cudaFree(data_);
            data_ = replacement;
            capacity_ = bytes;
        }
        return data_;
    }

private:
    void* data_ = nullptr;
    std::size_t capacity_ = 0U;
};

inline delta_pg::PriceGradientConfiguration legacy_spot_configuration(
    legacy_delta::SpotBumpConfiguration bump
) {
    legacy_delta::validate_device_context(bump);
    return {{{"model.spot", {
        static_cast<double>(0.5f * bump.relative_width),
        delta_pg::BumpScale::relative,
        delta_pg::BoundaryRule::central_only
    }}}};
}

template<typename Graph, typename HostPlan>
void launch_legacy_rough_graph(
    const HostPlan& host,
    typename Graph::Cache& cache,
    std::size_t paths,
    unsigned threads,
    std::size_t blocks,
    std::uint64_t seed,
    float* prices,
    float* price_errors,
    float* deltas,
    float* delta_errors,
    void* supplied_workspace = nullptr,
    std::size_t supplied_bytes = 0U
) {
    if (!prices || !price_errors || !deltas || !delta_errors)
        throw std::invalid_argument("Rough delta graph requires four output arrays.");
    const delta_pg::LaunchConfiguration launch{
        delta_pg::PricingMethod::monte_carlo,
        0U, host.result_count, paths, threads,
        std::max<std::size_t>(blocks, 1U), seed, 1U
    };
    const auto plan = Graph::plan(host, cache, launch);
    static thread_local LegacyGraphStorage storage;
    static thread_local LegacyGraphStorage stencil_storage;
    void* workspace = supplied_workspace;
    std::size_t workspace_bytes = supplied_bytes;
    if (workspace_bytes < plan.bytes) {
        workspace = storage.ensure(plan.bytes);
        workspace_bytes = plan.bytes;
    }
    auto* stencils = static_cast<delta_pg::SensitivityStencil<4U>*>(
        stencil_storage.ensure(
            host.result_count * sizeof(delta_pg::SensitivityStencil<4U>)
        )
    );
    Graph::launch(
        host, cache, launch,
        {prices, price_errors, deltas, delta_errors,
         nullptr, nullptr, host.result_count, host.result_count},
        delta_mcpg::DevicePreparedStencilOutputs<4U>{
            stencils, host.result_count, nullptr
        },
        plan, workspace, workspace_bytes
    );
}

template<typename Graph, typename Model, typename Product, typename Prepare>
void launch_legacy_fft_spot_delta(
    const Model* host_models,
    std::size_t model_count,
    const Product* host_products,
    std::size_t product_count,
    PriceConstruction construction,
    std::size_t result_count,
    std::size_t result_index,
    std::size_t paths,
    float day_fraction,
    float target_dt,
    std::size_t step_count,
    std::size_t path_chunk_size,
    void* device_workspace,
    std::size_t workspace_bytes,
    std::uint64_t base_seed,
    legacy_delta::SpotBumpConfiguration bump,
    float* prices,
    float* price_errors,
    float* deltas,
    float* delta_errors,
    Prepare prepare
) {
    if (!host_models || !host_products || !device_workspace
        || !prices || !price_errors || !deltas || !delta_errors
        || result_count != price_row_count(
            model_count, product_count, construction)
        || result_index >= result_count || paths == 0U
        || !std::isfinite(day_fraction) || !std::isfinite(target_dt)
        || !(day_fraction > 0.0f) || !(target_dt > 0.0f)) {
        throw std::invalid_argument("Invalid rough FFT delta graph inputs.");
    }
    const float ratio = day_fraction / target_dt;
    const auto steps_per_day = static_cast<std::uint32_t>(std::lround(ratio));
    if (steps_per_day == 0U
        || std::fabs(ratio - static_cast<float>(steps_per_day)) > 1.0e-5f)
        throw std::invalid_argument("Invalid rough FFT delta time grid.");
    const auto indices = decode_model_product_result_index(
        result_index, product_count, construction
    );
    const Model& model = host_models[indices.model_index];
    const Product& product = host_products[indices.product_index];
    legacy_delta::validate_spot_bump(model.spot, bump);
    if (step_count != static_cast<std::size_t>(product.maturity_days)
            * steps_per_day
        || workspace_bytes < required_hybrid_fft_price_delta_workspace_bytes(
            step_count, paths, path_chunk_size
        )) {
        throw std::invalid_argument("Invalid rough FFT delta workspace or step count.");
    }
    const auto host = prepare(
        std::span<const Model>(&model, 1U),
        std::span<const Product>(&product, 1U),
        PriceConstruction::Aligned,
        delta_pg::TimeConfiguration{target_dt, steps_per_day},
        legacy_spot_configuration(bump),
        delta_pg::SensitivityRequest{delta_pg::SensitivityOrders::first}
    );
    typename Graph::Cache cache;
    launch_legacy_rough_graph<Graph>(
        host, cache, paths, 256U, 1U, base_seed + result_index,
        prices + result_index, price_errors + result_index,
        deltas + result_index, delta_errors + result_index,
        device_workspace, workspace_bytes
    );
}

template<typename Graph, typename Model, typename Product,
         typename Prepared, typename Prepare>
void launch_legacy_lift_spot_delta(
    const Model* host_models,
    std::size_t model_count,
    const Prepared* device_prepared,
    std::size_t prepared_count,
    const Product* host_products,
    std::size_t product_count,
    PriceConstruction construction,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    std::size_t paths,
    float dt,
    std::uint32_t steps_per_day,
    unsigned threads,
    std::size_t blocks,
    std::uint64_t base_seed,
    legacy_delta::SpotBumpConfiguration bump,
    float* prices,
    float* price_errors,
    float* deltas,
    float* delta_errors,
    Prepare prepare
) {
    if (!host_models || !host_products || !device_prepared
        || !prices || !price_errors || !deltas || !delta_errors
        || result_count != price_row_count(
            model_count, product_count, construction)
        || result_offset >= result_count
        || launch_result_count == 0U
        || launch_result_count > result_count - result_offset
        || prepared_count < model_count || paths == 0U) {
        throw std::invalid_argument("Invalid rough lift delta graph inputs.");
    }
    std::vector<Model> models;
    std::vector<Product> products;
    models.reserve(launch_result_count);
    products.reserve(launch_result_count);
    float horizon = 1.0f / 252.0f;
    for (std::size_t local = 0U; local < launch_result_count; ++local) {
        const auto indices = decode_model_product_result_index(
            result_offset + local, product_count, construction
        );
        const auto& model = host_models[indices.model_index];
        const auto& product = host_products[indices.product_index];
        legacy_delta::validate_spot_bump(model.spot, bump);
        models.push_back(model);
        products.push_back(product);
        horizon = std::max(
            horizon, static_cast<float>(product.maturity_days)
                / 252.0f
        );
    }
    const auto host = prepare(
        std::span<const Model>(models),
        std::span<const Product>(products),
        PriceConstruction::Aligned,
        delta_pg::TimeConfiguration{dt, steps_per_day},
        legacy_spot_configuration(bump),
        delta_pg::SensitivityRequest{delta_pg::SensitivityOrders::first}
    );
    typename Graph::Cache cache(horizon, dt);
    std::vector<Prepared> prepared_by_model(model_count);
    std::vector<unsigned char> prepared_loaded(model_count, 0U);
    for (std::size_t local = 0U; local < launch_result_count; ++local) {
        const auto indices = decode_model_product_result_index(
            result_offset + local, product_count, construction
        );
        if (!prepared_loaded[indices.model_index]) {
            check_cuda(cudaMemcpy(
                &prepared_by_model[indices.model_index],
                device_prepared + indices.model_index,
                sizeof(Prepared), cudaMemcpyDeviceToHost
            ), "rough delta prepared dynamics download");
            prepared_loaded[indices.model_index] = 1U;
        }
        const Prepared& prepared = prepared_by_model[indices.model_index];
        cache.insert_prepared(models[local], prepared);
    }
    launch_legacy_rough_graph<Graph>(
        host, cache, paths, threads, blocks,
        base_seed + result_offset,
        prices + result_offset, price_errors + result_offset,
        deltas + result_offset, delta_errors + result_offset
    );
}

}  // namespace ai_factory::workbench::volterra::price_gradients
