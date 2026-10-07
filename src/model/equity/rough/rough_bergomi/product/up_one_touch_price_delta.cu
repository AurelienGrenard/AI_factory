// Rough-Bergomi up-one-touch legacy delta API backed by the rough graph.
#include "model/equity/rough/rough_bergomi/product/up_one_touch_price_delta.cuh"
#include "model/equity/rough/rough_bergomi/product/up_one_touch_price_gradients.cuh"
#include "common/volterra/price_gradients/legacy_spot_delta_graph.cuh"

namespace ai_factory::workbench::model::equity::rough_bergomi {

void launch_rough_bergomi_up_one_touch_price_delta_cuda(
    const ModelParameters* host_models,
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::UpOneTouchParameters* host_products,
    const product::UpOneTouchParameters* device_products,
    std::size_t product_count,
    PriceConstruction construction,
    std::size_t result_count,
    std::size_t result_index,
    std::size_t monte_carlo_paths_per_price,
    float day_fraction,
    float target_dt,
    std::size_t step_count,
    std::size_t path_chunk_size,
    void* device_workspace,
    std::size_t workspace_bytes,
    std::uint64_t base_seed,
    ::ai_factory::workbench::equity::price_delta::SpotBumpConfiguration bump,
    float* device_prices,
    float* device_standard_errors,
    float* device_deltas,
    float* device_delta_errors
) {
    (void)device_models;
    (void)device_products;
    using Graph = UpOneTouchNodeGraph<::ai_factory::workbench::price_gradients::SensitivityOrders::first, 1U>;
    volterra::price_gradients::launch_legacy_fft_spot_delta<Graph>(
        host_models, model_count, host_products, product_count,
        construction, result_count, result_index,
        monte_carlo_paths_per_price, day_fraction, target_dt,
        step_count, path_chunk_size, device_workspace, workspace_bytes,
        base_seed, bump, device_prices, device_standard_errors,
        device_deltas, device_delta_errors,
        prepare_rough_bergomi_up_one_touch_sensitivities
    );
}


}  // namespace ai_factory::workbench::model::equity::rough_bergomi
