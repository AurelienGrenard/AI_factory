// Log-modulated rough-Bergomi up-one-touch composition over the Volterra FFT engine.
#include "model/equity/rough/log_modulated_rough_bergomi/product/up_one_touch.cuh"

#include "common/volterra/hybrid_schedule.cuh"
#include "model/equity/rough/log_modulated_rough_bergomi/volterra_fft_pricing.cuh"
#include "product/up_one_touch/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::log_modulated_rough_bergomi {

void launch_log_modulated_rough_bergomi_up_one_touch_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
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
    float* device_prices,
    float* device_standard_errors
) {
    using ProductPolicy = product::UpOneTouchPathPolicy;
    launch_log_modulated_rough_bergomi_hybrid_fft_cuda<
        ProductPolicy,
        volterra::DenseHybridSchedule
    >(
        device_models,
        model_count,
        device_products,
        product_count,
        construction,
        result_count,
        result_index,
        monte_carlo_paths_per_price,
        volterra::HybridTimeConfiguration{day_fraction, target_dt},
        step_count,
        path_chunk_size,
        device_workspace,
        workspace_bytes,
        base_seed,
        device_prices,
        device_standard_errors,
        "log_modulated_rough_bergomi.up_one_touch",
        "default"
    );
}


}  // namespace ai_factory::workbench::model::equity::log_modulated_rough_bergomi
