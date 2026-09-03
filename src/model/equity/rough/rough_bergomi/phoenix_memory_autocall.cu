// Rough-Bergomi phoenix-memory-autocall composition over the Volterra FFT engine.
#include "model/equity/rough/rough_bergomi/phoenix_memory_autocall.cuh"

#include "common/volterra/hybrid_schedule.cuh"
#include "model/equity/rough/rough_bergomi/hybrid_pricing.cuh"
#include "product/phoenix_memory_autocall/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::rough_bergomi {

void launch_rough_bergomi_phoenix_memory_autocall_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::PhoenixMemoryAutocallParameters* device_products,
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
    using ProductPolicy = product::PhoenixMemoryAutocallPathPolicy;
    launch_rough_bergomi_hybrid_fft_cuda<
        ProductPolicy,
        volterra::RegularHybridSchedule
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
        "rough_bergomi.phoenix_memory_autocall",
        "default"
    );
}


}  // namespace ai_factory::workbench::model::equity::rough_bergomi
