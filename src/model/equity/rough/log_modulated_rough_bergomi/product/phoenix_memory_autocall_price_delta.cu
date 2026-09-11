// Log-modulated rough-Bergomi phoenix-memory-autocall composition over the Volterra FFT engine.
#include "model/equity/rough/log_modulated_rough_bergomi/product/phoenix_memory_autocall_price_delta.cuh"

#include "common/volterra/hybrid_schedule.cuh"
#include "common/volterra/log_modulated_hybrid_kernel.cuh"
#include "common/volterra/hybrid_fft_price_delta.cuh"
#include "model/equity/rough/log_modulated_rough_bergomi/dynamics_impl.cuh"
#include "product/phoenix_memory_autocall/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::log_modulated_rough_bergomi {

void launch_log_modulated_rough_bergomi_phoenix_memory_autocall_price_delta_cuda(
    const ModelParameters* host_models,
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::PhoenixMemoryAutocallParameters* host_products,
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
    ::ai_factory::workbench::equity::price_delta::SpotBumpConfiguration bump,
    float* device_prices,
    float* device_standard_errors,
    float* device_deltas,
    float* device_delta_errors
) {
    using ProductPolicy = product::PhoenixMemoryAutocallPathPolicy;
    volterra::hybrid_fft::launch_price_delta_cuda<
        volterra::LogModulatedHybridKernelPolicy, PathPolicy, ProductPolicy, volterra::RegularHybridSchedule,
        ::ai_factory::workbench::equity::price_delta::MultiplicativeVolterraSpotPath<PathPolicy>
    >(
        host_models,
        device_models,
        model_count,
        host_products,
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
        bump,
        device_prices,
        device_standard_errors,
        device_deltas,
        device_delta_errors,
        "log_modulated_rough_bergomi.phoenix_memory_autocall.price_delta",
        "default"
    );
}


}  // namespace ai_factory::workbench::model::equity::log_modulated_rough_bergomi
