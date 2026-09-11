// Log-modulated rough-Bergomi digital-option composition over the Volterra FFT engine.
#include "model/equity/rough/log_modulated_rough_bergomi/product/digital_option_price_delta.cuh"

#include "common/volterra/hybrid_schedule.cuh"
#include "common/volterra/log_modulated_hybrid_kernel.cuh"
#include "common/volterra/hybrid_fft_price_delta.cuh"
#include "model/equity/rough/log_modulated_rough_bergomi/dynamics_impl.cuh"
#include "product/digital_option/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::log_modulated_rough_bergomi {

template<OptionSide Side>
void launch_log_modulated_rough_bergomi_digital_option_price_delta_cuda(
    const ModelParameters* host_models,
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::DigitalOptionParameters* host_products,
    const product::DigitalOptionParameters* device_products,
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
    using ProductPolicy = product::DigitalOptionPathPolicy<Side>;
    volterra::hybrid_fft::launch_price_delta_cuda<
        volterra::LogModulatedHybridKernelPolicy, PathPolicy, ProductPolicy, volterra::TerminalHybridSchedule,
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
        "log_modulated_rough_bergomi.digital_option.price_delta",
        option_side_name(Side)
    );
}

template void launch_log_modulated_rough_bergomi_digital_option_price_delta_cuda<
    OptionSide::call
>(
    const ModelParameters*, const ModelParameters*, std::size_t,
    const product::DigitalOptionParameters*, const product::DigitalOptionParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, std::size_t,
    float, float, std::size_t, std::size_t,
    void*, std::size_t, std::uint64_t, ::ai_factory::workbench::equity::price_delta::SpotBumpConfiguration, float*, float*, float*, float*
);

template void launch_log_modulated_rough_bergomi_digital_option_price_delta_cuda<
    OptionSide::put
>(
    const ModelParameters*, const ModelParameters*, std::size_t,
    const product::DigitalOptionParameters*, const product::DigitalOptionParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, std::size_t,
    float, float, std::size_t, std::size_t,
    void*, std::size_t, std::uint64_t, ::ai_factory::workbench::equity::price_delta::SpotBumpConfiguration, float*, float*, float*, float*
);

}  // namespace ai_factory::workbench::model::equity::log_modulated_rough_bergomi
