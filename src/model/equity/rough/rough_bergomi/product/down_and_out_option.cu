// Rough-Bergomi down-and-out-option composition over the Volterra FFT engine.
#include "model/equity/rough/rough_bergomi/product/down_and_out_option.cuh"

#include "common/volterra/hybrid_schedule.cuh"
#include "model/equity/rough/rough_bergomi/volterra_fft_pricing.cuh"
#include "product/down_and_out_option/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::rough_bergomi {

template<OptionSide Side>
void launch_rough_bergomi_down_and_out_option_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::DownAndOutOptionParameters* device_products,
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
    using ProductPolicy = product::DownAndOutOptionPathPolicy<Side>;
    launch_rough_bergomi_hybrid_fft_cuda<
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
        "rough_bergomi.down_and_out_option",
        option_side_name(Side)
    );
}

template void launch_rough_bergomi_down_and_out_option_cuda<
    OptionSide::call
>(
    const ModelParameters*, std::size_t,
    const product::DownAndOutOptionParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, std::size_t,
    float, float, std::size_t, std::size_t,
    void*, std::size_t, std::uint64_t, float*, float*
);

template void launch_rough_bergomi_down_and_out_option_cuda<
    OptionSide::put
>(
    const ModelParameters*, std::size_t,
    const product::DownAndOutOptionParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, std::size_t,
    float, float, std::size_t, std::size_t,
    void*, std::size_t, std::uint64_t, float*, float*
);

}  // namespace ai_factory::workbench::model::equity::rough_bergomi
