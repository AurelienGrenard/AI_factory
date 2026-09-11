// Rough Stein-Stein up-one-touch composition over the Volterra FFT engine.
#include "model/equity/rough/rough_stein_stein/product/up_one_touch_price_delta.cuh"

#include "common/volterra/hybrid_schedule.cuh"
#include "common/volterra/fractional_resolvent_hybrid_kernel.cuh"
#include "common/volterra/hybrid_fft_price_delta.cuh"
#include "model/equity/rough/rough_stein_stein/dynamics_impl.cuh"
#include "product/up_one_touch/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::rough_stein_stein {

void launch_rough_stein_stein_up_one_touch_price_delta_cuda(
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
    using ProductPolicy = product::UpOneTouchPathPolicy;
    volterra::hybrid_fft::launch_price_delta_cuda<
        volterra::FractionalResolventHybridKernelPolicy, PathPolicy, ProductPolicy, volterra::DenseHybridSchedule,
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
        "rough_stein_stein.up_one_touch.price_delta",
        "default"
    );
}


}  // namespace ai_factory::workbench::model::equity::rough_stein_stein
