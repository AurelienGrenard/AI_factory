// Generic rough-Bergomi binding for Gaussian-Volterra product policies.
#pragma once

#include "common/price_construction.cuh"

#include "common/volterra/fractional_hybrid_kernel.cuh"
#include "common/volterra/hybrid_fft_pricer.cuh"
#include "model/equity/rough/rough_bergomi/parameters.hpp"

#include "model/equity/rough/rough_bergomi/dynamics_impl.cuh"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::equity::rough_bergomi {

// SchedulePolicy selects terminal, dense, regular or static-calendar
// observations. ProductPolicy supplies ProductParameters, calendar,
// PreparedProduct, Handler, prepare_product, make_handler and finalize.
template<typename ProductPolicy, typename SchedulePolicy>
void launch_rough_bergomi_hybrid_fft_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const typename ProductPolicy::ProductParameters* device_products,
    std::size_t product_count,
    PriceConstruction construction,
    std::size_t result_count,
    std::size_t result_index,
    std::size_t monte_carlo_paths_per_price,
    volterra::HybridTimeConfiguration time_configuration,
    std::size_t step_count,
    std::size_t path_chunk_size,
    void* device_workspace,
    std::size_t workspace_bytes,
    std::uint64_t base_seed,
    float* device_prices,
    float* device_standard_errors,
    const char* diagnostic_name,
    const char* diagnostic_variant
) {
    volterra::hybrid_fft::launch_pricing_cuda<
        volterra::FractionalHybridKernelPolicy,
        PathPolicy,
        ProductPolicy,
        SchedulePolicy
    >(
        device_models,
        model_count,
        device_products,
        product_count,
        construction,
        result_count,
        result_index,
        monte_carlo_paths_per_price,
        time_configuration,
        step_count,
        path_chunk_size,
        device_workspace,
        workspace_bytes,
        base_seed,
        device_prices,
        device_standard_errors,
        diagnostic_name,
        diagnostic_variant
    );
}

}  // namespace ai_factory::workbench::model::equity::rough_bergomi
