// Product-generic causal FFT binding for quadratic rough Heston.
#pragma once

#include "common/volterra/causal_fft_pricer.cuh"
#include "model/equity/rough/quadratic_rough_heston/causal_fft_dynamics.cuh"

namespace ai_factory::workbench::model::equity::quadratic_rough_heston {

template<typename ProductPolicy, typename SchedulePolicy>
inline auto plan_causal_fft_workspace(
    std::size_t steps, std::size_t paths_per_price,
    std::size_t path_chunk_size, std::size_t row_batch_size
) {
    return volterra::causal_fft::plan_workspace<
        CausalFftPathPolicy, ProductPolicy, SchedulePolicy
    >(steps, paths_per_price, path_chunk_size, row_batch_size);
}

template<typename ProductPolicy, typename SchedulePolicy>
void launch_quadratic_rough_heston_causal_fft_cuda(
    const ModelParameters* models, std::size_t model_count,
    const typename ProductPolicy::ProductParameters* products,
    std::size_t product_count, PriceConstruction construction,
    std::size_t result_count, std::size_t result_offset,
    std::size_t launch_result_count, std::size_t paths_per_price,
    volterra::HybridTimeConfiguration time_config, std::uint32_t steps,
    std::uint32_t tile_size, std::size_t path_chunk_size,
    std::size_t row_batch_size, void* workspace,
    std::size_t workspace_bytes, std::uint64_t seed,
    float* prices, float* standard_errors
) {
    volterra::causal_fft::launch_pricing_cuda<
        CausalFftPathPolicy, ProductPolicy, SchedulePolicy
    >(
        models, model_count, products, product_count, construction,
        result_count, result_offset, launch_result_count, paths_per_price,
        time_config, steps, tile_size, path_chunk_size, row_batch_size,
        workspace, workspace_bytes, seed, prices, standard_errors
    );
}

}  // namespace ai_factory::workbench::model::equity::quadratic_rough_heston
