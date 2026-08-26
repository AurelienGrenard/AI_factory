// cuFFTDx-fused rough-Bergomi European-option pricing.
#pragma once

#include "common/option_side.cuh"
#include "model/equity/rough_bergomi/parameters.hpp"
#include "product/european_option/parameters.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::equity::rough_bergomi {

// Caller-owned scratch reused sequentially by every result row.
struct RoughBergomiFftWorkspacePlan {
    std::size_t maximum_step_count;
    std::size_t maximum_fft_length;
    std::size_t monte_carlo_paths_per_price;
    std::size_t path_chunk_size;
    std::size_t convolution_bytes;
    std::size_t partial_moment_count;
    std::size_t workspace_bytes;
};

// Size one spectrum, one deterministic-correction vector, and block moments.
RoughBergomiFftWorkspacePlan plan_european_option_fft_workspace(
    std::size_t maximum_step_count,
    std::size_t monte_carlo_paths_per_price,
    std::size_t path_chunk_size
);

// Price one result row. Consecutive calls may reuse the same workspace because
// all preparation, path, and finalization kernels are ordered on one stream.
template<OptionSide Side>
void launch_rough_bergomi_european_option_fft_cuda(
    const RoughBergomiModelParameters* device_models,
    std::size_t model_count,
    const product::EuropeanOptionParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_index,
    std::size_t monte_carlo_paths_per_price,
    float target_dt,
    std::size_t step_count,
    std::size_t path_chunk_size,
    void* device_workspace,
    std::size_t workspace_bytes,
    std::uint64_t base_seed,
    float* device_prices,
    float* device_standard_errors
);

}  // namespace ai_factory::workbench::model::equity::rough_bergomi
