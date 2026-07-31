// Public launcher for the Heston American-put CUDA implementation.
#pragma once

#include "model/heston/dataset.hpp"
#include "product/american_put/dataset.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::heston {

// Summarize the device-dependent batching selected for one complete run.
struct AmericanPutLaunchResult {
    double kernel_seconds;
    std::size_t batch_count;
    std::size_t kernel_launch_count;
    std::size_t maximum_prices_per_batch;
    std::size_t blocks_per_price;
    std::size_t workspace_bytes;
};

// Price every row with multi-block Longstaff-Schwartz batches.
AmericanPutLaunchResult launch_heston_american_put_cuda(
    const HestonModelParameters* device_models,
    std::size_t model_count,
    const product::AmericanPutParameters* host_products,
    const product::AmericanPutParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t monte_carlo_paths_per_price,
    float target_dt,
    unsigned int threads_per_block,
    std::size_t blocks_per_price,
    std::uint64_t base_seed,
    float* device_prices,
    float* device_standard_errors
);

}  // namespace ai_factory::workbench::heston
