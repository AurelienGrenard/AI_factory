// Shared CUDA Runtime error check used by host generators and CUDA launchers.
// It keeps every call site explicit while standardizing the exception message.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench {

// Throw a readable C++ exception when one CUDA Runtime operation fails.
inline void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status)
        );
    }
}

// Confirm that a pointer passed to a kernel comes from cudaMalloc.
inline void validate_device_pointer(const void* pointer, const char* name) {
    if (pointer == nullptr) {
        throw std::invalid_argument(std::string(name) + " is null.");
    }
    cudaPointerAttributes attributes{};
    const cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
    if (status != cudaSuccess || attributes.type != cudaMemoryTypeDevice) {
        throw std::invalid_argument(
            std::string(name) + " must point to device memory allocated by CUDA."
        );
    }
}

// Validate aligned or Cartesian construction from one model and one product.
inline void validate_model_product_construction(
    std::size_t model_count,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count
) {
    if (model_count == 0U || product_count == 0U || result_count == 0U) {
        throw std::invalid_argument(
            "Model, product, and result counts must be positive."
        );
    }
    if (!cartesian_product) {
        if (model_count != product_count || result_count != model_count) {
            throw std::invalid_argument(
                "Aligned construction requires equal model, product, and result counts."
            );
        }
        return;
    }
    if (model_count > std::numeric_limits<std::size_t>::max() / product_count) {
        throw std::overflow_error("Cartesian result count exceeds size_t.");
    }
    if (result_count != model_count * product_count) {
        throw std::invalid_argument(
            "Cartesian result count must equal model_count * product_count."
        );
    }
}

// Validate aligned or Cartesian construction from model, curve, and product.
inline void validate_model_curve_product_construction(
    std::size_t model_count,
    std::size_t curve_count,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count
) {
    if (model_count == 0U || curve_count == 0U || product_count == 0U
        || result_count == 0U) {
        throw std::invalid_argument(
            "Model, curve, product, and result counts must be positive."
        );
    }
    if (!cartesian_product) {
        if (model_count != curve_count || model_count != product_count
            || result_count != model_count) {
            throw std::invalid_argument(
                "Aligned construction requires equal model, curve, product, and result counts."
            );
        }
        return;
    }
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    if (model_count > maximum / curve_count
        || model_count * curve_count > maximum / product_count) {
        throw std::overflow_error("Cartesian result count exceeds size_t.");
    }
    if (result_count != model_count * curve_count * product_count) {
        throw std::invalid_argument(
            "Cartesian result count must equal model_count * curve_count * product_count."
        );
    }
}

// Validate parameters shared by discretized Monte Carlo pricing kernels.
inline void validate_monte_carlo_parameters(
    std::size_t paths_per_result,
    float target_dt
) {
    if (paths_per_result < 2U) {
        throw std::invalid_argument(
            "Monte Carlo pricing requires at least two paths per result."
        );
    }
    if (!(target_dt > 0.0f)) {
        throw std::invalid_argument("target_dt must be positive.");
    }
}

// Validate a block size used by warp-based reductions such as reduce_block().
inline void validate_warp_aligned_block_size(unsigned int threads_per_block) {
    if (threads_per_block == 0U || threads_per_block > 1024U
        || threads_per_block % 32U != 0U) {
        throw std::invalid_argument(
            "threads_per_block must be a multiple of 32 between 32 and 1024."
        );
    }
}

// Validate a kernel mapping exactly one CUDA block to each result row.
inline void validate_one_block_per_result_grid(std::size_t result_count) {
    if (result_count > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error("CUDA grid exceeds gridDim.x.");
    }
}

// Validate the mapping seed(row) = base_seed + row_index.
inline void validate_row_seed_range(
    std::size_t result_count,
    std::uint64_t base_seed
) {
    if (result_count > std::numeric_limits<std::uint64_t>::max() - base_seed) {
        throw std::overflow_error("Row seed exceeds uint64_t.");
    }
}

}  // namespace ai_factory::workbench
