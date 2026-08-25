// Shared CUDA Runtime error check used by host generators and CUDA launchers.
// It keeps every call site explicit while standardizing the exception message.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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

// Accept dynamically composed operation labels without temporary c_str calls.
inline void check_cuda(cudaError_t status, const std::string& operation) {
    check_cuda(status, operation.c_str());
}

// Multiply CUDA allocation dimensions while rejecting size_t overflow.
inline std::size_t checked_workspace_product(
    std::size_t left,
    std::size_t right,
    const char* message
) {
    if (left != 0U
        && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error(message);
    }
    return left * right;
}

// Clamp the configured grid to the number of results available to one launch.
inline std::size_t bounded_block_count(
    std::size_t result_count,
    std::size_t block_count
) {
    if (result_count == 0U || block_count == 0U) {
        throw std::invalid_argument(
            "Block planning requires positive result and block counts."
        );
    }
    return std::min(result_count, block_count);
}

// Reject an empty grid or a grid containing blocks with no result row.
inline void validate_block_count(
    std::size_t result_count,
    std::size_t block_count
) {
    if (result_count == 0U || block_count == 0U) {
        throw std::invalid_argument(
            "A CUDA launch requires positive result and block counts."
        );
    }
    if (block_count > result_count) {
        throw std::invalid_argument(
            "A CUDA launch cannot use more blocks than result rows."
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

// Validate the path count shared by every Monte Carlo pricing kernel.
inline void validate_monte_carlo_path_count(std::size_t paths_per_result) {
    if (paths_per_result < 2U) {
        throw std::invalid_argument(
            "Monte Carlo pricing requires at least two paths per result."
        );
    }
}

// Validate the year fraction represented by one contractual calendar day.
inline void validate_day_fraction(float day_fraction) {
    if (!(day_fraction > 0.0f) || !std::isfinite(day_fraction)) {
        throw std::invalid_argument(
            "day_fraction must be positive and finite."
        );
    }
}

// Validate one numerical time step independently of Monte Carlo sampling.
inline void validate_time_step(float dt) {
    if (!(dt > 0.0f) || !std::isfinite(dt)) {
        throw std::invalid_argument("dt must be positive and finite.");
    }
}

// Validate the path count and step requested by a discretized simulation.
inline void validate_monte_carlo_parameters(
    std::size_t paths_per_result,
    float dt
) {
    validate_monte_carlo_path_count(paths_per_result);
    validate_time_step(dt);
}

// Reject an empty numerical grid within one product business day.
inline void validate_simulation_steps_per_day(
    std::uint32_t simulation_steps_per_day
) {
    if (simulation_steps_per_day == 0U) {
        throw std::invalid_argument(
            "simulation_steps_per_day must be positive."
        );
    }
}

// Validate a positive block size against the current CUDA device limit.
inline void validate_cuda_block_size(unsigned int threads_per_block) {
    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    check_cuda(
        cudaGetDeviceProperties(&properties, device),
        "cudaGetDeviceProperties"
    );
    if (threads_per_block == 0U
        || threads_per_block
            > static_cast<unsigned int>(properties.maxThreadsPerBlock)) {
        throw std::invalid_argument(
            "threads_per_block must be positive and within the device limit."
        );
    }
}

// Require whole warps for kernels using warp-based block reductions.
inline void validate_reduction_block_size(unsigned int threads_per_block) {
    validate_cuda_block_size(threads_per_block);
    if (threads_per_block % 32U != 0U) {
        throw std::invalid_argument(
            "A reduction block must contain a whole number of warps."
        );
    }
}

// Validate one requested gridDim.x against the current device limit.
inline void validate_grid_x_size(std::size_t block_count) {
    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    check_cuda(
        cudaGetDeviceProperties(&properties, device),
        "cudaGetDeviceProperties"
    );
    if (block_count
        > static_cast<std::size_t>(properties.maxGridSize[0])) {
        throw std::overflow_error(
            "Block count exceeds the current device gridDim.x limit."
        );
    }
}

// Validate the mapping seed(row) = base_seed + row_index.
inline void validate_row_seed_range(
    std::size_t result_count,
    std::uint64_t base_seed
) {
    if (result_count > 0U
        && result_count - 1U
            > std::numeric_limits<std::uint64_t>::max() - base_seed) {
        throw std::overflow_error("Row seed exceeds uint64_t.");
    }
}

}  // namespace ai_factory::workbench
