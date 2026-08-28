// Caller-owned workspace contract for chunked Gaussian-Volterra FFT pricing.
#pragma once

#include "common/check_cuda.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace ai_factory::workbench::volterra {

inline constexpr std::size_t kMaximumHybridFftStepCount = 4096U;
inline constexpr std::size_t kMaximumHybridFftLength = 8192U;
inline constexpr std::size_t kHybridFftSpectrumBytes =
    kMaximumHybridFftLength * sizeof(float2);
inline constexpr std::size_t kHybridFftVolterraVarianceBytes =
    kMaximumHybridFftStepCount * sizeof(float);
inline constexpr std::size_t kHybridFftPreparedRowBytes = 1024U;
inline constexpr std::size_t kHybridFftPreparedRowOffset =
    kHybridFftSpectrumBytes + kHybridFftVolterraVarianceBytes;
inline constexpr std::size_t kHybridFftConvolutionOffset =
    kHybridFftPreparedRowOffset + kHybridFftPreparedRowBytes;
inline constexpr std::size_t kHybridFftPathThreads = 256U;

struct HybridFftWorkspacePlan {
    std::size_t maximum_step_count;
    std::size_t maximum_fft_length;
    std::size_t monte_carlo_paths_per_price;
    std::size_t path_chunk_size;
    std::size_t spectrum_bytes;
    std::size_t volterra_variance_bytes;
    std::size_t convolution_bytes;
    std::size_t partial_moment_count;
    std::size_t workspace_bytes;
};

inline std::size_t checked_hybrid_fft_product(
    std::size_t left,
    std::size_t right,
    const char* description
) {
    if (right != 0U
        && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::overflow_error(description);
    }
    return left * right;
}

inline std::size_t hybrid_fft_convolution_bytes(
    std::size_t step_count,
    std::size_t path_chunk_size
) {
    return checked_hybrid_fft_product(
        checked_hybrid_fft_product(
            (path_chunk_size + 1U) / 2U,
            step_count,
            "Volterra hybrid FFT convolution element count overflows size_t."
        ),
        sizeof(float2),
        "Volterra hybrid FFT convolution bytes overflow size_t."
    );
}

inline std::size_t hybrid_fft_partial_moment_count(std::size_t path_count) {
    return (path_count + kHybridFftPathThreads - 1U)
        / kHybridFftPathThreads;
}

inline std::size_t required_hybrid_fft_workspace_bytes(
    std::size_t step_count,
    std::size_t path_count,
    std::size_t path_chunk_size
) {
    constexpr std::size_t partial_moment_bytes = 2U * sizeof(double);
    return kHybridFftConvolutionOffset
        + hybrid_fft_convolution_bytes(step_count, path_chunk_size)
        + checked_hybrid_fft_product(
            hybrid_fft_partial_moment_count(path_count),
            partial_moment_bytes,
            "Volterra hybrid FFT partial-moment bytes overflow size_t."
        );
}

inline HybridFftWorkspacePlan plan_hybrid_fft_workspace(
    std::size_t maximum_step_count,
    std::size_t monte_carlo_paths_per_price,
    std::size_t path_chunk_size
) {
    validate_monte_carlo_path_count(monte_carlo_paths_per_price);
    if (maximum_step_count == 0U
        || maximum_step_count > kMaximumHybridFftStepCount) {
        throw std::invalid_argument(
            "Volterra hybrid FFT maximum_step_count must be in [1, 4096]."
        );
    }
    if (path_chunk_size == 0U
        || path_chunk_size > monte_carlo_paths_per_price
        || path_chunk_size % kHybridFftPathThreads != 0U) {
        throw std::invalid_argument(
            "Volterra hybrid FFT path_chunk_size must be a positive "
            "multiple of 256 not exceeding the path count."
        );
    }
    validate_grid_x_size((path_chunk_size + 1U) / 2U);
    const std::size_t convolution_bytes = hybrid_fft_convolution_bytes(
        maximum_step_count,
        path_chunk_size
    );
    const std::size_t partial_count = hybrid_fft_partial_moment_count(
        monte_carlo_paths_per_price
    );
    return {
        maximum_step_count,
        kMaximumHybridFftLength,
        monte_carlo_paths_per_price,
        path_chunk_size,
        kHybridFftSpectrumBytes,
        kHybridFftVolterraVarianceBytes,
        convolution_bytes,
        partial_count,
        required_hybrid_fft_workspace_bytes(
            maximum_step_count,
            monte_carlo_paths_per_price,
            path_chunk_size
        ),
    };
}

}  // namespace ai_factory::workbench::volterra
