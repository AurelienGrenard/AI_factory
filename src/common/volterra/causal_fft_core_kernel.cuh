// Product-independent causal Volterra history and FFT operations.
#pragma once

#include "common/volterra/block_fft_convolution.cuh"
#include "common/volterra/hybrid_fft.cuh"
#include "common/volterra/hybrid_fft_tuning.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ai_factory::workbench::volterra::causal_fft {

constexpr unsigned int kPathGroup = 8U;

__host__ __device__ inline std::size_t history_index(
    std::size_t path, std::size_t step, std::size_t steps
) {
    const std::size_t group = path / kPathGroup;
    const std::size_t lane = path % kPathGroup;
    return (group * steps + step) * kPathGroup + lane;
}

template<typename Row>
__global__ void prepare_weights_kernel(
    const Row* rows,
    std::size_t row_count,
    std::uint32_t steps,
    float* weights
) {
    const std::size_t row_index = blockIdx.y;
    const std::uint32_t lag = blockIdx.x * blockDim.x + threadIdx.x + 1U;
    if (row_index >= row_count || lag > steps) return;
    using Kernel = typename Row::Kernel;
    weights[row_index * steps + lag - 1U] =
        Kernel::cell_average_weight(rows[row_index].kernel, lag);
}

template<unsigned int Length, unsigned int ElementsPerThread,
         unsigned int FftsPerBlock, typename Row>
__global__ void prepare_spectrum_kernel(
    const Row* rows,
    std::size_t row_count,
    std::uint32_t steps,
    std::uint32_t scale,
    std::size_t offset,
    std::size_t spectrum_stride,
    const float* weights,
    float2* spectra
) {
    using Types = hybrid_fft::FftTypes<
        Length, ElementsPerThread, FftsPerBlock
    >;
    using Forward = typename Types::Forward;
    using Complex = typename Forward::value_type;
    const std::size_t row_index = blockIdx.x * FftsPerBlock + threadIdx.y;
    const bool active = row_index < row_count;
    Complex values[Forward::storage_size];
    if (Forward::working_group::is_thread_active()) {
        #pragma unroll
        for (unsigned int item = 0U; item < Forward::input_ept; ++item) {
            const unsigned int index =
                item * Forward::stride + threadIdx.x;
            const float weight = active && index < 2U * scale - 1U
                && index < steps
                ? weights[row_index * steps + index] : 0.0f;
            reinterpret_cast<float2*>(values)[item] = {weight, 0.0f};
        }
    }
    extern __shared__ __align__(16) unsigned char scratch[];
    Forward().execute(values, reinterpret_cast<Complex*>(scratch));
    if (active && Forward::working_group::is_thread_active()) {
        #pragma unroll
        for (unsigned int item = 0U; item < Forward::output_ept; ++item) {
            const unsigned int index =
                item * Forward::stride + threadIdx.x;
            if (index < Length) {
                spectra[row_index * spectrum_stride + offset + index] =
                    reinterpret_cast<float2*>(values)[item];
            }
        }
    }
}

template<unsigned int Length, unsigned int ElementsPerThread,
         unsigned int FftsPerBlock>
__global__ void update_history_kernel(
    const float2* spectra,
    std::size_t spectrum_stride,
    std::size_t spectrum_offset,
    float* history,
    std::size_t row_count,
    std::size_t chunk_capacity,
    std::size_t path_count,
    std::uint32_t steps,
    std::uint32_t boundary,
    std::uint32_t scale
) {
    using Types = hybrid_fft::FftTypes<
        Length, ElementsPerThread, FftsPerBlock
    >;
    using Forward = typename Types::Forward;
    using Inverse = typename Types::Inverse;
    using Complex = typename Forward::value_type;
    const std::size_t row_index = blockIdx.y;
    if (row_index >= row_count) return;
    const std::size_t pair = blockIdx.x * FftsPerBlock + threadIdx.y;
    const std::size_t path0 = 2U * pair;
    float* row_history = history + row_index * chunk_capacity * steps;
    const float2* row_spectrum = spectra
        + row_index * spectrum_stride + spectrum_offset;
    struct Loader {
        const float* history;
        std::size_t path0;
        std::size_t paths;
        std::uint32_t first;
        std::uint32_t scale;
        std::uint32_t steps;
        __device__ __forceinline__ float2 operator()(unsigned int index) const {
            if (index >= scale || path0 >= paths) return {0.0f, 0.0f};
            const std::uint32_t step = first + index;
            const float first_value = history[history_index(
                path0, step, steps
            )];
            const float second_value = path0 + 1U < paths
                ? history[history_index(path0 + 1U, step, steps)]
                : 0.0f;
            return {first_value, second_value};
        }
    };
    struct Store {
        float* history;
        std::size_t path0;
        std::size_t paths;
        std::uint32_t boundary;
        std::uint32_t scale;
        std::uint32_t steps;
        __device__ __forceinline__ void operator()(
            unsigned int index, float2 value
        ) const {
            if (index + 1U < scale || index >= 2U * scale - 1U
                || path0 >= paths) return;
            const std::uint32_t step = boundary + index + 1U - scale;
            if (step >= steps) return;
            history[history_index(path0, step, steps)] += value.x;
            if (path0 + 1U < paths) {
                history[history_index(path0 + 1U, step, steps)]
                    += value.y;
            }
        }
    };
    extern __shared__ __align__(16) unsigned char scratch[];
    execute_padded_linear_convolution<Length, Forward, Inverse>(
        row_spectrum,
        Loader{row_history, path0, path_count, boundary - scale,
               scale, steps},
        Store{row_history, path0, path_count, boundary, scale,
              steps},
        reinterpret_cast<Complex*>(scratch)
    );
}

template<typename Callback>
void dispatch_scale(std::uint32_t scale, Callback&& callback) {
    if (scale < 32U || scale > 4096U || (scale & (scale - 1U)) != 0U)
        throw std::invalid_argument("unsupported causal FFT scale");
    // The existing Gaussian FFT tuning selects a transform of size 2*scale.
    tuning::dispatch_hybrid_fft_specialization<8192U>(
        scale, [&]<typename Specialization>() {
            callback.template operator()<
                Specialization::kLength,
                Specialization::kPricingElementsPerThread,
                Specialization::kPricingFftsPerBlock
            >();
        }
    );
}

}  // namespace ai_factory::workbench::volterra::causal_fft
