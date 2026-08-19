// Rough-Bergomi European options with an in-block cuFFTDx convolution.
#include "model/equity/rough_bergomi/european_option_fft.cuh"

#include "common/check_cuda.cuh"
#include "common/cuda_kernel_diagnostics.cuh"
#include "common/philox.cuh"
#include "common/reductions.cuh"
#include "common/result_index.cuh"
#include "model/equity/rough_bergomi/dynamics.cu"

#include <cufftdx.hpp>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ai_factory::workbench::rough_bergomi {
namespace {

constexpr std::size_t kMaximumStepCount = 4096U;
constexpr std::size_t kMaximumFftLength = 8192U;
constexpr std::size_t kSpectrumBytes =
    kMaximumFftLength * sizeof(float2);
constexpr std::size_t kCorrectionBytes =
    kMaximumStepCount * sizeof(float);
constexpr std::size_t kPreparedRowBytes = 128U;
constexpr std::size_t kConvolutionOffset =
    kSpectrumBytes + kCorrectionBytes + kPreparedRowBytes;
constexpr unsigned int kPayoffThreads = 256U;

struct PartialMoments {
    double sum;
    double sumsq;
};

struct PreparedFftRow {
    RoughBergomiPreparedParameters model;
    philox::PhiloxKey key;
    float strike;
    float discount;
};
static_assert(sizeof(PreparedFftRow) <= kPreparedRowBytes);

std::size_t checked_product(
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

std::size_t convolution_bytes(
    std::size_t step_count,
    std::size_t path_chunk_size
) {
    return checked_product(
        checked_product(
            (path_chunk_size + 1U) / 2U,
            step_count,
            "rough-Bergomi FFT convolution element count overflows size_t."
        ),
        sizeof(float2),
        "rough-Bergomi FFT convolution bytes overflow size_t."
    );
}

std::size_t partial_moment_count(std::size_t path_count) {
    return (path_count + kPayoffThreads - 1U) / kPayoffThreads;
}

std::size_t required_workspace_bytes(
    std::size_t step_count,
    std::size_t path_count,
    std::size_t path_chunk_size
) {
    return kConvolutionOffset
        + convolution_bytes(step_count, path_chunk_size)
        + checked_product(
        partial_moment_count(path_count),
        sizeof(PartialMoments),
        "rough-Bergomi FFT partial-moment bytes overflow size_t."
    );
}

// Random access to the exact normal produced by UniformSequence and
// NormalPairCache at a scalar normal index. This lets FFT lanes generate one
// Brownian cell independently while preserving the canonical Philox mapping.
__device__ __forceinline__ float normal_at(
    philox::PhiloxKey key,
    std::uint64_t path,
    std::uint64_t normal_index
) {
    const std::uint64_t pair_index = normal_index >> 1U;
    const philox::RandomQuad uniforms = philox::uniform_quad(
        key, path, pair_index >> 1U
    );
    const bool upper_pair = (pair_index & 1U) != 0U;
    const philox::NormalPair normals = philox::box_muller(
        upper_pair ? uniforms.third : uniforms.first,
        upper_pair ? uniforms.fourth : uniforms.second
    );
    return (normal_index & 1U) == 0U ? normals.first : normals.second;
}

template<unsigned int Length, unsigned int ElementsPerThread,
         unsigned int FftsPerBlock>
struct FftTypes {
    using Base = decltype(
        cufftdx::Block()
        + cufftdx::Size<Length>()
        + cufftdx::Type<cufftdx::fft_type::c2c>()
        + cufftdx::Precision<float>()
        + cufftdx::SM<890>()
        + cufftdx::ElementsPerThread<ElementsPerThread>()
        + cufftdx::FFTsPerBlock<FftsPerBlock>()
    );
    using Forward = decltype(
        Base() + cufftdx::Direction<cufftdx::fft_direction::forward>()
    );
    using Inverse = decltype(
        Base() + cufftdx::Direction<cufftdx::fft_direction::inverse>()
    );
};

template<OptionSide Side>
__device__ __forceinline__ float payoff(
    float terminal_spot,
    float strike,
    float discount
) {
    if constexpr (Side == OptionSide::call) {
        return discount * fmaxf(terminal_spot - strike, 0.0f);
    } else {
        return discount * fmaxf(strike - terminal_spot, 0.0f);
    }
}

// Build the far-cell kernel and its spectrum once for the current result row.
template<unsigned int Length, class Forward>
__global__ void prepare_fft_row_kernel(
    const RoughBergomiModelParameters* __restrict__ models,
    const product::EuropeanOptionParameters* __restrict__ products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_index,
    std::size_t step_count,
    std::uint64_t base_seed,
    float2* __restrict__ kernel_spectrum,
    float* __restrict__ log_variance_corrections,
    PreparedFftRow* __restrict__ prepared_row
) {
    using Complex = typename Forward::value_type;
    Complex thread_data[Forward::storage_size];
    const ModelProductIndices indices = decode_model_product_result_index(
        result_index, product_count, cartesian_product
    );
    const RoughBergomiModelParameters model = models[indices.model_index];
    const float maturity = products[indices.product_index].maturity;
    const float dt = maturity / static_cast<float>(step_count);
    const float alpha = model.hurst_exponent - 0.5f;
    const float alpha_plus_one = model.hurst_exponent + 0.5f;
    const float dt_to_alpha = powf(dt, alpha);

    if (threadIdx.x == 0U && threadIdx.y == 0U) {
        *prepared_row = {
            prepare_model(model, maturity, step_count),
            philox::make_key(base_seed + result_index),
            products[indices.product_index].strike,
            expf(-model.risk_free_rate * maturity),
        };
    }

    if (Forward::working_group::is_thread_active()) {
        #pragma unroll
        for (unsigned int item = 0U; item < Forward::input_ept; ++item) {
            const unsigned int index =
                item * Forward::stride + threadIdx.x;
            float value = 0.0f;
            if (threadIdx.y == 0U && index + 1U < step_count) {
                const float lag = static_cast<float>(index + 2U);
                const float previous_lag = static_cast<float>(index + 1U);
                value = dt_to_alpha
                    * (powf(lag, alpha_plus_one)
                       - powf(previous_lag, alpha_plus_one))
                    / alpha_plus_one;
            }
            reinterpret_cast<float2*>(thread_data)[item] = {value, 0.0f};
            if (threadIdx.y == 0U && index < step_count) {
                const float time = static_cast<float>(index + 1U) * dt;
                log_variance_corrections[index] = logf(model.xi_0)
                    - 0.5f * model.eta * model.eta
                        * powf(time, 2.0f * model.hurst_exponent);
            }
        }
    }

    extern __shared__ __align__(16) unsigned char shared_storage[];
    Forward().execute(
        thread_data,
        reinterpret_cast<Complex*>(shared_storage)
    );
    if (threadIdx.y == 0U && Forward::working_group::is_thread_active()) {
        #pragma unroll
        for (unsigned int item = 0U; item < Forward::output_ept; ++item) {
            const unsigned int index =
                item * Forward::stride + threadIdx.x;
            if (index < Length) {
                kernel_spectrum[index] =
                    reinterpret_cast<float2*>(thread_data)[item];
            }
        }
    }
}

template<unsigned int Length, class Forward, class Inverse>
__global__ void rough_bergomi_fft_convolution_kernel(
    std::size_t global_path_offset,
    std::size_t chunk_path_count,
    std::size_t step_count,
    const PreparedFftRow* __restrict__ prepared_row,
    const float2* __restrict__ kernel_spectrum,
    float2* __restrict__ convolutions
) {
    using Complex = typename Forward::value_type;
    const unsigned int local_fft = threadIdx.y;
    const std::size_t local_pair =
        static_cast<std::size_t>(blockIdx.x) * Forward::ffts_per_block
        + local_fft;
    const std::size_t first_local_path = 2U * local_pair;
    const std::size_t first_global_path =
        global_path_offset + first_local_path;
    const PreparedFftRow row = *prepared_row;
    Complex thread_data[Forward::storage_size];

    if (Forward::working_group::is_thread_active()) {
        #pragma unroll
        for (unsigned int item = 0U; item < Forward::input_ept; ++item) {
            const unsigned int step =
                item * Forward::stride + threadIdx.x;
            float2 increment{0.0f, 0.0f};
            if (step < step_count && first_local_path < chunk_path_count) {
                increment.x = row.model.sqrt_time_step * normal_at(
                    row.key, first_global_path, 3ULL * step
                );
                if (first_local_path + 1U < chunk_path_count) {
                    increment.y = row.model.sqrt_time_step * normal_at(
                        row.key, first_global_path + 1U, 3ULL * step
                    );
                }
            }
            reinterpret_cast<float2*>(thread_data)[item] = increment;
        }
    }

    extern __shared__ __align__(16) unsigned char shared_storage[];
    Complex* const shared_values =
        reinterpret_cast<Complex*>(shared_storage);
    Forward().execute(thread_data, shared_values);
    constexpr float inverse_length = 1.0f / static_cast<float>(Length);
    if (Forward::working_group::is_thread_active()) {
        #pragma unroll
        for (unsigned int item = 0U; item < Forward::output_ept; ++item) {
            const unsigned int frequency =
                item * Forward::stride + threadIdx.x;
            if (frequency < Length) {
                const float2 value =
                    reinterpret_cast<float2*>(thread_data)[item];
                const float2 kernel = kernel_spectrum[frequency];
                reinterpret_cast<float2*>(thread_data)[item] = {
                    (value.x * kernel.x - value.y * kernel.y)
                        * inverse_length,
                    (value.x * kernel.y + value.y * kernel.x)
                        * inverse_length,
                };
            }
        }
    }
    Inverse().execute(thread_data, shared_values);

    if (Inverse::working_group::is_thread_active()
        && first_local_path < chunk_path_count) {
        #pragma unroll
        for (unsigned int item = 0U; item < Inverse::output_ept; ++item) {
            const unsigned int step =
                item * Inverse::stride + threadIdx.x;
            if (step < step_count) {
                convolutions[local_pair * step_count + step] =
                    reinterpret_cast<float2*>(thread_data)[item];
            }
        }
    }
}

template<OptionSide Side>
__global__ void rough_bergomi_fft_payoff_kernel(
    const PreparedFftRow* __restrict__ prepared_row,
    std::size_t global_path_offset,
    std::size_t chunk_path_count,
    std::size_t step_count,
    const float* __restrict__ log_variance_corrections,
    const float2* __restrict__ convolutions,
    PartialMoments* __restrict__ partial_moments
) {
    const PreparedFftRow row = *prepared_row;
    const std::size_t local_path =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t global_path = global_path_offset + local_path;
    double sum = 0.0;
    double sumsq = 0.0;
    if (local_path < chunk_path_count) {
        const std::size_t local_pair = local_path / 2U;
        const bool imaginary_lane = (local_path & 1U) != 0U;
        float log_spot = row.model.initial_log_spot;
        float variance = row.model.initial_variance;
        philox::UniformSequence uniforms(row.key, global_path);
        philox::NormalPairCache normal_cache;
        for (std::size_t step = 0U; step < step_count; ++step) {
            const float rough_normal =
                philox::next_normal(uniforms, normal_cache);
            const float singular_normal =
                philox::next_normal(uniforms, normal_cache);
            const float spot_normal =
                philox::next_normal(uniforms, normal_cache);
            const float brownian_increment =
                row.model.sqrt_time_step * rough_normal;
            const float spot_increment = fmaf(
                row.model.rho,
                brownian_increment,
                row.model.orthogonal_correlation
                    * row.model.sqrt_time_step * spot_normal
            );
            log_spot = fmaf(
                sqrtf(variance),
                spot_increment,
                log_spot + row.model.drift_time_step
                    - 0.5f * variance * row.model.time_step
            );
            float far_convolution = 0.0f;
            if (step != 0U) {
                const float2 far =
                    convolutions[local_pair * step_count + step - 1U];
                far_convolution = imaginary_lane ? far.y : far.x;
            }
            const float singular_integral = fmaf(
                row.model.singular_driver_loading,
                rough_normal,
                row.model.singular_independent_loading * singular_normal
            );
            variance = expf(fmaf(
                row.model.eta,
                row.model.sqrt_two_h
                    * (singular_integral + far_convolution),
                log_variance_corrections[step]
            ));
        }
        const float value = payoff<Side>(
            expf(log_spot), row.strike, row.discount
        );
        sum = static_cast<double>(value);
        sumsq = sum * sum;
    }

    const reductions::MomentSums total =
        reductions::reduce_block(sum, sumsq);
    if (threadIdx.x == 0U) {
        const std::size_t partial_index =
            global_path_offset / blockDim.x + blockIdx.x;
        partial_moments[partial_index] = {total.sum, total.sumsq};
    }
}

template<OptionSide Side>
__global__ void finalize_fft_price_kernel(
    const PartialMoments* __restrict__ partial_moments,
    std::size_t partial_count,
    std::size_t path_count,
    std::size_t result_index,
    float* __restrict__ prices,
    float* __restrict__ standard_errors
) {
    double sum = 0.0;
    double sumsq = 0.0;
    for (std::size_t partial = threadIdx.x;
         partial < partial_count;
         partial += blockDim.x) {
        sum += partial_moments[partial].sum;
        sumsq += partial_moments[partial].sumsq;
    }
    const reductions::MomentSums total =
        reductions::reduce_block(sum, sumsq);
    if (threadIdx.x == 0U) {
        double price = 0.0;
        double standard_error = 0.0;
        reductions::compute_statistics(
            total, path_count, price, standard_error
        );
        prices[result_index] = static_cast<float>(price);
        standard_errors[result_index] = static_cast<float>(standard_error);
    }
}

template<OptionSide Side, unsigned int Length, unsigned int ElementsPerThread,
         unsigned int FftsPerBlock>
void launch_fft_length(
    const RoughBergomiModelParameters* device_models,
    const product::EuropeanOptionParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_index,
    std::size_t path_count,
    std::size_t step_count,
    std::size_t path_chunk_size,
    float2* kernel_spectrum,
    float* corrections,
    PreparedFftRow* prepared_row,
    float2* convolutions,
    PartialMoments* partial_moments,
    std::uint64_t base_seed,
    float* device_prices,
    float* device_standard_errors
) {
    using Types = FftTypes<Length, ElementsPerThread, FftsPerBlock>;
    using Forward = typename Types::Forward;
    using Inverse = typename Types::Inverse;
    static_assert(!Forward::requires_workspace);
    static_assert(!Inverse::requires_workspace);
    constexpr std::size_t fft_shared_bytes =
        Forward::shared_memory_size > Inverse::shared_memory_size
            ? Forward::shared_memory_size
            : Inverse::shared_memory_size;
    constexpr std::size_t prepare_shared_bytes = Forward::shared_memory_size;

    static const int active_blocks = [] {
        check_cuda(
            cudaFuncSetAttribute(
                prepare_fft_row_kernel<Length, Forward>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(prepare_shared_bytes)
            ),
            "rough-Bergomi cuFFTDx preparation shared-memory opt-in"
        );
        check_cuda(
            cudaFuncSetAttribute(
                rough_bergomi_fft_convolution_kernel<
                    Length, Forward, Inverse
                >,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(fft_shared_bytes)
            ),
            "rough-Bergomi cuFFTDx convolution shared-memory opt-in"
        );
        int resident_blocks = 0;
        check_cuda(
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &resident_blocks,
                rough_bergomi_fft_convolution_kernel<
                    Length, Forward, Inverse
                >,
                static_cast<int>(
                    Forward::block_dim.x * Forward::block_dim.y
                ),
                fft_shared_bytes
            ),
            "rough-Bergomi cuFFTDx convolution occupancy check"
        );
        return resident_blocks;
    }();
    if (active_blocks == 0) {
        throw std::invalid_argument(
            "rough-Bergomi cuFFTDx convolution kernel cannot reside on an SM."
        );
    }

    prepare_fft_row_kernel<Length, Forward><<<
        1U, Forward::block_dim, prepare_shared_bytes
    >>>(
        device_models,
        device_products,
        product_count,
        cartesian_product,
        result_index,
        step_count,
        base_seed,
        kernel_spectrum,
        corrections,
        prepared_row
    );
    check_cuda(
        cudaGetLastError(), "rough-Bergomi cuFFTDx row preparation"
    );

    constexpr std::size_t payoff_shared_bytes =
        2U * (kPayoffThreads / 32U) * sizeof(double);
    for (std::size_t path_offset = 0U;
         path_offset < path_count;
         path_offset += path_chunk_size) {
        const std::size_t chunk_path_count = std::min(
            path_chunk_size, path_count - path_offset
        );
        const std::size_t chunk_pair_count =
            (chunk_path_count + 1U) / 2U;
        const std::size_t convolution_block_count =
            (chunk_pair_count + FftsPerBlock - 1U) / FftsPerBlock;
        report_cuda_kernel_launch_if_enabled(
            "rough_bergomi.european_option_fft_convolution",
            option_side_name(Side),
            rough_bergomi_fft_convolution_kernel<
                Length, Forward, Inverse
            >,
            dim3(static_cast<unsigned int>(convolution_block_count)),
            Forward::block_dim,
            fft_shared_bytes
        );
        rough_bergomi_fft_convolution_kernel<
            Length, Forward, Inverse
        ><<<
            static_cast<unsigned int>(convolution_block_count),
            Forward::block_dim,
            fft_shared_bytes
        >>>(
            path_offset,
            chunk_path_count,
            step_count,
            prepared_row,
            kernel_spectrum,
            convolutions
        );
        check_cuda(
            cudaGetLastError(), "rough-Bergomi cuFFTDx convolution kernel"
        );

        const std::size_t payoff_block_count =
            (chunk_path_count + kPayoffThreads - 1U) / kPayoffThreads;
        rough_bergomi_fft_payoff_kernel<Side><<<
            static_cast<unsigned int>(payoff_block_count),
            kPayoffThreads,
            payoff_shared_bytes
        >>>(
            prepared_row,
            path_offset,
            chunk_path_count,
            step_count,
            corrections,
            convolutions,
            partial_moments
        );
        check_cuda(cudaGetLastError(), "rough-Bergomi cuFFTDx payoff kernel");
    }

    constexpr unsigned int finalize_threads = 256U;
    constexpr std::size_t finalize_shared_bytes =
        2U * (finalize_threads / 32U) * sizeof(double);
    finalize_fft_price_kernel<Side><<<
        1U, finalize_threads, finalize_shared_bytes
    >>>(
        partial_moments,
        partial_moment_count(path_count),
        path_count,
        result_index,
        device_prices,
        device_standard_errors
    );
    check_cuda(cudaGetLastError(), "rough-Bergomi cuFFTDx finalization");
}

void validate_fft_launch(
    const RoughBergomiModelParameters* device_models,
    std::size_t model_count,
    const product::EuropeanOptionParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_index,
    std::size_t path_count,
    float target_dt,
    std::size_t step_count,
    std::size_t path_chunk_size,
    const void* device_workspace,
    std::size_t workspace_bytes,
    std::uint64_t base_seed,
    const float* device_prices,
    const float* device_standard_errors
) {
    validate_device_pointer(device_models, "device_models");
    validate_device_pointer(device_products, "device_products");
    validate_device_pointer(device_workspace, "device_workspace");
    validate_device_pointer(device_prices, "device_prices");
    validate_device_pointer(device_standard_errors, "device_standard_errors");
    validate_model_product_construction(
        model_count, product_count, cartesian_product, result_count
    );
    if (result_index >= result_count) {
        throw std::invalid_argument(
            "rough-Bergomi FFT result_index exceeds the result array."
        );
    }
    validate_monte_carlo_parameters(path_count, target_dt);
    validate_row_seed_range(result_count, base_seed);
    if (step_count == 0U || step_count > kMaximumStepCount) {
        throw std::invalid_argument(
            "rough-Bergomi FFT step_count must be in [1, 4096]."
        );
    }
    if (path_chunk_size == 0U
        || path_chunk_size > path_count
        || path_chunk_size % kPayoffThreads != 0U) {
        throw std::invalid_argument(
            "rough-Bergomi FFT path_chunk_size must be a positive multiple "
            "of 256 not exceeding path_count."
        );
    }
    validate_grid_x_size((path_chunk_size + 1U) / 2U);
    if (workspace_bytes < required_workspace_bytes(
            step_count, path_count, path_chunk_size
        )) {
        throw std::invalid_argument(
            "rough-Bergomi FFT workspace is smaller than its launch plan."
        );
    }
}

}  // namespace

RoughBergomiFftWorkspacePlan plan_european_option_fft_workspace(
    std::size_t maximum_step_count,
    std::size_t monte_carlo_paths_per_price,
    std::size_t path_chunk_size
) {
    validate_monte_carlo_path_count(monte_carlo_paths_per_price);
    if (maximum_step_count == 0U
        || maximum_step_count > kMaximumStepCount) {
        throw std::invalid_argument(
            "rough-Bergomi FFT maximum_step_count must be in [1, 4096]."
        );
    }
    if (path_chunk_size == 0U
        || path_chunk_size > monte_carlo_paths_per_price
        || path_chunk_size % kPayoffThreads != 0U) {
        throw std::invalid_argument(
            "rough-Bergomi FFT path_chunk_size must be a positive multiple "
            "of 256 not exceeding the path count."
        );
    }
    validate_grid_x_size((path_chunk_size + 1U) / 2U);
    const std::size_t planned_convolution_bytes = convolution_bytes(
        maximum_step_count, path_chunk_size
    );
    const std::size_t planned_partial_count = partial_moment_count(
        monte_carlo_paths_per_price
    );
    return {
        maximum_step_count,
        kMaximumFftLength,
        monte_carlo_paths_per_price,
        path_chunk_size,
        planned_convolution_bytes,
        planned_partial_count,
        required_workspace_bytes(
            maximum_step_count,
            monte_carlo_paths_per_price,
            path_chunk_size
        ),
    };
}

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
) {
    validate_fft_launch(
        device_models,
        model_count,
        device_products,
        product_count,
        cartesian_product,
        result_count,
        result_index,
        monte_carlo_paths_per_price,
        target_dt,
        step_count,
        path_chunk_size,
        device_workspace,
        workspace_bytes,
        base_seed,
        device_prices,
        device_standard_errors
    );

    auto* const workspace = static_cast<unsigned char*>(device_workspace);
    auto* const spectrum = reinterpret_cast<float2*>(workspace);
    auto* const corrections = reinterpret_cast<float*>(
        workspace + kSpectrumBytes
    );
    auto* const prepared_row = reinterpret_cast<PreparedFftRow*>(
        workspace + kSpectrumBytes + kCorrectionBytes
    );
    auto* const convolutions = reinterpret_cast<float2*>(
        workspace + kConvolutionOffset
    );
    auto* const partials = reinterpret_cast<PartialMoments*>(
        workspace + kConvolutionOffset
            + convolution_bytes(step_count, path_chunk_size)
    );

    if (step_count <= 8U) {
        launch_fft_length<Side, 16U, 8U, 16U>(
            device_models, device_products, product_count, cartesian_product,
            result_index, monte_carlo_paths_per_price, step_count,
            path_chunk_size, spectrum, corrections, prepared_row,
            convolutions, partials, base_seed,
            device_prices, device_standard_errors
        );
    } else if (step_count <= 32U) {
        launch_fft_length<Side, 64U, 8U, 8U>(
            device_models, device_products, product_count, cartesian_product,
            result_index, monte_carlo_paths_per_price, step_count,
            path_chunk_size, spectrum, corrections, prepared_row,
            convolutions, partials, base_seed,
            device_prices, device_standard_errors
        );
    } else if (step_count <= 64U) {
        launch_fft_length<Side, 128U, 8U, 8U>(
            device_models, device_products, product_count, cartesian_product,
            result_index, monte_carlo_paths_per_price, step_count,
            path_chunk_size, spectrum, corrections, prepared_row,
            convolutions, partials, base_seed,
            device_prices, device_standard_errors
        );
    } else if (step_count <= 128U) {
        launch_fft_length<Side, 256U, 16U, 8U>(
            device_models, device_products, product_count, cartesian_product,
            result_index, monte_carlo_paths_per_price, step_count,
            path_chunk_size, spectrum, corrections, prepared_row,
            convolutions, partials, base_seed,
            device_prices, device_standard_errors
        );
    } else if (step_count <= 256U) {
        launch_fft_length<Side, 512U, 8U, 2U>(
            device_models, device_products, product_count, cartesian_product,
            result_index, monte_carlo_paths_per_price, step_count,
            path_chunk_size, spectrum, corrections, prepared_row,
            convolutions, partials, base_seed,
            device_prices, device_standard_errors
        );
    } else if (step_count <= 512U) {
        launch_fft_length<Side, 1024U, 16U, 1U>(
            device_models, device_products, product_count, cartesian_product,
            result_index, monte_carlo_paths_per_price, step_count,
            path_chunk_size, spectrum, corrections, prepared_row,
            convolutions, partials, base_seed,
            device_prices, device_standard_errors
        );
    } else if (step_count <= 1024U) {
        launch_fft_length<Side, 2048U, 16U, 1U>(
            device_models, device_products, product_count, cartesian_product,
            result_index, monte_carlo_paths_per_price, step_count,
            path_chunk_size, spectrum, corrections, prepared_row,
            convolutions, partials, base_seed,
            device_prices, device_standard_errors
        );
    } else if (step_count <= 2048U) {
        launch_fft_length<Side, 4096U, 16U, 1U>(
            device_models, device_products, product_count, cartesian_product,
            result_index, monte_carlo_paths_per_price, step_count,
            path_chunk_size, spectrum, corrections, prepared_row,
            convolutions, partials, base_seed,
            device_prices, device_standard_errors
        );
    } else {
        launch_fft_length<Side, 8192U, 32U, 1U>(
            device_models, device_products, product_count, cartesian_product,
            result_index, monte_carlo_paths_per_price, step_count,
            path_chunk_size, spectrum, corrections, prepared_row,
            convolutions, partials, base_seed,
            device_prices, device_standard_errors
        );
    }
}

using FftLaunchSignature = decltype(
    launch_rough_bergomi_european_option_fft_cuda<OptionSide::call>
);
template FftLaunchSignature
    launch_rough_bergomi_european_option_fft_cuda<OptionSide::call>;
template FftLaunchSignature
    launch_rough_bergomi_european_option_fft_cuda<OptionSide::put>;

}  // namespace ai_factory::workbench::rough_bergomi
