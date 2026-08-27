// Product-generic hybrid/FFT Monte Carlo engine for Gaussian Volterra paths.
#pragma once

#include "common/check_cuda.cuh"
#include "common/equity/path_product_policy.cuh"
#include "common/cuda_kernel_diagnostics.cuh"
#include "common/philox.cuh"
#include "common/reductions.cuh"
#include "common/result_index.cuh"
#include "common/volterra/block_fft_convolution.cuh"
#include "common/volterra/hybrid_fft.cuh"
#include "common/volterra/hybrid_fft_workspace.cuh"
#include "common/volterra/hybrid_schedule.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace ai_factory::workbench::volterra::hybrid_fft {

constexpr unsigned int kFinalizationThreads = 256U;
constexpr unsigned int kPathThreads = 256U;
#ifndef AI_FACTORY_VOLTERRA_DIRECT_MAX_STEP_COUNT
#define AI_FACTORY_VOLTERRA_DIRECT_MAX_STEP_COUNT 0
#endif
inline constexpr std::uint32_t kDirectMaximumStepCount =
    AI_FACTORY_VOLTERRA_DIRECT_MAX_STEP_COUNT;

struct PartialMoments {
    double sum;
    double sumsq;
};

template<typename DriverPolicy, typename ModelPathPolicy,
         typename ProductPolicy, typename SchedulePolicy>
struct PreparedRow {
    using Driver = DriverPolicy;
    using Path = ModelPathPolicy;
    using Product = ProductPolicy;
    using Schedule = SchedulePolicy;

    typename Driver::PreparedDriver driver;
    typename Path::PreparedModel model;
    typename Product::PreparedProduct product;
    typename Schedule::PreparedSchedule schedule;
    philox::PhiloxKey key;
};

#if AI_FACTORY_VOLTERRA_DIRECT_MAX_STEP_COUNT > 0
template<typename DriverPolicy, typename ModelPathPolicy,
         typename ProductPolicy, typename SchedulePolicy>
__global__ void prepare_direct_row_kernel(
    const typename ModelPathPolicy::Parameters* __restrict__ models,
    const typename ProductPolicy::ProductParameters* __restrict__ products,
    std::size_t product_count,
    PriceConstruction construction,
    std::size_t result_index,
    std::uint32_t step_count,
    HybridTimeConfiguration time_configuration,
    std::uint64_t base_seed,
    float* __restrict__ driver_variances,
    PreparedRow<
        DriverPolicy,
        ModelPathPolicy,
        ProductPolicy,
        SchedulePolicy
    >* __restrict__ prepared_row
) {
    using Row = PreparedRow<
        DriverPolicy,
        ModelPathPolicy,
        ProductPolicy,
        SchedulePolicy
    >;
    const ModelProductIndices indices = decode_model_product_result_index_32(
        static_cast<std::uint32_t>(result_index),
        static_cast<std::uint32_t>(product_count),
        construction
    );
    const typename ModelPathPolicy::Parameters parameters =
        models[indices.model_index];
    const typename ProductPolicy::ProductParameters product_parameters =
        products[indices.product_index];
    const typename SchedulePolicy::PreparedSchedule schedule =
        SchedulePolicy::prepare(
            ProductPolicy::calendar(product_parameters),
            time_configuration,
            step_count
        );
    __shared__ typename DriverPolicy::PreparedDriver shared_driver;
    if (threadIdx.x == 0U) {
        shared_driver = DriverPolicy::prepare(
            ModelPathPolicy::driver_parameters(parameters),
            schedule.time_step
        );
    }
    __syncthreads();
    const typename DriverPolicy::PreparedDriver driver = shared_driver;
    if (threadIdx.x == 0U) {
        *prepared_row = Row{
            driver,
            ModelPathPolicy::prepare_model(parameters, schedule.time_step),
            ProductPolicy::prepare_product(
                parameters,
                product_parameters,
                equity::ProductPreparationContext{
                    time_configuration.day_fraction,
                    schedule.maturity_years,
                }
            ),
            schedule,
            philox::make_key(base_seed + result_index),
        };
    }
    for (std::uint32_t step = threadIdx.x;
         step < step_count;
         step += blockDim.x) {
        const float time = static_cast<float>(step + 1U)
            * schedule.time_step;
        if constexpr (ModelPathPolicy::kUsesDriverVariance) {
            driver_variances[step] = DriverPolicy::variance(driver, time);
        } else {
            driver_variances[step] = 0.0f;
        }
    }
}
#endif

template<typename DriverPolicy, typename ModelPathPolicy,
         typename ProductPolicy, typename SchedulePolicy,
         unsigned int Length, class KernelForward>
__global__ void prepare_row_kernel(
    const typename ModelPathPolicy::Parameters* __restrict__ models,
    const typename ProductPolicy::ProductParameters* __restrict__ products,
    std::size_t product_count,
    PriceConstruction construction,
    std::size_t result_index,
    std::uint32_t step_count,
    HybridTimeConfiguration time_configuration,
    std::uint64_t base_seed,
    float2* __restrict__ kernel_spectrum,
    float* __restrict__ driver_variances,
    PreparedRow<
        DriverPolicy,
        ModelPathPolicy,
        ProductPolicy,
        SchedulePolicy
    >* __restrict__ prepared_row
) {
    using Row = PreparedRow<
        DriverPolicy,
        ModelPathPolicy,
        ProductPolicy,
        SchedulePolicy
    >;
    using Complex = typename KernelForward::value_type;
    Complex thread_data[KernelForward::storage_size];

    const ModelProductIndices indices = decode_model_product_result_index_32(
        static_cast<std::uint32_t>(result_index),
        static_cast<std::uint32_t>(product_count),
        construction
    );
    const typename ModelPathPolicy::Parameters parameters =
        models[indices.model_index];
    const typename ProductPolicy::ProductParameters product_parameters =
        products[indices.product_index];
    const typename SchedulePolicy::Calendar calendar =
        ProductPolicy::calendar(product_parameters);
    const typename SchedulePolicy::PreparedSchedule schedule =
        SchedulePolicy::prepare(
            calendar,
            time_configuration,
            step_count
        );
    __shared__ typename DriverPolicy::PreparedDriver shared_driver;
    if (threadIdx.x == 0U && threadIdx.y == 0U) {
        shared_driver = DriverPolicy::prepare(
            ModelPathPolicy::driver_parameters(parameters),
            schedule.time_step
        );
    }
    __syncthreads();
    const typename DriverPolicy::PreparedDriver driver = shared_driver;

    if (threadIdx.x == 0U && threadIdx.y == 0U) {
        *prepared_row = Row{
            driver,
            ModelPathPolicy::prepare_model(
                parameters,
                schedule.time_step
            ),
            ProductPolicy::prepare_product(
                parameters,
                product_parameters,
                equity::ProductPreparationContext{
                    time_configuration.day_fraction,
                    schedule.maturity_years,
                }
            ),
            schedule,
            philox::make_key(base_seed + result_index),
        };
    }

    if (KernelForward::working_group::is_thread_active()) {
        #pragma unroll
        for (unsigned int item = 0U;
             item < KernelForward::input_ept;
             ++item) {
            const unsigned int index =
                item * KernelForward::stride + threadIdx.x;
            float weight = 0.0f;
            if (index + 1U < step_count) {
                weight = DriverPolicy::far_cell_weight(
                    driver,
                    index + 2U
                );
            }
            reinterpret_cast<float2*>(thread_data)[item] = {weight, 0.0f};
            if (index < step_count) {
                const float time =
                    static_cast<float>(index + 1U) * schedule.time_step;
                if constexpr (ModelPathPolicy::kUsesDriverVariance) {
                    driver_variances[index] =
                        DriverPolicy::variance(driver, time);
                } else {
                    driver_variances[index] = 0.0f;
                }
            }
        }
    }

    extern __shared__ __align__(16) unsigned char shared_storage[];
    KernelForward().execute(
        thread_data,
        reinterpret_cast<Complex*>(shared_storage)
    );
    if (KernelForward::working_group::is_thread_active()) {
        #pragma unroll
        for (unsigned int item = 0U;
             item < KernelForward::output_ept;
             ++item) {
            const unsigned int index =
                item * KernelForward::stride + threadIdx.x;
            if (index < Length) {
                kernel_spectrum[index] =
                    reinterpret_cast<float2*>(thread_data)[item];
            }
        }
    }
}

template<typename DriverPolicy, typename ModelPathPolicy,
         typename ProductPolicy, typename SchedulePolicy,
         unsigned int Length, class Forward, class Inverse>
__global__ void convolve_paths_kernel(
    std::size_t global_path_offset,
    std::size_t chunk_path_count,
    const PreparedRow<
        DriverPolicy,
        ModelPathPolicy,
        ProductPolicy,
        SchedulePolicy
    >* __restrict__ prepared_row,
    const float2* __restrict__ kernel_spectrum,
    float2* __restrict__ convolutions
) {
    using Row = PreparedRow<
        DriverPolicy,
        ModelPathPolicy,
        ProductPolicy,
        SchedulePolicy
    >;
    using Complex = typename Forward::value_type;
    constexpr unsigned int ffts_per_block = Forward::ffts_per_block;
    const Row row = *prepared_row;
    extern __shared__ __align__(16) unsigned char shared_storage[];
    const unsigned int local_fft = threadIdx.y;
    const std::size_t local_pair =
        static_cast<std::size_t>(blockIdx.x) * ffts_per_block + local_fft;
    const std::size_t first_local_path = 2U * local_pair;
    const std::size_t first_global_path =
        global_path_offset + first_local_path;

    struct Loader {
        const Row& row;
        std::size_t first_local_path;
        std::size_t first_global_path;
        std::size_t chunk_path_count;

        __device__ __forceinline__ float2 operator()(
            unsigned int step
        ) const {
            float2 increment{0.0f, 0.0f};
            if (step < row.schedule.step_count
                && first_local_path < chunk_path_count) {
                increment.x = row.driver.sqrt_time_step * normal_at(
                    row.key,
                    first_global_path,
                    3ULL * step
                );
                if (first_local_path + 1U < chunk_path_count) {
                    increment.y = row.driver.sqrt_time_step * normal_at(
                        row.key,
                        first_global_path + 1U,
                        3ULL * step
                    );
                }
            }
            return increment;
        }
    };
    struct Store {
        float2* outputs;
        std::size_t local_pair;
        std::uint32_t step_count;

        __device__ __forceinline__ void operator()(
            unsigned int index,
            float2 value
        ) const {
            if (index < step_count) {
                outputs[local_pair * step_count + index] = value;
            }
        }
    };

    volterra::execute_padded_linear_convolution<Length, Forward, Inverse>(
        kernel_spectrum,
        Loader{
            row,
            first_local_path,
            first_global_path,
            chunk_path_count,
        },
        Store{convolutions, local_pair, row.schedule.step_count},
        reinterpret_cast<Complex*>(shared_storage)
    );
}

template<typename DriverPolicy, typename ModelPathPolicy,
         typename ProductPolicy, typename SchedulePolicy>
__global__ void evaluate_paths_kernel(
    std::size_t global_path_offset,
    std::size_t chunk_path_count,
    const PreparedRow<
        DriverPolicy,
        ModelPathPolicy,
        ProductPolicy,
        SchedulePolicy
    >* __restrict__ prepared_row,
    const float* __restrict__ driver_variances,
    const float2* __restrict__ convolutions,
    PartialMoments* __restrict__ partial_moments
) {
    using Row = PreparedRow<
        DriverPolicy,
        ModelPathPolicy,
        ProductPolicy,
        SchedulePolicy
    >;
    __shared__ Row row;
    if (threadIdx.x == 0U) row = *prepared_row;
    __syncthreads();

    const std::size_t local_path =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t path = global_path_offset + local_path;
    double sum = 0.0;
    double sumsq = 0.0;
    if (local_path < chunk_path_count) {
        const std::size_t local_pair = local_path / 2U;
        const bool imaginary_lane = (local_path & 1U) != 0U;
        typename ModelPathPolicy::State state =
            ModelPathPolicy::initial_state(row.model);
        auto handler = ProductPolicy::make_handler(row.product);
        equity::PathProductObservationAdapter<
            ModelPathPolicy,
            decltype(handler),
            ProductPolicy::kObservationCoordinate
        > observation_adapter{handler};
        typename SchedulePolicy::Cursor cursor =
            SchedulePolicy::make_cursor(row.schedule);
        bool keep_running = SchedulePolicy::on_initial_state(
            row.schedule,
            cursor,
            state,
            observation_adapter
        );
        philox::UniformSequence uniforms(row.key, path);
        philox::NormalPairCache normal_cache;
        for (std::uint32_t step = 0U;
             keep_running && step < row.schedule.step_count;
             ++step) {
            const float rough_normal =
                philox::next_normal(uniforms, normal_cache);
            const float singular_normal =
                philox::next_normal(uniforms, normal_cache);
            const float spot_normal =
                philox::next_normal(uniforms, normal_cache);
            float far_convolution = 0.0f;
            if (step > 0U) {
                const float2 packed = convolutions[
                    local_pair * row.schedule.step_count + step - 1U
                ];
                far_convolution = imaginary_lane ? packed.y : packed.x;
            }
            const float driver_value = DriverPolicy::value(
                row.driver,
                far_convolution,
                rough_normal,
                singular_normal
            );
            ModelPathPolicy::advance(
                row.model,
                driver_value,
                driver_variances[step],
                rough_normal,
                spot_normal,
                state
            );
            keep_running = SchedulePolicy::on_step(
                row.schedule,
                cursor,
                step,
                state,
                observation_adapter
            );
        }
        const float payoff = ProductPolicy::template finalize<
            ModelPathPolicy
        >(row.product, state, handler);
        sum = static_cast<double>(payoff);
        sumsq = sum * sum;
    }

    const reductions::MomentSums total = reductions::reduce_block(
        sum,
        sumsq
    );
    if (threadIdx.x == 0U) {
        partial_moments[
            global_path_offset / kPathThreads + blockIdx.x
        ] = {total.sum, total.sumsq};
    }
}

#if AI_FACTORY_VOLTERRA_DIRECT_MAX_STEP_COUNT > 0
template<typename DriverPolicy, typename ModelPathPolicy,
         typename ProductPolicy, typename SchedulePolicy>
__global__ void evaluate_direct_paths_kernel(
    std::size_t global_path_offset,
    std::size_t chunk_path_count,
    const PreparedRow<
        DriverPolicy,
        ModelPathPolicy,
        ProductPolicy,
        SchedulePolicy
    >* __restrict__ prepared_row,
    const float* __restrict__ driver_variances,
    PartialMoments* __restrict__ partial_moments
) {
    using Row = PreparedRow<
        DriverPolicy,
        ModelPathPolicy,
        ProductPolicy,
        SchedulePolicy
    >;
    __shared__ Row row;
    if (threadIdx.x == 0U) row = *prepared_row;
    __syncthreads();

    const std::size_t local_path =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t path = global_path_offset + local_path;
    double sum = 0.0;
    double sumsq = 0.0;
    if (local_path < chunk_path_count) {
        typename ModelPathPolicy::State state =
            ModelPathPolicy::initial_state(row.model);
        auto handler = ProductPolicy::make_handler(row.product);
        equity::PathProductObservationAdapter<
            ModelPathPolicy,
            decltype(handler),
            ProductPolicy::kObservationCoordinate
        > observation_adapter{handler};
        typename SchedulePolicy::Cursor cursor =
            SchedulePolicy::make_cursor(row.schedule);
        bool keep_running = SchedulePolicy::on_initial_state(
            row.schedule,
            cursor,
            state,
            observation_adapter
        );
        philox::UniformSequence uniforms(row.key, path);
        philox::NormalPairCache normal_cache;
        for (std::uint32_t step = 0U;
             keep_running && step < row.schedule.step_count;
             ++step) {
            const float rough_normal =
                philox::next_normal(uniforms, normal_cache);
            const float singular_normal =
                philox::next_normal(uniforms, normal_cache);
            const float spot_normal =
                philox::next_normal(uniforms, normal_cache);
            float far_convolution = 0.0f;
            for (std::uint32_t prior_step = 0U;
                 prior_step < step;
                 ++prior_step) {
                const float increment = row.driver.sqrt_time_step * normal_at(
                    row.key,
                    path,
                    3ULL * prior_step
                );
                far_convolution += increment * DriverPolicy::far_cell_weight(
                    row.driver,
                    step - prior_step + 1U
                );
            }
            const float driver_value = DriverPolicy::value(
                row.driver,
                far_convolution,
                rough_normal,
                singular_normal
            );
            ModelPathPolicy::advance(
                row.model,
                driver_value,
                driver_variances[step],
                rough_normal,
                spot_normal,
                state
            );
            keep_running = SchedulePolicy::on_step(
                row.schedule,
                cursor,
                step,
                state,
                observation_adapter
            );
        }
        const float payoff = ProductPolicy::template finalize<ModelPathPolicy>(
            row.product,
            state,
            handler
        );
        sum = static_cast<double>(payoff);
        sumsq = sum * sum;
    }

    const reductions::MomentSums total = reductions::reduce_block(sum, sumsq);
    if (threadIdx.x == 0U) {
        partial_moments[
            global_path_offset / kPathThreads + blockIdx.x
        ] = {total.sum, total.sumsq};
    }
}
#endif

static __global__ void finalize_price_kernel(
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
            total,
            path_count,
            price,
            standard_error
        );
        prices[result_index] = static_cast<float>(price);
        standard_errors[result_index] =
            static_cast<float>(standard_error);
    }
}

template<typename DriverPolicy, typename ModelPathPolicy,
         typename ProductPolicy, typename SchedulePolicy>
void validate_launch(
    const typename ModelPathPolicy::Parameters* device_models,
    std::size_t model_count,
    const typename ProductPolicy::ProductParameters* device_products,
    std::size_t product_count,
    PriceConstruction construction,
    std::size_t result_count,
    std::size_t result_index,
    std::size_t path_count,
    HybridTimeConfiguration time_configuration,
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
        model_count,
        product_count,
        construction,
        result_count
    );
    if (result_index >= result_count) {
        throw std::invalid_argument(
            "Volterra hybrid FFT result_index exceeds the result array."
        );
    }
    validate_monte_carlo_path_count(path_count);
    validate_time_configuration(time_configuration);
    validate_row_seed_range(result_count, base_seed);
    if (step_count == 0U || step_count > kMaximumHybridFftStepCount) {
        throw std::invalid_argument(
            "Volterra hybrid FFT step_count must be in [1, 4096]."
        );
    }
    if (path_chunk_size == 0U
        || path_chunk_size > path_count
        || path_chunk_size % kPathThreads != 0U) {
        throw std::invalid_argument(
            "Volterra hybrid FFT path_chunk_size must be a positive "
            "multiple of 256 not exceeding the path count."
        );
    }
    validate_grid_x_size((path_chunk_size + 1U) / 2U);
    if (workspace_bytes < required_hybrid_fft_workspace_bytes(
            step_count,
            path_count,
            path_chunk_size
        )) {
        throw std::invalid_argument(
            "Volterra hybrid FFT workspace is smaller than its launch plan."
        );
    }
}

template<typename DriverPolicy, typename ModelPathPolicy,
         typename ProductPolicy, typename SchedulePolicy,
         unsigned int Length, unsigned int ElementsPerThread,
         unsigned int FftsPerBlock>
void launch_fft_length(
    const typename ModelPathPolicy::Parameters* device_models,
    const typename ProductPolicy::ProductParameters* device_products,
    std::size_t product_count,
    PriceConstruction construction,
    std::size_t result_index,
    std::size_t path_count,
    std::uint32_t step_count,
    HybridTimeConfiguration time_configuration,
    std::size_t path_chunk_size,
    float2* kernel_spectrum,
    float* driver_variances,
    PreparedRow<
        DriverPolicy,
        ModelPathPolicy,
        ProductPolicy,
        SchedulePolicy
    >* prepared_row,
    float2* convolutions,
    PartialMoments* partial_moments,
    std::uint64_t base_seed,
    float* device_prices,
    float* device_standard_errors,
    const char* diagnostic_name,
    const char* diagnostic_variant
) {
    using ExecutionTypes = FftTypes<
        Length,
        ElementsPerThread,
        FftsPerBlock
    >;
    using Forward = typename ExecutionTypes::Forward;
    using Inverse = typename ExecutionTypes::Inverse;
    using PreparationTypes = FftTypes<
        Length,
        ElementsPerThread,
        1U
    >;
    using KernelForward = typename PreparationTypes::Forward;
    using Row = PreparedRow<
        DriverPolicy,
        ModelPathPolicy,
        ProductPolicy,
        SchedulePolicy
    >;
    static_assert(!Forward::requires_workspace);
    static_assert(!Inverse::requires_workspace);
    static_assert(!KernelForward::requires_workspace);
    static_assert(sizeof(Row) <= kHybridFftPreparedRowBytes);
    constexpr std::size_t execution_shared_bytes = std::max({
        static_cast<std::size_t>(Forward::shared_memory_size),
        static_cast<std::size_t>(Inverse::shared_memory_size),
    });
    constexpr std::size_t preparation_shared_bytes =
        KernelForward::shared_memory_size;

    static const int active_blocks = [] {
        check_cuda(
            cudaFuncSetAttribute(
                prepare_row_kernel<
                    DriverPolicy,
                    ModelPathPolicy,
                    ProductPolicy,
                    SchedulePolicy,
                    Length,
                    KernelForward
                >,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(preparation_shared_bytes)
            ),
            "Volterra hybrid FFT row-preparation shared-memory opt-in"
        );
        check_cuda(
            cudaFuncSetAttribute(
                convolve_paths_kernel<
                    DriverPolicy,
                    ModelPathPolicy,
                    ProductPolicy,
                    SchedulePolicy,
                    Length,
                    Forward,
                    Inverse
                >,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(execution_shared_bytes)
            ),
            "Volterra hybrid FFT convolution shared-memory opt-in"
        );
        int resident_blocks = 0;
        check_cuda(
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &resident_blocks,
                convolve_paths_kernel<
                    DriverPolicy,
                    ModelPathPolicy,
                    ProductPolicy,
                    SchedulePolicy,
                    Length,
                    Forward,
                    Inverse
                >,
                static_cast<int>(
                    Forward::block_dim.x * Forward::block_dim.y
                ),
                execution_shared_bytes
            ),
            "Volterra hybrid FFT convolution occupancy check"
        );
        return resident_blocks;
    }();
    if (active_blocks == 0) {
        throw std::invalid_argument(
            "The Volterra hybrid FFT convolution kernel cannot reside on an SM."
        );
    }

    prepare_row_kernel<
        DriverPolicy,
        ModelPathPolicy,
        ProductPolicy,
        SchedulePolicy,
        Length,
        KernelForward
    ><<<1U, KernelForward::block_dim, preparation_shared_bytes>>>(
        device_models,
        device_products,
        product_count,
        construction,
        result_index,
        step_count,
        time_configuration,
        base_seed,
        kernel_spectrum,
        driver_variances,
        prepared_row
    );
    check_cuda(cudaGetLastError(), "Volterra hybrid FFT row preparation");

    constexpr std::size_t path_shared_bytes =
        2U * (kPathThreads / 32U) * sizeof(double);
    for (std::size_t path_offset = 0U;
         path_offset < path_count;
         path_offset += path_chunk_size) {
        const std::size_t chunk_path_count = std::min(
            path_chunk_size,
            path_count - path_offset
        );
        const std::size_t chunk_pair_count =
            (chunk_path_count + 1U) / 2U;
        const std::size_t convolution_block_count =
            (chunk_pair_count + FftsPerBlock - 1U) / FftsPerBlock;
        report_cuda_kernel_launch_if_enabled(
            diagnostic_name,
            diagnostic_variant,
            convolve_paths_kernel<
                DriverPolicy,
                ModelPathPolicy,
                ProductPolicy,
                SchedulePolicy,
                Length,
                Forward,
                Inverse
            >,
            dim3(static_cast<unsigned int>(convolution_block_count)),
            Forward::block_dim,
            execution_shared_bytes
        );
        convolve_paths_kernel<
            DriverPolicy,
            ModelPathPolicy,
            ProductPolicy,
            SchedulePolicy,
            Length,
            Forward,
            Inverse
        ><<<
            static_cast<unsigned int>(convolution_block_count),
            Forward::block_dim,
            execution_shared_bytes
        >>>(
            path_offset,
            chunk_path_count,
            prepared_row,
            kernel_spectrum,
            convolutions
        );
        check_cuda(cudaGetLastError(), "Volterra hybrid FFT convolution");

        const std::size_t path_block_count =
            (chunk_path_count + kPathThreads - 1U) / kPathThreads;
        evaluate_paths_kernel<
            DriverPolicy,
            ModelPathPolicy,
            ProductPolicy,
            SchedulePolicy
        ><<<
            static_cast<unsigned int>(path_block_count),
            kPathThreads,
            path_shared_bytes
        >>>(
            path_offset,
            chunk_path_count,
            prepared_row,
            driver_variances,
            convolutions,
            partial_moments
        );
        check_cuda(cudaGetLastError(), "Volterra hybrid FFT path evaluation");
    }

    constexpr std::size_t finalization_shared_bytes =
        2U * (kFinalizationThreads / 32U) * sizeof(double);
    finalize_price_kernel<<<
        1U,
        kFinalizationThreads,
        finalization_shared_bytes
    >>>(
        partial_moments,
        (path_count + kPathThreads - 1U) / kPathThreads,
        path_count,
        result_index,
        device_prices,
        device_standard_errors
    );
    check_cuda(cudaGetLastError(), "Volterra hybrid FFT finalization");
}

#if AI_FACTORY_VOLTERRA_DIRECT_MAX_STEP_COUNT > 0
template<typename DriverPolicy, typename ModelPathPolicy,
         typename ProductPolicy, typename SchedulePolicy>
void launch_direct(
    const typename ModelPathPolicy::Parameters* device_models,
    const typename ProductPolicy::ProductParameters* device_products,
    std::size_t product_count,
    PriceConstruction construction,
    std::size_t result_index,
    std::size_t path_count,
    std::uint32_t step_count,
    HybridTimeConfiguration time_configuration,
    std::size_t path_chunk_size,
    float* driver_variances,
    PreparedRow<
        DriverPolicy,
        ModelPathPolicy,
        ProductPolicy,
        SchedulePolicy
    >* prepared_row,
    PartialMoments* partial_moments,
    std::uint64_t base_seed,
    float* device_prices,
    float* device_standard_errors,
    const char* diagnostic_name,
    const char* diagnostic_variant
) {
    prepare_direct_row_kernel<
        DriverPolicy,
        ModelPathPolicy,
        ProductPolicy,
        SchedulePolicy
    ><<<1U, kPathThreads>>>(
        device_models,
        device_products,
        product_count,
        construction,
        result_index,
        step_count,
        time_configuration,
        base_seed,
        driver_variances,
        prepared_row
    );
    check_cuda(cudaGetLastError(), "Volterra direct row preparation");

    constexpr std::size_t path_shared_bytes =
        2U * (kPathThreads / 32U) * sizeof(double);
    for (std::size_t path_offset = 0U;
         path_offset < path_count;
         path_offset += path_chunk_size) {
        const std::size_t chunk_path_count = std::min(
            path_chunk_size,
            path_count - path_offset
        );
        const std::size_t path_block_count =
            (chunk_path_count + kPathThreads - 1U) / kPathThreads;
        report_cuda_kernel_launch_if_enabled(
            diagnostic_name,
            diagnostic_variant,
            evaluate_direct_paths_kernel<
                DriverPolicy,
                ModelPathPolicy,
                ProductPolicy,
                SchedulePolicy
            >,
            dim3(static_cast<unsigned int>(path_block_count)),
            dim3(kPathThreads),
            path_shared_bytes
        );
        evaluate_direct_paths_kernel<
            DriverPolicy,
            ModelPathPolicy,
            ProductPolicy,
            SchedulePolicy
        ><<<
            static_cast<unsigned int>(path_block_count),
            kPathThreads,
            path_shared_bytes
        >>>(
            path_offset,
            chunk_path_count,
            prepared_row,
            driver_variances,
            partial_moments
        );
        check_cuda(cudaGetLastError(), "Volterra direct path evaluation");
    }

    constexpr std::size_t finalization_shared_bytes =
        2U * (kFinalizationThreads / 32U) * sizeof(double);
    finalize_price_kernel<<<
        1U,
        kFinalizationThreads,
        finalization_shared_bytes
    >>>(
        partial_moments,
        (path_count + kPathThreads - 1U) / kPathThreads,
        path_count,
        result_index,
        device_prices,
        device_standard_errors
    );
    check_cuda(cudaGetLastError(), "Volterra direct finalization");
}
#endif

template<typename DriverPolicy, typename ModelPathPolicy,
         typename ProductPolicy, typename SchedulePolicy>
void launch_pricing_cuda(
    const typename ModelPathPolicy::Parameters* device_models,
    std::size_t model_count,
    const typename ProductPolicy::ProductParameters* device_products,
    std::size_t product_count,
    PriceConstruction construction,
    std::size_t result_count,
    std::size_t result_index,
    std::size_t monte_carlo_paths_per_price,
    HybridTimeConfiguration time_configuration,
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
    validate_launch<
        DriverPolicy,
        ModelPathPolicy,
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
        device_standard_errors
    );

    auto* const workspace = static_cast<unsigned char*>(device_workspace);
    auto* const spectrum = reinterpret_cast<float2*>(workspace);
    auto* const driver_variances = reinterpret_cast<float*>(
        workspace + kHybridFftSpectrumBytes
    );
    using Row = PreparedRow<
        DriverPolicy,
        ModelPathPolicy,
        ProductPolicy,
        SchedulePolicy
    >;
    auto* const prepared_row = reinterpret_cast<Row*>(
        workspace + kHybridFftPreparedRowOffset
    );
    auto* const convolutions = reinterpret_cast<float2*>(
        workspace + kHybridFftConvolutionOffset
    );
    auto* const partial_moments = reinterpret_cast<PartialMoments*>(
        workspace + kHybridFftConvolutionOffset
            + hybrid_fft_convolution_bytes(step_count, path_chunk_size)
    );
    const std::uint32_t device_step_count =
        static_cast<std::uint32_t>(step_count);

#if AI_FACTORY_VOLTERRA_DIRECT_MAX_STEP_COUNT > 0
    if (step_count <= kDirectMaximumStepCount) {
        launch_direct<
            DriverPolicy,
            ModelPathPolicy,
            ProductPolicy,
            SchedulePolicy
        >(
            device_models,
            device_products,
            product_count,
            construction,
            result_index,
            monte_carlo_paths_per_price,
            device_step_count,
            time_configuration,
            path_chunk_size,
            driver_variances,
            prepared_row,
            partial_moments,
            base_seed,
            device_prices,
            device_standard_errors,
            diagnostic_name,
            diagnostic_variant
        );
    } else
#endif
    if (step_count <= 8U) {
        launch_fft_length<
            DriverPolicy, ModelPathPolicy, ProductPolicy, SchedulePolicy,
            16U, 8U, 16U
        >(
            device_models, device_products, product_count, construction,
            result_index, monte_carlo_paths_per_price, device_step_count,
            time_configuration, path_chunk_size, spectrum,
            driver_variances, prepared_row, convolutions, partial_moments, base_seed,
            device_prices, device_standard_errors, diagnostic_name,
            diagnostic_variant
        );
    } else if (step_count <= 32U) {
        launch_fft_length<
            DriverPolicy, ModelPathPolicy, ProductPolicy, SchedulePolicy,
            64U, 8U, 8U
        >(
            device_models, device_products, product_count, construction,
            result_index, monte_carlo_paths_per_price, device_step_count,
            time_configuration, path_chunk_size, spectrum,
            driver_variances, prepared_row, convolutions, partial_moments, base_seed,
            device_prices, device_standard_errors, diagnostic_name,
            diagnostic_variant
        );
    } else if (step_count <= 64U) {
        launch_fft_length<
            DriverPolicy, ModelPathPolicy, ProductPolicy, SchedulePolicy,
            128U, 8U, 8U
        >(
            device_models, device_products, product_count, construction,
            result_index, monte_carlo_paths_per_price, device_step_count,
            time_configuration, path_chunk_size, spectrum,
            driver_variances, prepared_row, convolutions, partial_moments, base_seed,
            device_prices, device_standard_errors, diagnostic_name,
            diagnostic_variant
        );
    } else if (step_count <= 128U) {
        launch_fft_length<
            DriverPolicy, ModelPathPolicy, ProductPolicy, SchedulePolicy,
            256U, 16U, 8U
        >(
            device_models, device_products, product_count, construction,
            result_index, monte_carlo_paths_per_price, device_step_count,
            time_configuration, path_chunk_size, spectrum,
            driver_variances, prepared_row, convolutions, partial_moments, base_seed,
            device_prices, device_standard_errors, diagnostic_name,
            diagnostic_variant
        );
    } else if (step_count <= 256U) {
        launch_fft_length<
            DriverPolicy, ModelPathPolicy, ProductPolicy, SchedulePolicy,
            512U, 8U, 2U
        >(
            device_models, device_products, product_count, construction,
            result_index, monte_carlo_paths_per_price, device_step_count,
            time_configuration, path_chunk_size, spectrum,
            driver_variances, prepared_row, convolutions, partial_moments, base_seed,
            device_prices, device_standard_errors, diagnostic_name,
            diagnostic_variant
        );
    } else if (step_count <= 512U) {
        launch_fft_length<
            DriverPolicy, ModelPathPolicy, ProductPolicy, SchedulePolicy,
            1024U, 16U, 1U
        >(
            device_models, device_products, product_count, construction,
            result_index, monte_carlo_paths_per_price, device_step_count,
            time_configuration, path_chunk_size, spectrum,
            driver_variances, prepared_row, convolutions, partial_moments, base_seed,
            device_prices, device_standard_errors, diagnostic_name,
            diagnostic_variant
        );
    } else if (step_count <= 1024U) {
        launch_fft_length<
            DriverPolicy, ModelPathPolicy, ProductPolicy, SchedulePolicy,
            2048U, 16U, 1U
        >(
            device_models, device_products, product_count, construction,
            result_index, monte_carlo_paths_per_price, device_step_count,
            time_configuration, path_chunk_size, spectrum,
            driver_variances, prepared_row, convolutions, partial_moments, base_seed,
            device_prices, device_standard_errors, diagnostic_name,
            diagnostic_variant
        );
    } else if (step_count <= 2048U) {
        launch_fft_length<
            DriverPolicy, ModelPathPolicy, ProductPolicy, SchedulePolicy,
            4096U, 16U, 1U
        >(
            device_models, device_products, product_count, construction,
            result_index, monte_carlo_paths_per_price, device_step_count,
            time_configuration, path_chunk_size, spectrum,
            driver_variances, prepared_row, convolutions, partial_moments, base_seed,
            device_prices, device_standard_errors, diagnostic_name,
            diagnostic_variant
        );
    } else {
        launch_fft_length<
            DriverPolicy, ModelPathPolicy, ProductPolicy, SchedulePolicy,
            8192U, 16U, 1U
        >(
            device_models, device_products, product_count, construction,
            result_index, monte_carlo_paths_per_price, device_step_count,
            time_configuration, path_chunk_size, spectrum,
            driver_variances, prepared_row, convolutions, partial_moments, base_seed,
            device_prices, device_standard_errors, diagnostic_name,
            diagnostic_variant
        );
    }
}

}  // namespace ai_factory::workbench::volterra::hybrid_fft
