// Causal, block-dyadic Volterra Monte Carlo pricing.
#pragma once

#include "common/check_cuda.cuh"
#include "common/equity/path_product_policy.cuh"
#include "common/result_index.cuh"
#include "common/reductions.cuh"
#include "common/volterra/causal_fft_core.cuh"
#include "common/volterra/hybrid_fft.cuh"
#include "common/volterra/hybrid_fft_tuning.cuh"
#include "common/volterra/hybrid_schedule.cuh"
#include "common/volterra/pricing_moments.cuh"
#include "common/volterra/pricing_row.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace ai_factory::workbench::volterra::causal_fft {

constexpr unsigned int kPathThreads = tuning::kPricingPathThreads;

// The model policy owns only one time cell.  The engine owns history,
// calendar observations, FFTs, chunking and payoff moments.
template<typename PathPolicy, typename ProductPolicy,
         typename SchedulePolicy>
using PreparedRow = PricingRow<
    typename PathPolicy::KernelPolicy,
    PathPolicy,
    ProductPolicy,
    SchedulePolicy
>;

template<typename PathPolicy, typename ProductPolicy,
         typename SchedulePolicy>
struct PathContext {
    typename PathPolicy::State state;
    typename ProductPolicy::Handler handler;
    typename SchedulePolicy::Cursor cursor;
    bool active;
};

inline std::size_t aligned_size(std::size_t value, std::size_t alignment) {
    if (value > std::numeric_limits<std::size_t>::max() - alignment)
        throw std::overflow_error("causal FFT workspace alignment overflow");
    return (value + alignment - 1U) / alignment * alignment;
}

inline std::size_t checked_bytes(std::size_t left, std::size_t right) {
    if (right && left > std::numeric_limits<std::size_t>::max() / right)
        throw std::overflow_error("causal FFT workspace size overflow");
    return left * right;
}

inline std::size_t checked_add(std::size_t left, std::size_t right) {
    if (left > std::numeric_limits<std::size_t>::max() - right)
        throw std::overflow_error("causal FFT workspace size overflow");
    return left + right;
}

struct WorkspacePlan {
    std::size_t row_offset;
    std::size_t weight_offset;
    std::size_t spectrum_offset;
    std::size_t history_offset;
    std::size_t context_offset;
    std::size_t moment_offset;
    std::size_t spectrum_stride;
    std::size_t partials_per_row;
    std::size_t workspace_bytes;
};

template<typename PathPolicy, typename ProductPolicy,
         typename SchedulePolicy>
WorkspacePlan plan_workspace(
    std::size_t steps, std::size_t paths_per_row,
    std::size_t path_chunk_size, std::size_t row_batch_size
) {
    if (steps == 0U || steps > 8192U || paths_per_row < 2U
        || path_chunk_size == 0U || path_chunk_size % kPathThreads != 0U
        || row_batch_size == 0U || row_batch_size > 65535U) {
        throw std::invalid_argument("invalid causal FFT workspace dimensions");
    }
    using Row = PreparedRow<PathPolicy, ProductPolicy, SchedulePolicy>;
    using Context = PathContext<PathPolicy, ProductPolicy, SchedulePolicy>;
    const std::size_t partials =
        (paths_per_row + kPathThreads - 1U) / kPathThreads;
    // Each dyadic scale uses 2s complex coefficients.  4N is a safe
    // upper bound for their concatenation, including the largest scale.
    const std::size_t spectrum_stride = checked_bytes(steps, 4U);
    std::size_t end = 0U;
    const std::size_t row_offset = end;
    end = checked_add(end, checked_bytes(row_batch_size, sizeof(Row)));
    end = aligned_size(end, alignof(float));
    const std::size_t weight_offset = end;
    end = checked_add(end, checked_bytes(
        checked_bytes(row_batch_size, steps), sizeof(float)
    ));
    end = aligned_size(end, alignof(float2));
    const std::size_t spectrum_offset = end;
    end = checked_add(end, checked_bytes(
        checked_bytes(row_batch_size, spectrum_stride), sizeof(float2)
    ));
    end = aligned_size(end, alignof(float));
    const std::size_t history_offset = end;
    end = checked_add(end, checked_bytes(
        checked_bytes(checked_bytes(row_batch_size, path_chunk_size), steps),
        sizeof(float)
    ));
    end = aligned_size(end, alignof(Context));
    const std::size_t context_offset = end;
    end = checked_add(end, checked_bytes(
        checked_bytes(row_batch_size, path_chunk_size), sizeof(Context)
    ));
    end = aligned_size(end, alignof(PartialMoments));
    const std::size_t moment_offset = end;
    end = checked_add(end, checked_bytes(
        checked_bytes(row_batch_size, partials), sizeof(PartialMoments)
    ));
    return {
        row_offset, weight_offset, spectrum_offset, history_offset,
        context_offset, moment_offset, spectrum_stride, partials, end,
    };
}

template<typename PathPolicy, typename ProductPolicy,
         typename SchedulePolicy>
__global__ void prepare_rows_kernel(
    const typename PathPolicy::Parameters* models,
    const typename ProductPolicy::ProductParameters* products,
    std::size_t product_count,
    PriceConstruction construction,
    std::size_t result_offset,
    std::size_t row_count,
    std::uint32_t steps,
    HybridTimeConfiguration time_config,
    std::uint64_t base_seed,
    PreparedRow<PathPolicy, ProductPolicy, SchedulePolicy>* rows
) {
    const std::size_t local_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_row >= row_count) return;
    const std::size_t result = result_offset + local_row;
    const auto indices = decode_model_product_result_index_32(
        static_cast<std::uint32_t>(result),
        static_cast<std::uint32_t>(product_count), construction
    );
    const auto model = models[indices.model_index];
    const auto product = products[indices.product_index];
    const auto schedule = SchedulePolicy::prepare(
        ProductPolicy::calendar(product), time_config, steps
    );
    using Kernel = typename PathPolicy::KernelPolicy;
    rows[local_row] = {
        Kernel::prepare(PathPolicy::kernel_parameters(model),
                        schedule.time_step),
        PathPolicy::prepare_model(model, schedule.time_step),
        ProductPolicy::prepare_product(
            model, product,
            equity::ProductPreparationContext{
                time_config.day_fraction, schedule.maturity_years
            }
        ),
        schedule,
        philox::make_key(base_seed + result),
        sqrtf(schedule.time_step),
    };
}

template<typename PathPolicy, typename ProductPolicy,
         typename SchedulePolicy>
__global__ void advance_tile_kernel(
    const PreparedRow<PathPolicy, ProductPolicy, SchedulePolicy>* rows,
    std::size_t row_count,
    std::size_t path_offset,
    std::size_t path_count,
    std::size_t chunk_capacity,
    std::uint32_t steps,
    std::uint32_t first_step,
    std::uint32_t tile_size,
    const float* weights,
    float* history,
    PathContext<PathPolicy, ProductPolicy, SchedulePolicy>* contexts,
    PartialMoments* moments,
    std::size_t partials_per_row
) {
    const std::size_t row_index = blockIdx.y;
    if (row_index >= row_count) return;
    const std::size_t path = blockIdx.x * blockDim.x + threadIdx.x;
    using Row = PreparedRow<PathPolicy, ProductPolicy, SchedulePolicy>;
    __shared__ Row row;
    if (threadIdx.x == 0U) row = rows[row_index];
    __syncthreads();
    float* row_history = history + row_index * chunk_capacity * steps;
    const float* row_weights = weights + row_index * steps;
    auto* row_context = contexts + row_index * chunk_capacity;
    const std::uint32_t end = min(first_step + tile_size, steps);
    double payoff = 0.0;
    if (path < path_count) {
        using Context = PathContext<PathPolicy, ProductPolicy, SchedulePolicy>;
        Context context;
        if (first_step == 0U) {
            context.state = PathPolicy::initial_state(row.model);
            context.handler = ProductPolicy::make_handler(row.product);
            context.cursor = SchedulePolicy::make_cursor(row.schedule);
            equity::PathProductObservationAdapter<
                PathPolicy, typename ProductPolicy::Handler,
                ProductPolicy::kObservationCoordinate
            > observer{context.handler};
            context.active = SchedulePolicy::on_initial_state(
                row.schedule, context.cursor, context.state, observer
            );
        } else {
            context = row_context[path];
        }
        for (std::uint32_t step = first_step; step < end; ++step) {
            const std::size_t slot = history_index(
                path, step, steps
            );
            if (context.active) {
                float far = step == 0U ? 0.0f : row_history[slot];
                for (std::uint32_t prior = first_step; prior < step;
                     ++prior) {
                    far = fmaf(
                        row_weights[step - prior - 1U],
                        row_history[history_index(
                            path, prior, steps
                        )],
                        far
                    );
                }
                const float source = PathPolicy::advance_cell(
                    row.model, row.kernel, far, row.key,
                    path_offset + path, step, context.state
                );
                row_history[slot] = source;
                equity::PathProductObservationAdapter<
                    PathPolicy, typename ProductPolicy::Handler,
                    ProductPolicy::kObservationCoordinate
                > observer{context.handler};
                context.active = SchedulePolicy::on_step(
                    row.schedule, context.cursor, step,
                    context.state, observer
                );
            } else {
                row_history[slot] = 0.0f;
            }
        }
        row_context[path] = context;
        if (end == steps) payoff = ProductPolicy::template finalize<PathPolicy>(
            row.product, context.state, context.handler
        );
    }
    if (end == steps) {
        const auto sums = reductions::reduce_block(payoff, payoff * payoff);
        if (threadIdx.x == 0U) {
            moments[row_index * partials_per_row
                + path_offset / kPathThreads + blockIdx.x] = {
                    sums.sum, sums.sumsq
                };
        }
    }
}

template<typename PathPolicy, typename ProductPolicy,
         typename SchedulePolicy>
void launch_pricing_cuda(
    const typename PathPolicy::Parameters* device_models,
    std::size_t model_count,
    const typename ProductPolicy::ProductParameters* device_products,
    std::size_t product_count,
    PriceConstruction construction,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    std::size_t paths_per_price,
    HybridTimeConfiguration time_config,
    std::uint32_t steps,
    std::uint32_t direct_tile_size,
    std::size_t path_chunk_size,
    std::size_t row_batch_size,
    void* device_workspace,
    std::size_t workspace_bytes,
    std::uint64_t base_seed,
    float* device_prices,
    float* device_standard_errors
) {
    static_assert(equity::EquityPathProductPolicy<ProductPolicy, PathPolicy>);
    static_assert(std::is_trivially_copyable_v<
        PathContext<PathPolicy, ProductPolicy, SchedulePolicy>
    >);
    if (!device_models || !device_products || !device_workspace
        || !device_prices || !device_standard_errors || model_count == 0U
        || product_count == 0U || result_offset > result_count
        || launch_result_count > result_count - result_offset
        || paths_per_price < 2U || steps == 0U || steps > 8192U
        || (direct_tile_size != 32U && direct_tile_size != 64U)
        || path_chunk_size == 0U || path_chunk_size % kPathThreads != 0U
        || row_batch_size == 0U || row_batch_size > 65535U) {
        throw std::invalid_argument("invalid causal FFT pricing arguments");
    }
    validate_time_configuration(time_config);
    if (price_row_count(model_count, product_count, construction)
            != result_count
        || result_count > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("invalid causal FFT result cardinality");
    }
    const auto plan = plan_workspace<
        PathPolicy, ProductPolicy, SchedulePolicy
    >(steps, paths_per_price, path_chunk_size, row_batch_size);
    if (workspace_bytes < plan.workspace_bytes)
        throw std::invalid_argument("causal FFT workspace is too small");
    auto* bytes = static_cast<unsigned char*>(device_workspace);
    using Row = PreparedRow<PathPolicy, ProductPolicy, SchedulePolicy>;
    using Context = PathContext<PathPolicy, ProductPolicy, SchedulePolicy>;
    auto* rows = reinterpret_cast<Row*>(bytes + plan.row_offset);
    auto* weights = reinterpret_cast<float*>(bytes + plan.weight_offset);
    auto* spectra = reinterpret_cast<float2*>(bytes + plan.spectrum_offset);
    auto* history = reinterpret_cast<float*>(bytes + plan.history_offset);
    auto* contexts = reinterpret_cast<Context*>(bytes + plan.context_offset);
    auto* moments = reinterpret_cast<PartialMoments*>(bytes + plan.moment_offset);

    for (std::size_t row_offset = 0U; row_offset < launch_result_count;
         row_offset += row_batch_size) {
        const std::size_t batch_rows = std::min(
            row_batch_size, launch_result_count - row_offset
        );
        prepare_rows_kernel<PathPolicy, ProductPolicy, SchedulePolicy>
            <<<static_cast<unsigned int>((batch_rows + 63U) / 64U), 64U>>>(
                device_models, device_products, product_count, construction,
                result_offset + row_offset, batch_rows, steps, time_config,
                base_seed, rows
            );
        check_cuda(cudaGetLastError(), "causal FFT row preparation");
        prepare_weights_kernel<Row>
            <<<dim3(static_cast<unsigned int>((steps + 255U) / 256U),
                    static_cast<unsigned int>(batch_rows)), 256U>>>(
                rows, batch_rows, steps, weights
            );
        check_cuda(cudaGetLastError(), "causal FFT weight preparation");

        for (std::uint32_t scale = direct_tile_size; scale < steps;
             scale <<= 1U) {
            const std::size_t offset = 2U * (scale - direct_tile_size);
            dispatch_scale(scale, [&]<unsigned int Length,
                unsigned int ElementsPerThread,
                unsigned int FftsPerBlock>() {
                using Types = hybrid_fft::FftTypes<
                    Length, ElementsPerThread, FftsPerBlock
                >;
                using Forward = typename Types::Forward;
                constexpr auto kernel = prepare_spectrum_kernel<
                    Length, ElementsPerThread, FftsPerBlock,
                    Row
                >;
                constexpr std::size_t shared = Forward::shared_memory_size;
                check_cuda(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(shared)
                ), "causal FFT spectrum shared memory");
                kernel<<<static_cast<unsigned int>((batch_rows
                    + FftsPerBlock - 1U) / FftsPerBlock),
                    Forward::block_dim, shared>>>(
                        rows, batch_rows, steps, scale, offset,
                        plan.spectrum_stride, weights, spectra
                    );
                check_cuda(cudaGetLastError(), "causal FFT spectrum preparation");
                using Inverse = typename Types::Inverse;
                constexpr auto update = update_history_kernel<
                    Length, ElementsPerThread, FftsPerBlock
                >;
                constexpr std::size_t update_shared = std::max(
                    static_cast<std::size_t>(Forward::shared_memory_size),
                    static_cast<std::size_t>(Inverse::shared_memory_size)
                );
                check_cuda(cudaFuncSetAttribute(
                    update, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(update_shared)
                ), "causal FFT update shared memory");
            });
        }

        for (std::size_t path_offset = 0U; path_offset < paths_per_price;
             path_offset += path_chunk_size) {
            const std::size_t chunk_paths = std::min(
                path_chunk_size, paths_per_price - path_offset
            );
            check_cuda(cudaMemsetAsync(
                history, 0, batch_rows * path_chunk_size * steps
                    * sizeof(float)
            ), "causal FFT clear history");
            for (std::uint32_t first = 0U; first < steps;
                 first += direct_tile_size) {
                advance_tile_kernel<
                    PathPolicy, ProductPolicy, SchedulePolicy
                ><<<dim3(static_cast<unsigned int>((chunk_paths
                        + kPathThreads - 1U) / kPathThreads),
                        static_cast<unsigned int>(batch_rows)),
                    kPathThreads>>>(
                        rows, batch_rows, path_offset, chunk_paths,
                        path_chunk_size, steps, first, direct_tile_size,
                        weights, history, contexts, moments, plan.partials_per_row
                    );
                check_cuda(cudaGetLastError(), "causal FFT direct tile");
                const std::uint32_t boundary = first + direct_tile_size;
                if (boundary >= steps) continue;
                const std::uint32_t scale = boundary & (~boundary + 1U);
                const std::size_t offset = 2U * (scale - direct_tile_size);
                dispatch_scale(scale, [&]<unsigned int Length,
                    unsigned int ElementsPerThread,
                    unsigned int FftsPerBlock>() {
                    using Types = hybrid_fft::FftTypes<
                        Length, ElementsPerThread, FftsPerBlock
                    >;
                    using Forward = typename Types::Forward;
                    using Inverse = typename Types::Inverse;
                    constexpr auto kernel = update_history_kernel<
                        Length, ElementsPerThread, FftsPerBlock
                    >;
                    constexpr std::size_t shared = std::max(
                        static_cast<std::size_t>(Forward::shared_memory_size),
                        static_cast<std::size_t>(Inverse::shared_memory_size)
                    );
                    kernel<<<dim3(static_cast<unsigned int>((
                        (chunk_paths + 1U) / 2U + FftsPerBlock - 1U
                        ) / FftsPerBlock),
                        static_cast<unsigned int>(batch_rows)),
                        Forward::block_dim, shared>>>(
                            spectra, plan.spectrum_stride, offset,
                            history, batch_rows, path_chunk_size,
                            chunk_paths, steps, boundary, scale
                        );
                    check_cuda(cudaGetLastError(), "causal FFT history update");
                });
            }
        }
        for (std::size_t local_row = 0U; local_row < batch_rows;
             ++local_row) {
            constexpr std::size_t shared = 2U *
                (kPathThreads / 32U) * sizeof(double);
            volterra::finalize_price_kernel<<<1U, kPathThreads, shared>>>(
                moments + local_row * plan.partials_per_row,
                plan.partials_per_row, paths_per_price,
                result_offset + row_offset + local_row,
                device_prices, device_standard_errors
            );
            check_cuda(cudaGetLastError(), "causal FFT finalization");
        }
    }
}

}  // namespace ai_factory::workbench::volterra::causal_fft
