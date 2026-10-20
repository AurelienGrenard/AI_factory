// Product-independent causal FFT generation of a rough volatility path.
#pragma once

#include "common/check_cuda.cuh"
#include "common/multi_asset_price_row.hpp"
#include "common/simulation/correlated_brownian.cuh"
#include "common/volterra/causal_fft_core.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ai_factory::workbench::volterra::causal_fft {

template<typename Policy>
struct VolatilityRow {
    using Kernel = typename Policy::KernelPolicy;
    typename Kernel::PreparedKernel kernel;
    typename Policy::PreparedModel model;
    philox::PhiloxKey key;
    const float* packed_lower;
};

template<typename Policy>
struct VolatilityContext {
    typename Policy::VolatilityState state;
};

struct VolatilityWorkspacePlan {
    std::size_t weights_offset;
    std::size_t spectra_offset;
    std::size_t history_offset;
    std::size_t context_offset;
    std::size_t bytes;
};

inline std::size_t volatility_aligned_size(
    std::size_t bytes, std::size_t alignment
) {
    if (bytes > std::numeric_limits<std::size_t>::max() - alignment + 1U)
        throw std::overflow_error("causal volatility workspace alignment overflow");
    return (bytes + alignment - 1U) / alignment * alignment;
}

inline std::size_t volatility_checked_mul(
    std::size_t left, std::size_t right
) {
    if (right && left > std::numeric_limits<std::size_t>::max() / right)
        throw std::overflow_error("causal volatility workspace size overflow");
    return left * right;
}

inline std::size_t volatility_checked_add(
    std::size_t left, std::size_t right
) {
    if (left > std::numeric_limits<std::size_t>::max() - right)
        throw std::overflow_error("causal volatility workspace size overflow");
    return left + right;
}

template<typename Policy, typename Context = VolatilityContext<Policy>>
VolatilityWorkspacePlan plan_volatility_workspace(
    std::size_t steps, std::size_t path_count
) {
    if (steps == 0U || steps > 8192U || path_count == 0U)
        throw std::invalid_argument("invalid causal volatility workspace size");
    std::size_t end = sizeof(VolatilityRow<Policy>);
    end = volatility_aligned_size(end, alignof(float));
    const std::size_t weights = end;
    end = volatility_checked_add(end,
        volatility_checked_mul(steps, sizeof(float)));
    end = volatility_aligned_size(end, alignof(float2));
    const std::size_t spectra = end;
    end = volatility_checked_add(end,
        volatility_checked_mul(
            volatility_checked_mul(4U, steps), sizeof(float2)));
    end = volatility_aligned_size(end, alignof(float));
    const std::size_t history = end;
    end = volatility_checked_add(end,
        volatility_checked_mul(
            volatility_checked_mul(path_count, steps), sizeof(float)));
    end = volatility_aligned_size(end, alignof(Context));
    const std::size_t contexts = end;
    end = volatility_checked_add(end,
        volatility_checked_mul(path_count, sizeof(Context)));
    return {weights, spectra, history, contexts, end};
}

template<typename Policy, std::size_t Dimension>
__global__ void prepare_volatility_row_kernel(
    const typename Policy::Parameters* models,
    const std::uint32_t* scenario_model_indices,
    const monte_carlo::MultiAssetPriceRow* price_rows,
    const float* packed_factors,
    std::size_t result_index,
    float dt,
    std::uint64_t base_seed,
    std::size_t scenario_stride,
    std::size_t scenario_slot,
    VolatilityRow<Policy>* output
) {
    if (threadIdx.x != 0U) return;
    const auto price = price_rows[result_index];
    const auto parameters = models[scenario_model_indices[
        static_cast<std::size_t>(price.scenario_index) * scenario_stride
            + scenario_slot
    ]];
    output->kernel = Policy::KernelPolicy::prepare(
        Policy::kernel_parameters(parameters), dt
    );
    output->model = Policy::prepare_model(parameters, dt);
    output->key = philox::make_key(base_seed + result_index);
    output->packed_lower = packed_factors
        + static_cast<std::size_t>(price.correlation_index)
            * Dimension * (Dimension + 1U) / 2U;
}

template<typename Policy>
__global__ void advance_volatility_tile_kernel(
    const VolatilityRow<Policy>* rows,
    std::size_t path_offset,
    std::size_t path_count,
    std::uint32_t steps,
    std::uint32_t first_step,
    std::uint32_t tile_size,
    const float* weights,
    float* history,
    VolatilityContext<Policy>* contexts,
    float* volatility_path,
    std::size_t volatility_factor,
    std::uint32_t auxiliary_source
) {
    const std::size_t path = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ VolatilityRow<Policy> row;
    if (threadIdx.x == 0U) row = rows[0];
    __syncthreads();
    if (path >= path_count) return;
    auto context = first_step == 0U
        ? VolatilityContext<Policy>{
            Policy::initial_volatility_state(row.model)
        } : contexts[path];
    const std::uint32_t end = min(first_step + tile_size, steps);
    for (std::uint32_t step = first_step; step < end; ++step) {
        const std::size_t slot = history_index(path, step, steps);
        float far = step == 0U ? 0.0f : history[slot];
        for (std::uint32_t prior = first_step; prior < step; ++prior) {
            far = fmaf(weights[step - prior - 1U],
                history[history_index(path, prior, steps)], far);
        }
        const float normal = simulation::correlated_brownian_factor(
            row.key, path_offset + path, step, volatility_factor,
            row.packed_lower
        );
        float singular = 0.0f;
        if constexpr (Policy::kBrownianDimension == 2U) {
            const auto uniforms = philox::domain_uniform_quad(
                row.key, path_offset + path, 0U,
                (auxiliary_source << 24U) | step
            );
            singular = philox::box_muller(
                uniforms.first, uniforms.second
            ).first;
        }
        float variance = 0.0f;
        history[slot] = Policy::advance_volatility_cell(
            row.model, row.kernel, far, normal, singular,
            step, context.state, variance
        );
        volatility_path[slot] = variance;
    }
    contexts[path] = context;
}

template<typename Policy, std::size_t Dimension, typename Context,
         typename TileLauncher>
void launch_causal_engine_cuda(
    const typename Policy::Parameters* models,
    const std::uint32_t* scenario_model_indices,
    const monte_carlo::MultiAssetPriceRow* price_rows,
    const float* packed_factors,
    std::size_t result_index,
    std::size_t path_offset,
    std::size_t path_count,
    float dt,
    std::uint32_t steps,
    std::uint32_t direct_tile_size,
    std::uint64_t base_seed,
    void* workspace,
    std::size_t workspace_bytes,
    TileLauncher launch_tile,
    std::size_t scenario_stride = 1U,
    std::size_t scenario_slot = 0U
) {
    if (!models || !scenario_model_indices || !price_rows
        || !packed_factors || !workspace
        || path_count == 0U || path_count % kPathGroup != 0U
        || steps == 0U || steps > 8192U
        || !(dt > 0.0f)
        || (direct_tile_size != 32U && direct_tile_size != 64U)) {
        throw std::invalid_argument("invalid causal volatility launch");
    }
    const auto plan = plan_volatility_workspace<Policy, Context>(steps, path_count);
    if (workspace_bytes < plan.bytes)
        throw std::invalid_argument("causal volatility workspace too small");
    auto* bytes = static_cast<unsigned char*>(workspace);
    using Row = VolatilityRow<Policy>;
    auto* row = reinterpret_cast<Row*>(bytes);
    auto* weights = reinterpret_cast<float*>(bytes + plan.weights_offset);
    auto* spectra = reinterpret_cast<float2*>(bytes + plan.spectra_offset);
    auto* history = reinterpret_cast<float*>(bytes + plan.history_offset);
    auto* contexts = reinterpret_cast<Context*>(
        bytes + plan.context_offset
    );
    prepare_volatility_row_kernel<Policy, Dimension><<<1U, 1U>>>(
        models, scenario_model_indices, price_rows, packed_factors,
        result_index, dt, base_seed,
        scenario_stride, scenario_slot, row
    );
    check_cuda(cudaGetLastError(), "causal volatility row preparation");
    prepare_weights_kernel<Row><<<dim3((steps + 255U) / 256U, 1U), 256U>>>(
        row, 1U, steps, weights
    );
    check_cuda(cudaGetLastError(), "causal volatility weight preparation");
    for (std::uint32_t scale = direct_tile_size; scale < steps;
         scale <<= 1U) {
        const std::size_t spectrum_offset =
            2U * (scale - direct_tile_size);
        dispatch_scale(scale, [&]<unsigned int Length,
            unsigned int ElementsPerThread,
            unsigned int FftsPerBlock>() {
            using Types = hybrid_fft::FftTypes<
                Length, ElementsPerThread, FftsPerBlock
            >;
            using Forward = typename Types::Forward;
            using Inverse = typename Types::Inverse;
            constexpr auto prepare = prepare_spectrum_kernel<
                Length, ElementsPerThread, FftsPerBlock, Row
            >;
            constexpr auto update = update_history_kernel<
                Length, ElementsPerThread, FftsPerBlock
            >;
            constexpr std::size_t update_shared = std::max(
                static_cast<std::size_t>(Forward::shared_memory_size),
                static_cast<std::size_t>(Inverse::shared_memory_size)
            );
            check_cuda(cudaFuncSetAttribute(prepare,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(Forward::shared_memory_size)),
                "causal volatility spectrum shared memory");
            check_cuda(cudaFuncSetAttribute(update,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(update_shared)),
                "causal volatility history shared memory");
            prepare<<<1U, Forward::block_dim,
                Forward::shared_memory_size>>>(
                    row, 1U, steps, scale, spectrum_offset,
                    4U * steps, weights, spectra
                );
            check_cuda(cudaGetLastError(),
                "causal volatility spectrum preparation");
        });
    }
    check_cuda(cudaMemsetAsync(history, 0,
        path_count * steps * sizeof(float)),
        "causal volatility clear history");
    for (std::uint32_t first = 0U; first < steps;
         first += direct_tile_size) {
        launch_tile(row, path_offset, path_count, steps, first,
            direct_tile_size, weights, history, contexts);
        const std::uint32_t boundary = first + direct_tile_size;
        if (boundary >= steps) continue;
        const std::uint32_t scale = boundary & (~boundary + 1U);
        const std::size_t spectrum_offset =
            2U * (scale - direct_tile_size);
        dispatch_scale(scale, [&]<unsigned int Length,
            unsigned int ElementsPerThread,
            unsigned int FftsPerBlock>() {
            using Types = hybrid_fft::FftTypes<
                Length, ElementsPerThread, FftsPerBlock
            >;
            using Forward = typename Types::Forward;
            using Inverse = typename Types::Inverse;
            constexpr auto update = update_history_kernel<
                Length, ElementsPerThread, FftsPerBlock
            >;
            constexpr std::size_t shared = std::max(
                static_cast<std::size_t>(Forward::shared_memory_size),
                static_cast<std::size_t>(Inverse::shared_memory_size)
            );
            update<<<static_cast<unsigned int>((
                (path_count + 1U) / 2U + FftsPerBlock - 1U)
                / FftsPerBlock), Forward::block_dim, shared>>>(
                spectra, 4U * steps, spectrum_offset, history,
                1U, path_count, path_count, steps, boundary, scale
            );
            check_cuda(cudaGetLastError(),
                "causal volatility history update");
        });
    }
}

template<typename Policy, std::size_t Dimension>
void launch_volatility_factor_cuda(
    const typename Policy::Parameters* models,
    const std::uint32_t* scenario_model_indices,
    const monte_carlo::MultiAssetPriceRow* price_rows,
    const float* packed_factors,
    std::size_t result_index,
    std::size_t path_offset,
    std::size_t path_count,
    float dt,
    std::uint32_t steps,
    std::uint32_t direct_tile_size,
    std::uint64_t base_seed,
    void* workspace,
    std::size_t workspace_bytes,
    float* volatility_path,
    std::size_t volatility_factor,
    std::uint32_t auxiliary_source,
    std::size_t scenario_stride = 1U,
    std::size_t scenario_slot = 0U
) {
    if (!volatility_path || volatility_factor >= Dimension
        || auxiliary_source < 2U || auxiliary_source > 255U) {
        throw std::invalid_argument("invalid causal volatility factor");
    }
    launch_causal_engine_cuda<
        Policy, Dimension, VolatilityContext<Policy>
    >(models, scenario_model_indices, price_rows, packed_factors,
        result_index, path_offset, path_count, dt, steps, direct_tile_size,
        base_seed, workspace, workspace_bytes,
        [=](const VolatilityRow<Policy>* row,
            std::size_t offset, std::size_t count,
            std::uint32_t total_steps, std::uint32_t first,
            std::uint32_t tile, const float* weights,
            float* history, VolatilityContext<Policy>* contexts) {
            advance_volatility_tile_kernel<Policy>
                <<<static_cast<unsigned int>((count + 127U) / 128U), 128U>>>(
                row, offset, count, total_steps, first, tile,
                weights, history, contexts, volatility_path,
                volatility_factor, auxiliary_source
            );
            check_cuda(cudaGetLastError(), "causal volatility direct tile");
        }, scenario_stride, scenario_slot
    );
}

template<typename Policy, std::size_t Dimension,
         std::size_t VolatilityFactor, std::uint32_t AuxiliarySource>
void launch_volatility_cuda(
    const typename Policy::Parameters* models,
    const std::uint32_t* scenario_model_indices,
    const monte_carlo::MultiAssetPriceRow* price_rows,
    const float* packed_factors,
    std::size_t result_index,
    std::size_t path_offset,
    std::size_t path_count,
    float dt,
    std::uint32_t steps,
    std::uint32_t direct_tile_size,
    std::uint64_t base_seed,
    void* workspace,
    std::size_t workspace_bytes,
    float* volatility_path,
    std::size_t scenario_stride = 1U,
    std::size_t scenario_slot = 0U
) {
    static_assert(VolatilityFactor < Dimension);
    static_assert(AuxiliarySource >= 2U && AuxiliarySource <= 255U);
    launch_volatility_factor_cuda<Policy, Dimension>(
        models, scenario_model_indices, price_rows, packed_factors,
        result_index, path_offset, path_count, dt, steps, direct_tile_size,
        base_seed, workspace, workspace_bytes, volatility_path,
        VolatilityFactor, AuxiliarySource, scenario_stride, scenario_slot
    );
}

}  // namespace ai_factory::workbench::volterra::causal_fft
