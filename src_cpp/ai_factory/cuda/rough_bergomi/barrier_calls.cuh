#pragma once

#include "ai_factory/cuda/common/barrier_pricing.cuh"
#include "ai_factory/cuda/common/reductions.cuh"
#include "ai_factory/cuda/common/runtime.cuh"
#include "ai_factory/cuda/rough_bergomi/dynamics.cuh"

#include <cuda_runtime.h>

#include <stdexcept>

namespace ai_factory::cuda::rough_bergomi_barrier_detail {

constexpr int kThreads = 128;

template <std::size_t MaxSteps, bool Up, bool KnockIn>
__global__ void partial_kernel(
    const RoughBergomiBarrierRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    std::size_t path_blocks_per_row,
    double* partial_sums,
    double* partial_sumsq
) {
    const auto row_index =
        static_cast<std::size_t>(blockIdx.x) / path_blocks_per_row;
    const auto path_block =
        static_cast<std::size_t>(blockIdx.x) % path_blocks_per_row;
    if (row_index >= row_count) {
        return;
    }
    const auto row = rows[row_index];
    const auto path = path_block * static_cast<std::size_t>(blockDim.x)
                      + static_cast<std::size_t>(threadIdx.x);
    const double dt = row.model.maturity / static_cast<double>(num_steps);
    __shared__ double weights[MaxSteps + 1U];
    __shared__ double variance_time_powers[MaxSteps];
    for (std::size_t step = threadIdx.x; step < num_steps; step += blockDim.x) {
        weights[step] = step >= 2U
            ? rough_bergomi_detail::rough_bergomi_optimal_weight(
                  row.model.alpha, dt, step
              )
            : 0.0;
        const double time = static_cast<double>(step) * dt;
        variance_time_powers[step] =
            pow(time, 2.0 * row.model.alpha + 1.0);
    }
    __syncthreads();

    double payoff = 0.0;
    if (path < num_paths) {
        double dws[MaxSteps];
        const double sqrt_dt = sqrt(dt);
        rough_bergomi_detail::prepare_rough_bergomi_path_normals<MaxSteps>(
            row.model, num_steps, path, sqrt_dt, dws
        );
        const auto first_index = path * num_steps;
        rng::NormalSequence residual_normals(row.model.seed, 1U, first_index);
        rng::NormalSequence stock_normals(row.model.seed, 2U, first_index);
        const double drift =
            (row.model.risk_free_rate - row.model.dividend_yield) * dt;
        const double rho_perp = sqrt(1.0 - row.model.rho * row.model.rho);
        double log_spot = log(row.model.spot);
        bool hit = false;
        for (std::size_t step = 0U; step < num_steps; ++step) {
            const double y =
                rough_bergomi_detail::rough_bergomi_volterra_state<MaxSteps>(
                    row.model,
                    num_steps,
                    path,
                    step,
                    dt,
                    sqrt_dt,
                    step == 0U ? 0.0 : residual_normals.next(),
                    weights,
                    dws
                );
            const double variance =
                rough_bergomi_detail::rough_bergomi_variance(
                    row.model, y, variance_time_powers[step]
                );
            const double dz = row.model.rho * dws[step]
                              + rho_perp * sqrt_dt * stock_normals.next();
            log_spot += drift - 0.5 * variance * dt + sqrt(variance) * dz;
            const double spot = exp(log_spot);
            hit = hit || (Up ? spot >= row.product.barrier
                             : spot <= row.product.barrier);
        }
        const bool active = KnockIn ? hit : !hit;
        if (active) {
            payoff = exp(-row.model.risk_free_rate * row.model.maturity)
                     * fmax(exp(log_spot) - row.model.strike, 0.0);
        }
    }
    double sum = payoff;
    double sumsq = payoff * payoff;
    reductions::reduce_block(sum, sumsq);
    if (threadIdx.x == 0) {
        extern __shared__ double shared[];
        const auto partial_index = row_index * path_blocks_per_row + path_block;
        partial_sums[partial_index] = shared[0];
        partial_sumsq[partial_index] = shared[blockDim.x];
    }
}

template <std::size_t MaxSteps, bool Up, bool KnockIn>
void launch_partial(
    const RoughBergomiBarrierRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    std::size_t path_blocks_per_row,
    double* partial_sums,
    double* partial_sumsq,
    std::size_t shared_bytes
) {
    partial_kernel<MaxSteps, Up, KnockIn><<<
        static_cast<unsigned int>(row_count * path_blocks_per_row),
        kThreads,
        shared_bytes
    >>>(
        rows,
        row_count,
        num_paths,
        num_steps,
        path_blocks_per_row,
        partial_sums,
        partial_sumsq
    );
}

template <bool Up, bool KnockIn>
void dispatch_partial(
    const RoughBergomiBarrierRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    std::size_t path_blocks_per_row,
    double* partial_sums,
    double* partial_sumsq,
    std::size_t shared_bytes
) {
    if (num_steps <= 128U) {
        launch_partial<128U, Up, KnockIn>(rows, row_count, num_paths, num_steps, path_blocks_per_row, partial_sums, partial_sumsq, shared_bytes);
    } else if (num_steps <= 160U) {
        launch_partial<160U, Up, KnockIn>(rows, row_count, num_paths, num_steps, path_blocks_per_row, partial_sums, partial_sumsq, shared_bytes);
    } else if (num_steps <= 208U) {
        launch_partial<208U, Up, KnockIn>(rows, row_count, num_paths, num_steps, path_blocks_per_row, partial_sums, partial_sumsq, shared_bytes);
    } else if (num_steps <= rough_bergomi_detail::kMaxRoughBergomiSteps) {
        launch_partial<rough_bergomi_detail::kMaxRoughBergomiSteps, Up, KnockIn>(
            rows, row_count, num_paths, num_steps, path_blocks_per_row,
            partial_sums, partial_sumsq, shared_bytes
        );
    } else {
        throw std::invalid_argument(
            "Optimized Rough Bergomi barrier kernels support at most 272 steps."
        );
    }
}

template <bool Up, bool KnockIn, typename Tag>
void run(
    const RoughBergomiBarrierRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    MonteCarloOutput* host_outputs,
    CudaTiming* timing
) {
    if (row_count == 0U || num_paths < 2U || num_steps == 0U) {
        throw std::invalid_argument(
            "Barrier pricing requires rows, steps, and at least two paths."
        );
    }
    const auto path_blocks_per_row =
        (num_paths + static_cast<std::size_t>(kThreads) - 1U)
        / static_cast<std::size_t>(kThreads);
    const auto partial_count = row_count * path_blocks_per_row;
    auto& workspace = detail::reusable_cuda_workspace<Tag, 3U>();
    auto* device_rows = workspace.template buffer<RoughBergomiBarrierRow>(
        0U, row_count, "cudaMalloc rough barrier rows"
    );
    auto* device_outputs = workspace.template buffer<MonteCarloOutput>(
        1U, row_count, "cudaMalloc rough barrier outputs"
    );
    auto* partials = workspace.template buffer<double>(
        2U, 2U * partial_count, "cudaMalloc rough barrier partials"
    );
    detail::check_cuda(
        cudaMemcpy(device_rows, host_rows,
                   row_count * sizeof(RoughBergomiBarrierRow),
                   cudaMemcpyHostToDevice),
        "cudaMemcpy rough barrier rows"
    );
    const auto start = workspace.start_event();
    const auto stop = workspace.stop_event();
    detail::check_cuda(cudaEventRecord(start), "rough barrier start");
    const auto model_shared_bytes =
        (rough_bergomi_detail::kMaxRoughBergomiSteps + 1U
         + rough_bergomi_detail::kMaxRoughBergomiSteps)
        * sizeof(double);
    const auto reduction_shared_bytes = 2U * kThreads * sizeof(double);
    const auto shared_bytes =
        model_shared_bytes > reduction_shared_bytes
            ? model_shared_bytes
            : reduction_shared_bytes;
    dispatch_partial<Up, KnockIn>(
        device_rows,
        row_count,
        num_paths,
        num_steps,
        path_blocks_per_row,
        partials,
        partials + partial_count,
        shared_bytes
    );
    detail::check_cuda(cudaGetLastError(), "rough barrier partial kernel");
    barrier_detail::finalize_kernel<Tag><<<
        static_cast<unsigned int>(row_count),
        kThreads,
        reduction_shared_bytes
    >>>(
        partials,
        partials + partial_count,
        row_count,
        path_blocks_per_row,
        num_paths,
        device_outputs
    );
    detail::check_cuda(cudaGetLastError(), "rough barrier finalize kernel");
    detail::check_cuda(cudaEventRecord(stop), "rough barrier stop");
    detail::check_cuda(cudaEventSynchronize(stop), "rough barrier synchronize");
    float elapsed = 0.0F;
    detail::check_cuda(cudaEventElapsedTime(&elapsed, start, stop), "rough barrier timing");
    detail::check_cuda(
        cudaMemcpy(host_outputs, device_outputs,
                   row_count * sizeof(MonteCarloOutput),
                   cudaMemcpyDeviceToHost),
        "cudaMemcpy rough barrier outputs"
    );
    if (timing != nullptr) {
        *timing = {elapsed, elapsed};
    }
}

}  // namespace ai_factory::cuda::rough_bergomi_barrier_detail
