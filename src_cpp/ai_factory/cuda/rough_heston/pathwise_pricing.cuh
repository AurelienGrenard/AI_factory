#pragma once

#include "ai_factory/cuda/common/reductions.cuh"
#include "ai_factory/cuda/common/runtime.cuh"
#include "ai_factory/cuda/rough_heston/dynamics.cuh"

#include <cuda_runtime.h>

#include <cmath>

namespace ai_factory::cuda::rough_heston_pathwise {

enum class Statistic { Maximum, Average, RealizedVolatility };

template <Statistic statistic>
struct Observer {
    double value;
    double previous_spot;
    __device__ explicit Observer(double spot)
        : value(statistic == Statistic::Maximum ? spot : 0.0),
          previous_spot(spot) {}
    __device__ void operator()(std::size_t, double spot, double) {
        if constexpr (statistic == Statistic::Maximum) {
            value = fmax(value, spot);
        } else if constexpr (statistic == Statistic::Average) {
            value += spot;
        } else {
            const double log_return = log(spot / previous_spot);
            value += log_return * log_return;
            previous_spot = spot;
        }
    }
    __device__ double finish(std::size_t num_steps) const {
        if constexpr (statistic == Statistic::Average) {
            return value / static_cast<double>(num_steps);
        } else if constexpr (statistic == Statistic::RealizedVolatility) {
            return sqrt(52.0 / static_cast<double>(num_steps) * value);
        }
        return value;
    }
};

template <Statistic statistic, bool linear>
__device__ double payoff(double state, double strike) {
    if constexpr (linear) return state - strike;
    return fmax(state - strike, 0.0);
}

template <Statistic statistic, bool linear, bool delta>
__global__ void partial_kernel(
    const RoughHestonRow* rows, std::size_t row_count,
    std::size_t num_paths, std::size_t num_steps,
    std::size_t blocks_per_row, double relative_bump,
    double* first, double* second, double* third, double* fourth
) {
    const auto row_index = static_cast<std::size_t>(blockIdx.x) / blocks_per_row;
    const auto block = static_cast<std::size_t>(blockIdx.x) % blocks_per_row;
    if (row_index >= row_count) return;
    const auto row = rows[row_index];
    const auto path = block * blockDim.x + threadIdx.x;
    double price = 0.0, price2 = 0.0, path_delta = 0.0, delta2 = 0.0;
    if (path < num_paths) {
        Observer<statistic> observer(row.spot);
        rough_heston_detail::simulate(row, path, num_steps, observer);
        const double state = observer.finish(num_steps);
        const double discount = exp(-row.risk_free_rate * row.maturity);
        price = discount * payoff<statistic, linear>(state, row.strike);
        price2 = price * price;
        if constexpr (delta) {
            const double up = discount * payoff<statistic, linear>(
                state * (1.0 + relative_bump), row.strike
            );
            const double down = discount * payoff<statistic, linear>(
                state * (1.0 - relative_bump), row.strike
            );
            path_delta = (up - down) / (2.0 * relative_bump * row.spot);
            delta2 = path_delta * path_delta;
        }
    }
    if constexpr (delta) {
        reductions::reduce_block_four(price, price2, path_delta, delta2);
    } else {
        reductions::reduce_block(price, price2);
    }
    if (threadIdx.x == 0) {
        extern __shared__ double shared[];
        const auto index = row_index * blocks_per_row + block;
        first[index] = shared[0];
        second[index] = shared[blockDim.x];
        if constexpr (delta) {
            third[index] = shared[2 * blockDim.x];
            fourth[index] = shared[3 * blockDim.x];
        }
    }
}

template <bool delta>
__global__ void finalize_kernel(
    const double* first, const double* second,
    const double* third, const double* fourth,
    std::size_t row_count, std::size_t blocks_per_row,
    std::size_t num_paths, void* raw_outputs
) {
    const auto row = static_cast<std::size_t>(blockIdx.x);
    if (row >= row_count) return;
    double a = 0.0, b = 0.0, c = 0.0, d = 0.0;
    for (std::size_t block = threadIdx.x; block < blocks_per_row; block += blockDim.x) {
        const auto index = row * blocks_per_row + block;
        a += first[index]; b += second[index];
        if constexpr (delta) { c += third[index]; d += fourth[index]; }
    }
    if constexpr (delta) reductions::reduce_block_four(a, b, c, d);
    else reductions::reduce_block(a, b);
    if (threadIdx.x == 0) {
        extern __shared__ double shared[];
        const double count = static_cast<double>(num_paths);
        const double mean = shared[0] / count;
        const double variance = (shared[blockDim.x] - count * mean * mean)
            / static_cast<double>(num_paths - 1U);
        if constexpr (delta) {
            const double delta_mean = shared[2 * blockDim.x] / count;
            const double delta_variance =
                (shared[3 * blockDim.x] - count * delta_mean * delta_mean)
                / static_cast<double>(num_paths - 1U);
            static_cast<PriceDeltaOutput*>(raw_outputs)[row] = {
                mean, sqrt(fmax(variance, 0.0) / count),
                delta_mean, sqrt(fmax(delta_variance, 0.0) / count)
            };
        } else {
            static_cast<MonteCarloOutput*>(raw_outputs)[row] = {
                mean, sqrt(fmax(variance, 0.0) / count)
            };
        }
    }
}

template <typename Tag, Statistic statistic, bool linear, bool delta>
void run(
    const RoughHestonRow* host_rows, std::size_t row_count,
    std::size_t num_paths, std::size_t num_steps, double relative_bump,
    void* host_outputs, CudaTiming* timing
) {
    constexpr auto threads = detail::kThreadsPerBlock;
    const auto blocks = (num_paths + threads - 1U) / threads;
    const auto partial_count = row_count * blocks;
    constexpr std::size_t buffers = delta ? 4U : 2U;
    auto& workspace = detail::reusable_cuda_workspace<Tag, 3U>();
    auto* rows = workspace.template buffer<RoughHestonRow>(0U, row_count, "rough Heston rows");
    const auto output_bytes = row_count * (delta ? sizeof(PriceDeltaOutput) : sizeof(MonteCarloOutput));
    auto* outputs = workspace.template buffer<unsigned char>(1U, output_bytes, "rough Heston outputs");
    auto* partials = workspace.template buffer<double>(2U, buffers * partial_count, "rough Heston partials");
    detail::check_cuda(cudaMemcpy(rows, host_rows, row_count * sizeof(RoughHestonRow), cudaMemcpyHostToDevice), "copy rough Heston rows");
    auto start = workspace.start_event(); auto stop = workspace.stop_event();
    detail::check_cuda(cudaEventRecord(start), "rough Heston start");
    partial_kernel<statistic, linear, delta><<<
        static_cast<unsigned int>(partial_count), threads,
        buffers * threads * sizeof(double)
    >>>(rows,row_count,num_paths,num_steps,blocks,relative_bump,partials,partials+partial_count,
        delta ? partials+2*partial_count : nullptr,
        delta ? partials+3*partial_count : nullptr);
    detail::check_cuda(cudaGetLastError(), "rough Heston partial kernel");
    finalize_kernel<delta><<<static_cast<unsigned int>(row_count),threads,buffers*threads*sizeof(double)>>>(
        partials,partials+partial_count,
        delta ? partials+2*partial_count : nullptr,
        delta ? partials+3*partial_count : nullptr,
        row_count,blocks,num_paths,outputs);
    detail::check_cuda(cudaGetLastError(), "rough Heston finalize kernel");
    detail::check_cuda(cudaEventRecord(stop), "rough Heston stop");
    detail::check_cuda(cudaEventSynchronize(stop), "rough Heston synchronize");
    float milliseconds = 0.0F;
    detail::check_cuda(cudaEventElapsedTime(&milliseconds,start,stop), "rough Heston timing");
    detail::check_cuda(cudaMemcpy(host_outputs,outputs,output_bytes,cudaMemcpyDeviceToHost), "copy rough Heston outputs");
    if (timing) *timing = {milliseconds,milliseconds};
}

}  // namespace ai_factory::cuda::rough_heston_pathwise
