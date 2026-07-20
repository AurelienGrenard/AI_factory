#include "ai_factory/cuda/rough_bergomi/api.cuh"
#include "ai_factory/cuda/common/runtime.cuh"
#include "ai_factory/cuda/rough_bergomi/dynamics.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>

namespace ai_factory::cuda {
namespace {

using detail::check_cuda;
using detail::kThreadsPerBlock;
using rough_bergomi_detail::kMaxRoughBergomiSteps;
using rough_bergomi_detail::rough_bergomi_optimal_weight;
using rough_bergomi_detail::simulate_rough_bergomi_spot_path;

template <std::size_t MaxSteps>
__global__ void rough_bergomi_spot_paths_kernel(
    const RoughBergomiRow* row_ptr,
    std::size_t num_paths,
    std::size_t num_steps,
    double* spot_paths
) {
    const auto path =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto row = *row_ptr;
    const double dt = row.maturity / static_cast<double>(num_steps);
    __shared__ double weights[MaxSteps + 1U];
    __shared__ double variance_time_powers[MaxSteps];
    for (std::size_t step = threadIdx.x; step < num_steps; step += blockDim.x) {
        if (step >= 2U) {
            weights[step] = rough_bergomi_optimal_weight(row.alpha, dt, step);
        } else {
            weights[step] = 0.0;
        }
        const double time = static_cast<double>(step) * dt;
        variance_time_powers[step] = pow(time, 2.0 * row.alpha + 1.0);
    }
    __syncthreads();

    if (path >= num_paths) {
        return;
    }
    simulate_rough_bergomi_spot_path<MaxSteps>(
        row,
        num_steps,
        path,
        weights,
        variance_time_powers,
        spot_paths
    );
}

template <std::size_t MaxSteps>
void launch_rough_bergomi_spot_paths_kernel(
    const RoughBergomiRow* device_row,
    std::size_t num_paths,
    std::size_t num_steps,
    double* device_spot_paths
) {
    const auto block_count =
        static_cast<unsigned int>((num_paths + kThreadsPerBlock - 1U)
                                  / kThreadsPerBlock);
    rough_bergomi_spot_paths_kernel<MaxSteps><<<
        block_count,
        kThreadsPerBlock
    >>>(
        device_row,
        num_paths,
        num_steps,
        device_spot_paths
    );
}

void dispatch_rough_bergomi_spot_paths_kernel(
    const RoughBergomiRow* device_row,
    std::size_t num_paths,
    std::size_t num_steps,
    double* device_spot_paths
) {
    if (num_steps <= 16U) {
        launch_rough_bergomi_spot_paths_kernel<16U>(
            device_row,
            num_paths,
            num_steps,
            device_spot_paths
        );
    } else if (num_steps <= 32U) {
        launch_rough_bergomi_spot_paths_kernel<32U>(
            device_row,
            num_paths,
            num_steps,
            device_spot_paths
        );
    } else if (num_steps <= 64U) {
        launch_rough_bergomi_spot_paths_kernel<64U>(
            device_row,
            num_paths,
            num_steps,
            device_spot_paths
        );
    } else if (num_steps <= 128U) {
        launch_rough_bergomi_spot_paths_kernel<128U>(
            device_row,
            num_paths,
            num_steps,
            device_spot_paths
        );
    } else if (num_steps <= 160U) {
        launch_rough_bergomi_spot_paths_kernel<160U>(
            device_row,
            num_paths,
            num_steps,
            device_spot_paths
        );
    } else if (num_steps <= kMaxRoughBergomiSteps) {
        launch_rough_bergomi_spot_paths_kernel<kMaxRoughBergomiSteps>(
            device_row,
            num_paths,
            num_steps,
            device_spot_paths
        );
    } else {
        throw std::invalid_argument(
            "Rough Bergomi CUDA path export supports at most 272 steps."
        );
    }
    check_cuda(cudaGetLastError(), "rough Bergomi spot-path kernel launch");
}

}  // namespace

void generate_rough_bergomi_spot_paths_cuda(
    const RoughBergomiRow* host_row,
    std::size_t num_paths,
    std::size_t num_steps,
    double* host_spot_paths,
    CudaTiming* timing
) {
    RoughBergomiRow* device_row = nullptr;
    double* device_spot_paths = nullptr;
    cudaEvent_t start{};
    cudaEvent_t stop{};
    const auto output_bytes = num_paths * (num_steps + 1U) * sizeof(double);

    check_cuda(cudaMalloc(&device_row, sizeof(RoughBergomiRow)), "cudaMalloc row");
    check_cuda(
        cudaMalloc(&device_spot_paths, output_bytes),
        "cudaMalloc rough spot paths"
    );
    try {
        check_cuda(
            cudaMemcpy(
                device_row,
                host_row,
                sizeof(RoughBergomiRow),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy rough row"
        );
        check_cuda(cudaEventCreate(&start), "cudaEventCreate rough path start");
        check_cuda(cudaEventCreate(&stop), "cudaEventCreate rough path stop");
        check_cuda(cudaEventRecord(start), "cudaEventRecord rough path start");

        dispatch_rough_bergomi_spot_paths_kernel(
            device_row,
            num_paths,
            num_steps,
            device_spot_paths
        );
        check_cuda(cudaEventRecord(stop), "cudaEventRecord rough path stop");
        check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize rough path stop");

        float elapsed_ms = 0.0F;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "rough path timing");
        check_cuda(
            cudaMemcpy(
                host_spot_paths,
                device_spot_paths,
                output_bytes,
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy rough spot paths"
        );
        if (timing != nullptr) {
            timing->simulation_ms = elapsed_ms;
            timing->total_ms = elapsed_ms;
        }
    } catch (...) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(device_row);
        cudaFree(device_spot_paths);
        throw;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(device_row);
    cudaFree(device_spot_paths);
}

}  // namespace ai_factory::cuda
