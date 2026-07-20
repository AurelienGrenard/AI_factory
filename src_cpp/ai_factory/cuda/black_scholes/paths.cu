#include "ai_factory/cuda/black_scholes/api.cuh"

#include "ai_factory/cuda/black_scholes/dynamics.cuh"
#include "ai_factory/cuda/common/runtime.cuh"

#include <cuda_runtime.h>

#include <cmath>

namespace ai_factory::cuda {
namespace {

using rng::standard_normal;
using detail::check_cuda;
using detail::kThreadsPerBlock;
using black_scholes_detail::log_step;

__global__ void spot_paths_kernel(
    const BlackScholesRow* row_ptr,
    std::size_t num_paths,
    std::size_t num_steps,
    double* paths
) {
    const auto path =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (path >= num_paths) {
        return;
    }
    const auto row = *row_ptr;
    const double dt = row.maturity / static_cast<double>(num_steps);
    const double sqrt_dt = sqrt(dt);
    const std::size_t stride = num_steps + 1U;
    rng::NormalSequence normals(row.seed, 0U, path * num_steps);
    double spot = row.spot;
    paths[path * stride] = spot;
    for (std::size_t step = 0; step < num_steps; ++step) {
        spot *= exp(log_step(row, dt, sqrt_dt, normals.next()));
        paths[path * stride + step + 1U] = spot;
    }
}

}  // namespace

void generate_black_scholes_spot_paths_cuda(
    const BlackScholesRow* host_row,
    std::size_t num_paths,
    std::size_t num_steps,
    double* host_spot_paths,
    CudaTiming* timing
) {
    BlackScholesRow* device_row = nullptr;
    double* device_paths = nullptr;
    cudaEvent_t start{};
    cudaEvent_t stop{};
    const auto output_bytes = num_paths * (num_steps + 1U) * sizeof(double);
    check_cuda(cudaMalloc(&device_row, sizeof(BlackScholesRow)), "cudaMalloc row");
    check_cuda(cudaMalloc(&device_paths, output_bytes), "cudaMalloc paths");
    check_cuda(cudaMemcpy(device_row, host_row, sizeof(BlackScholesRow), cudaMemcpyHostToDevice), "cudaMemcpy row");
    check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");
    check_cuda(cudaEventRecord(start), "cudaEventRecord start");
    const auto blocks = static_cast<unsigned int>((num_paths + kThreadsPerBlock - 1U) / kThreadsPerBlock);
    spot_paths_kernel<<<blocks, kThreadsPerBlock>>>(device_row, num_paths, num_steps, device_paths);
    check_cuda(cudaGetLastError(), "black scholes paths kernel");
    check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
    float elapsed_ms = 0.0F;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "kernel timing");
    check_cuda(cudaMemcpy(host_spot_paths, device_paths, output_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy paths");
    if (timing != nullptr) {
        timing->simulation_ms = elapsed_ms;
        timing->total_ms = elapsed_ms;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(device_row);
    cudaFree(device_paths);
}

}  // namespace ai_factory::cuda
