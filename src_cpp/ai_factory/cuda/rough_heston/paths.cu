#include "ai_factory/cuda/rough_heston/api.cuh"

#include "ai_factory/cuda/common/runtime.cuh"
#include "ai_factory/cuda/rough_heston/dynamics.cuh"

#include <cuda_runtime.h>

namespace ai_factory::cuda {
namespace {

struct RoughHestonPathWorkspaceTag {};

__global__ void rough_heston_spot_paths_kernel(
    const RoughHestonRow* row_ptr,
    std::size_t num_paths,
    std::size_t num_steps,
    double* paths,
    double* factor_paths
) {
    const auto path = static_cast<std::size_t>(blockIdx.x) * blockDim.x
                    + threadIdx.x;
    if (path >= num_paths) return;
    const auto row = *row_ptr;
    const auto stride = num_steps + 1U;
    auto* values = paths + path * stride;
    values[0] = row.spot;
    auto* path_factors = factor_paths == nullptr
        ? nullptr
        : factor_paths + path * stride * rough_heston_detail::kFactorCount;
    if (path_factors != nullptr) {
        #pragma unroll
        for (std::size_t factor = 0;
             factor < rough_heston_detail::kFactorCount;
             ++factor) {
            path_factors[factor] = 0.0;
        }
    }
    struct StorePath {
        double* values;
        double* factors;
        __device__ void operator()(
            std::size_t step, double spot, double, const double* state_factors
        ) const {
            const auto date = step + 1U;
            values[date] = spot;
            if (factors != nullptr) {
                #pragma unroll
                for (std::size_t factor = 0;
                     factor < rough_heston_detail::kFactorCount;
                     ++factor) {
                    factors[date * rough_heston_detail::kFactorCount + factor] =
                        state_factors[factor];
                }
            }
        }
    } observer{values, path_factors};
    rough_heston_detail::simulate_state_path(row, path, num_steps, observer);
}

}  // namespace

void generate_rough_heston_spot_paths_cuda(
    const RoughHestonRow* host_row,
    std::size_t num_paths,
    std::size_t num_steps,
    double* host_paths,
    CudaTiming* timing
) {
    using detail::check_cuda;
    auto& workspace = detail::reusable_cuda_workspace<
        RoughHestonPathWorkspaceTag, 3U
    >();
    auto* row = workspace.buffer<RoughHestonRow>(0U, 1U, "rough Heston path row");
    const auto value_count = num_paths * (num_steps + 1U);
    auto* paths = workspace.buffer<double>(1U, value_count, "rough Heston paths");
    check_cuda(
        cudaMemcpy(row, host_row, sizeof(RoughHestonRow), cudaMemcpyHostToDevice),
        "copy rough Heston path row"
    );
    const auto start = workspace.start_event();
    const auto stop = workspace.stop_event();
    check_cuda(cudaEventRecord(start), "rough Heston path start");
    const auto blocks = static_cast<unsigned int>(
        (num_paths + detail::kThreadsPerBlock - 1U) / detail::kThreadsPerBlock
    );
    rough_heston_spot_paths_kernel<<<blocks, detail::kThreadsPerBlock>>>(
        row, num_paths, num_steps, paths, nullptr
    );
    check_cuda(cudaGetLastError(), "rough Heston path kernel");
    check_cuda(cudaEventRecord(stop), "rough Heston path stop");
    check_cuda(cudaEventSynchronize(stop), "rough Heston path synchronize");
    float milliseconds = 0.0F;
    check_cuda(cudaEventElapsedTime(&milliseconds, start, stop), "rough Heston path timing");
    check_cuda(
        cudaMemcpy(
            host_paths, paths, value_count * sizeof(double), cudaMemcpyDeviceToHost
        ),
        "copy rough Heston paths"
    );
    if (timing) {
        timing->simulation_ms = milliseconds;
        timing->total_ms = milliseconds;
    }
}

void generate_rough_heston_state_paths_cuda(
    const RoughHestonRow* host_row,
    std::size_t num_paths,
    std::size_t num_steps,
    double* host_paths,
    double* host_factor_paths,
    CudaTiming* timing
) {
    using detail::check_cuda;
    auto& workspace = detail::reusable_cuda_workspace<
        RoughHestonPathWorkspaceTag, 3U
    >();
    auto* row = workspace.buffer<RoughHestonRow>(0U, 1U, "rough Heston state row");
    const auto value_count = num_paths * (num_steps + 1U);
    const auto factor_count = value_count * rough_heston_detail::kFactorCount;
    auto* paths = workspace.buffer<double>(1U, value_count, "rough Heston state spots");
    auto* factors = workspace.buffer<double>(
        2U, factor_count, "rough Heston state factors"
    );
    check_cuda(
        cudaMemcpy(row, host_row, sizeof(RoughHestonRow), cudaMemcpyHostToDevice),
        "copy rough Heston state row"
    );
    const auto start = workspace.start_event();
    const auto stop = workspace.stop_event();
    check_cuda(cudaEventRecord(start), "rough Heston state start");
    const auto blocks = static_cast<unsigned int>(
        (num_paths + detail::kThreadsPerBlock - 1U) / detail::kThreadsPerBlock
    );
    rough_heston_spot_paths_kernel<<<blocks, detail::kThreadsPerBlock>>>(
        row, num_paths, num_steps, paths, factors
    );
    check_cuda(cudaGetLastError(), "rough Heston state kernel");
    check_cuda(cudaEventRecord(stop), "rough Heston state stop");
    check_cuda(cudaEventSynchronize(stop), "rough Heston state synchronize");
    float milliseconds = 0.0F;
    check_cuda(cudaEventElapsedTime(&milliseconds, start, stop), "rough Heston state timing");
    check_cuda(
        cudaMemcpy(host_paths, paths, value_count * sizeof(double), cudaMemcpyDeviceToHost),
        "copy rough Heston state spots"
    );
    check_cuda(
        cudaMemcpy(
            host_factor_paths, factors, factor_count * sizeof(double),
            cudaMemcpyDeviceToHost
        ),
        "copy rough Heston state factors"
    );
    if (timing) {
        timing->simulation_ms = milliseconds;
        timing->total_ms = milliseconds;
    }
}

}  // namespace ai_factory::cuda
