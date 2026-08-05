// Runtime checks for CUDA kernel resource and occupancy inspection.
#include "common/cuda_kernel_diagnostics.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

__global__ void diagnostic_probe_kernel(float* values) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    values[index] = static_cast<float>(index);
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
    using namespace ai_factory::workbench;

    int device_count = 0;
    const cudaError_t device_status = cudaGetDeviceCount(&device_count);
    if (device_status != cudaSuccess || device_count == 0) {
        std::cerr << "CUDA device unavailable; skipping diagnostics test.\n";
        return 77;
    }

    constexpr unsigned int threads_per_block = 128U;
    constexpr unsigned int block_count = 2U;
    const CudaKernelLaunchDiagnostics diagnostics =
        inspect_cuda_kernel_launch(
            diagnostic_probe_kernel,
            dim3(block_count),
            dim3(threads_per_block),
            0U
        );

    require(
        diagnostics.grid_block_count == block_count,
        "Kernel diagnostics changed the grid block count."
    );
    require(
        diagnostics.grid_x == block_count
            && diagnostics.grid_y == 1U
            && diagnostics.grid_z == 1U,
        "Kernel diagnostics changed the grid geometry."
    );
    require(
        diagnostics.threads_per_block == threads_per_block,
        "Kernel diagnostics changed the block size."
    );
    require(
        diagnostics.block_x == threads_per_block
            && diagnostics.block_y == 1U
            && diagnostics.block_z == 1U,
        "Kernel diagnostics changed the block geometry."
    );
    require(
        diagnostics.active_blocks_per_multiprocessor > 0,
        "The probe kernel has no active block per multiprocessor."
    );
    require(
        diagnostics.maximum_warps_per_multiprocessor > 0,
        "The device exposes no resident warp capacity."
    );
    require(
        diagnostics.theoretical_occupancy > 0.0
            && diagnostics.theoretical_occupancy <= 1.0,
        "The reported theoretical occupancy is outside (0, 1]."
    );
    require(
        diagnostics.maximum_threads_per_block
            >= static_cast<int>(threads_per_block),
        "The inspected kernel rejects the test block size."
    );
    require(
        !diagnostics.device_name.empty(),
        "Kernel diagnostics did not report the CUDA device name."
    );
    require(
        reserve_cuda_kernel_launch_diagnostics(
            "diagnostic.probe",
            "default",
            dim3(block_count),
            dim3(threads_per_block),
            0U
        ),
        "The first diagnostic geometry was not reserved."
    );
    require(
        !reserve_cuda_kernel_launch_diagnostics(
            "diagnostic.probe",
            "default",
            dim3(block_count),
            dim3(threads_per_block),
            0U
        ),
        "A repeated diagnostic geometry was not deduplicated."
    );

    std::cout << "CUDA kernel diagnostics test passed.\n";
    return 0;
}
