// Optional host-side diagnostics for one concrete CUDA kernel launch.
#pragma once

#include "common/check_cuda.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench {

// Resources and theoretical occupancy for one kernel and launch geometry.
struct CudaKernelLaunchDiagnostics {
    int device_index;
    std::string device_name;
    int compute_capability_major;
    int compute_capability_minor;
    std::uint64_t grid_block_count;
    unsigned int grid_x;
    unsigned int grid_y;
    unsigned int grid_z;
    unsigned int threads_per_block;
    unsigned int block_x;
    unsigned int block_y;
    unsigned int block_z;
    std::size_t registers_per_thread;
    std::size_t static_shared_bytes_per_block;
    std::size_t dynamic_shared_bytes_per_block;
    std::size_t local_bytes_per_thread;
    int active_blocks_per_multiprocessor;
    int active_warps_per_multiprocessor;
    int maximum_warps_per_multiprocessor;
    double theoretical_occupancy;
    int maximum_threads_per_block;
    std::size_t maximum_dynamic_shared_bytes_per_block;
    int binary_version;
    int ptx_version;
};

// Diagnostics are opt-in so normal generators keep their usual output.
bool cuda_kernel_diagnostics_enabled() noexcept;

// Reserve a unique kernel, variant, and geometry before querying the GPU.
bool reserve_cuda_kernel_launch_diagnostics(
    const char* kernel_name,
    const char* variant,
    dim3 grid,
    dim3 block,
    std::size_t dynamic_shared_bytes
);

// Emit one JSON object for a launch reserved by the caller.
void emit_cuda_kernel_launch_diagnostics(
    const char* kernel_name,
    const char* variant,
    const CudaKernelLaunchDiagnostics& diagnostics
);

// Inspect the exact specialization and geometry about to be launched.
template<typename Kernel>
CudaKernelLaunchDiagnostics inspect_cuda_kernel_launch(
    Kernel kernel,
    dim3 grid,
    dim3 block,
    std::size_t dynamic_shared_bytes
) {
    const std::uint64_t threads_per_block =
        static_cast<std::uint64_t>(block.x)
        * static_cast<std::uint64_t>(block.y)
        * static_cast<std::uint64_t>(block.z);
    if (threads_per_block == 0U
        || threads_per_block
            > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "CUDA kernel diagnostics require a valid block geometry."
        );
    }

    cudaFuncAttributes attributes{};
    check_cuda(
        cudaFuncGetAttributes(&attributes, kernel),
        "cudaFuncGetAttributes for kernel diagnostics"
    );

    int active_blocks = 0;
    check_cuda(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks,
            kernel,
            static_cast<int>(threads_per_block),
            dynamic_shared_bytes
        ),
        "cudaOccupancyMaxActiveBlocksPerMultiprocessor for kernel diagnostics"
    );

    int device_index = 0;
    check_cuda(cudaGetDevice(&device_index), "cudaGetDevice for diagnostics");
    cudaDeviceProp properties{};
    check_cuda(
        cudaGetDeviceProperties(&properties, device_index),
        "cudaGetDeviceProperties for diagnostics"
    );

    const int warps_per_block = static_cast<int>(
        (threads_per_block + static_cast<std::uint64_t>(properties.warpSize) - 1U)
        / static_cast<std::uint64_t>(properties.warpSize)
    );
    const int maximum_warps =
        properties.maxThreadsPerMultiProcessor / properties.warpSize;
    const int active_warps = active_blocks * warps_per_block;
    const double theoretical_occupancy = maximum_warps == 0
        ? 0.0
        : static_cast<double>(active_warps)
            / static_cast<double>(maximum_warps);
    const std::uint64_t grid_block_count =
        static_cast<std::uint64_t>(grid.x)
        * static_cast<std::uint64_t>(grid.y)
        * static_cast<std::uint64_t>(grid.z);

    return {
        device_index,
        properties.name,
        properties.major,
        properties.minor,
        grid_block_count,
        grid.x,
        grid.y,
        grid.z,
        static_cast<unsigned int>(threads_per_block),
        block.x,
        block.y,
        block.z,
        static_cast<std::size_t>(attributes.numRegs),
        attributes.sharedSizeBytes,
        dynamic_shared_bytes,
        attributes.localSizeBytes,
        active_blocks,
        active_warps,
        maximum_warps,
        theoretical_occupancy,
        attributes.maxThreadsPerBlock,
        static_cast<std::size_t>(attributes.maxDynamicSharedSizeBytes),
        attributes.binaryVersion,
        attributes.ptxVersion,
    };
}

// Keep diagnostics entirely out of the normal path unless explicitly enabled.
template<typename Kernel>
void report_cuda_kernel_launch_if_enabled(
    const char* kernel_name,
    const char* variant,
    Kernel kernel,
    dim3 grid,
    dim3 block,
    std::size_t dynamic_shared_bytes = 0U
) {
    if (!cuda_kernel_diagnostics_enabled()
        || !reserve_cuda_kernel_launch_diagnostics(
            kernel_name,
            variant,
            grid,
            block,
            dynamic_shared_bytes
        )) {
        return;
    }
    emit_cuda_kernel_launch_diagnostics(
        kernel_name,
        variant,
        inspect_cuda_kernel_launch(
            kernel, grid, block, dynamic_shared_bytes
        )
    );
}

}  // namespace ai_factory::workbench
