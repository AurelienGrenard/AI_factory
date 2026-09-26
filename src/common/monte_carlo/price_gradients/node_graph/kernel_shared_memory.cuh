// Portable opt-in shared-memory validation for node-graph kernels.
#pragma once

#include "common/check_cuda.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace ai_factory::workbench::monte_carlo::price_gradients {

template<typename Kernel>
void configure_node_graph_dynamic_shared_memory(
    Kernel kernel,
    std::size_t dynamic_bytes,
    const char* attribute_error,
    const char* device_error,
    const char* properties_error,
    const char* capacity_error,
    const char* opt_in_error
) {
    cudaFuncAttributes attributes{};
    check_cuda(
        cudaFuncGetAttributes(&attributes, kernel), attribute_error
    );
    int device = 0;
    check_cuda(cudaGetDevice(&device), device_error);
    cudaDeviceProp properties{};
    check_cuda(
        cudaGetDeviceProperties(&properties, device), properties_error
    );
    const auto total_bytes = dynamic_bytes + attributes.sharedSizeBytes;
    if (total_bytes > properties.sharedMemPerBlockOptin
        || dynamic_bytes
            > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(capacity_error);
    }
    if (total_bytes > properties.sharedMemPerBlock) {
        check_cuda(
            cudaFuncSetAttribute(
                kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(dynamic_bytes)
            ),
            opt_in_error
        );
    }
}

}  // namespace ai_factory::workbench::monte_carlo::price_gradients
