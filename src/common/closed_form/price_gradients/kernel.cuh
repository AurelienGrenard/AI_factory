// Sequential scenario pairs per thread over an existing analytical pricing policy.
#pragma once
#include "common/price_gradients/launch.cuh"
#include "common/price_gradients/result.cuh"
#include "common/cuda_kernel_diagnostics.cuh"

namespace ai_factory::workbench::closed_form::price_gradients {
namespace pg = ::ai_factory::workbench::price_gradients;

template<typename Policy, unsigned int K>
__global__ void kernel(pg::DeviceInputs<typename Policy::InputRow> inputs,
                       pg::LaunchConfiguration launch, pg::Outputs outputs) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x)*blockDim.x + threadIdx.x;
         index < launch.result_count; index += stride) {
        const auto row = launch.result_offset + index;
        const auto* scenarios = inputs.scenarios + row*(1U+2U*K);
        const float central = Policy::evaluate(scenarios[0]);
        outputs.prices[row] = central;
        if constexpr (K > 0U) {
        #pragma unroll
        for (unsigned int i = 0; i < K; ++i) {
            const float first = Policy::evaluate(scenarios[2U*i+1U]);
            const float second = Policy::evaluate(scenarios[2U*i+2U]);
            outputs.gradients[row*K+i] = pg::difference(inputs.stencils[row*K+i], central, first, second);
        }
        }
    }
}

template<typename Policy, unsigned int K>
void launch(pg::DeviceInputs<typename Policy::InputRow> inputs, const pg::LaunchConfiguration& configuration,
            pg::Outputs outputs, const char* name, const char* variant) {
    const auto function = kernel<Policy, K>;
    int active = 0;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, function, configuration.threads_per_block, 0U), name);
    if (active == 0) throw std::invalid_argument("Gradient closed-form kernel has no resident block.");
    const std::string diagnostic_variant = std::string(variant) + "/K=" + std::to_string(K);
    report_cuda_kernel_launch_if_enabled(name, diagnostic_variant.c_str(), function, dim3(configuration.block_count),
        dim3(configuration.threads_per_block), 0U);
    kernel<Policy, K><<<configuration.block_count, configuration.threads_per_block>>>(inputs, configuration, outputs);
    check_cuda(cudaGetLastError(), name);
}
}  // namespace ai_factory::workbench::closed_form::price_gradients
