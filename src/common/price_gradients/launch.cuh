// Caller-owned gradient buffers, launch validation and host dispatch by cardinality.
#pragma once

#include "common/check_cuda.cuh"
#include "common/price_gradients/batching.hpp"
#include "common/equity/price_gradients/scenarios.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace ai_factory::workbench::price_gradients {

enum class PricingMethod { monte_carlo, closed_form };
struct LaunchConfiguration {
    PricingMethod method = PricingMethod::monte_carlo;
    std::size_t result_offset = 0U;
    std::size_t result_count = 1U;
    std::size_t paths_per_price = 1U << 20U;
    unsigned int threads_per_block = kDefaultThreadsPerBlock;
    std::size_t block_count = 1U;
    std::uint64_t base_seed = 0U;
    unsigned int sensitivity_batch_size = kDefaultSensitivityBatchSize;
};
template<typename Scenario>
struct DeviceInputs {
    const Scenario* scenarios;
    std::size_t scenario_capacity;
    const Stencil* stencils;
    std::size_t stencil_capacity;
};
struct Outputs {
    float* prices;
    float* price_standard_errors;
    float* gradients;
    float* gradient_standard_errors;
    std::size_t price_capacity;
    std::size_t gradient_capacity;  // row-major rows * K
};

struct BufferRange { std::uintptr_t begin, end; };
inline BufferRange checked_buffer_range(const void* pointer, std::size_t count, std::size_t element_bytes) {
    validate_device_pointer(pointer, "price-gradient buffer");
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (count > (std::numeric_limits<std::uintptr_t>::max() - begin) / element_bytes)
        throw std::overflow_error("Price-gradient buffer address overflow.");
    return {begin, begin + count * element_bytes};
}

template<typename Model, typename Product>
void validate_plan_buffers(
    const equity::price_gradients::ScenarioPlan<Model, Product>& host,
    DeviceInputs<typename equity::price_gradients::ScenarioPlan<Model, Product>::ScenarioType> device,
    const LaunchConfiguration& launch, Outputs outputs
) {
    const auto rows = host.result_count, k = host.sensitivity_count();
    host.time.validate();
    host.configuration.validate(k);
    if (rows == 0U || rows > std::numeric_limits<std::size_t>::max() / host.scenario_count()
        || host.scenarios.size() != rows * host.scenario_count() || host.stencils.size() != rows * k)
        throw std::invalid_argument("Malformed gradient scenario plan.");
    if (launch.result_offset >= rows || launch.result_count == 0U || launch.result_count > rows - launch.result_offset)
        throw std::invalid_argument("Price-gradient launch exceeds its result batch.");
    if (device.scenario_capacity < host.scenarios.size() || device.stencil_capacity < host.stencils.size()
        || outputs.price_capacity < rows || outputs.gradient_capacity < rows * k)
        throw std::invalid_argument("Insufficient price-gradient buffer capacity.");
    if (launch.method != PricingMethod::monte_carlo && launch.method != PricingMethod::closed_form)
        throw std::invalid_argument("Unknown price-gradient pricing method.");
    std::vector<BufferRange> inputs{
        checked_buffer_range(device.scenarios, host.scenarios.size(), sizeof(typename decltype(host.scenarios)::value_type))};
    if (k != 0U) inputs.push_back(checked_buffer_range(device.stencils, rows*k, sizeof(Stencil)));
    std::vector<BufferRange> out{checked_buffer_range(outputs.prices, rows, sizeof(float))};
    if (k != 0U) out.push_back(checked_buffer_range(outputs.gradients, rows*k, sizeof(float)));
    if (launch.method == PricingMethod::monte_carlo) {
        out.push_back(checked_buffer_range(outputs.price_standard_errors, rows, sizeof(float)));
        if (k != 0U) out.push_back(checked_buffer_range(outputs.gradient_standard_errors, rows*k, sizeof(float)));
    }
    const auto overlap = [](BufferRange a, BufferRange b) { return a.begin < b.end && b.begin < a.end; };
    for (std::size_t i = 0; i < out.size(); ++i) {
        for (const auto input : inputs) if (overlap(input, out[i]))
            throw std::invalid_argument("Gradient outputs overlap inputs.");
        for (std::size_t j = 0; j < i; ++j) if (overlap(out[i], out[j]))
            throw std::invalid_argument("Gradient output arrays overlap.");
    }
}

template<typename Model, typename Product>
void validate_launch(
    const equity::price_gradients::ScenarioPlan<Model, Product>& host,
    DeviceInputs<typename equity::price_gradients::ScenarioPlan<Model, Product>::ScenarioType> device,
    const LaunchConfiguration& launch, Outputs outputs
) {
    validate_plan_buffers(host, device, launch, outputs);
    const auto rows = host.result_count, k = host.sensitivity_count();
    validate_grid_x_size(launch.block_count);
    if (launch.method == PricingMethod::monte_carlo) {
        validate_monte_carlo_path_count(launch.paths_per_price);
        validate_reduction_block_size(launch.threads_per_block);
        validate_block_count(gradient_task_count(launch.result_count,
            sensitivity_batch_count(k, launch.sensitivity_batch_size)), launch.block_count);
        validate_row_seed_range(rows, launch.base_seed);
    } else {
        validate_cuda_block_size(launch.threads_per_block);
    }
}

template<unsigned int Maximum, unsigned int K = 0U, typename Function>
void dispatch_count(std::size_t count, Function&& function) {
    if (count == K) { function(std::integral_constant<unsigned int, K>{}); return; }
    if constexpr (K < Maximum) dispatch_count<Maximum, K+1U>(count, std::forward<Function>(function));
    else throw std::invalid_argument("Unsupported gradient cardinality.");
}

}  // namespace ai_factory::workbench::price_gradients
