// One block per (price, sensitivity batch); exact-width tails use a second grid.
#pragma once
#include "common/price_gradients/launch.cuh"
#include "common/equity/price_gradients/terminal_policy.cuh"
#include "common/cuda_kernel_diagnostics.cuh"
#include "common/reductions.cuh"
#include "common/monte_carlo/concepts.cuh"

namespace ai_factory::workbench::monte_carlo::price_gradients {
namespace pg = ::ai_factory::workbench::price_gradients;

// All tasks in one grid have the same compile-time width. The global selection
// remains row-major; a batch is a view, never a reordered or copied scenario plan.
struct BatchGroup {
    std::size_t sensitivity_count;
    std::size_t first_sensitivity;
    std::size_t batches_per_row;
};

// Every branch selecting these specializations is uniform across the block and
// outside the path loop. Only the owner has a price moment channel.
template<typename Policy, bool PriceOwner, bool IncludeCentral, bool IncludePayoff>
__device__ __forceinline__ void evaluate_and_reduce(
    const typename Policy::PreparedRow& prepared, philox::PhiloxKey key,
    const pg::LaunchConfiguration& launch, pg::Outputs outputs,
    std::size_t row, std::size_t first, std::size_t total_sensitivities
) {
    constexpr unsigned int K = Policy::kSensitivities;
    constexpr unsigned int price_channels = PriceOwner ? 1U : 0U;
    constexpr unsigned int channels = K + price_channels;
    static_assert(channels > 0U && (!PriceOwner || IncludePayoff));
    double sums[channels]{}, squares[channels]{};
    for (std::size_t path = threadIdx.x; path < launch.paths_per_price; path += blockDim.x) {
        const auto result = Policy::template evaluate_path<IncludeCentral,IncludePayoff>(prepared,key,path);
        if constexpr (PriceOwner) {
            const double price = static_cast<double>(result.price);
            sums[0] += price;
            squares[0] += price * price;
        }
        if constexpr (K > 0U) {
            #pragma unroll
            for (unsigned int i = 0; i < K; ++i) {
                const double gradient = static_cast<double>(result.gradients[i]);
                sums[i+price_channels] += gradient;
                squares[i+price_channels] += gradient * gradient;
            }
        }
    }
    #pragma unroll
    for (unsigned int channel = 0; channel < channels; ++channel) {
        const auto moments = reductions::reduce_block(sums[channel],squares[channel]);
        if (threadIdx.x == 0U) {
            double value, error;
            reductions::compute_statistics(moments,launch.paths_per_price,value,error,
                launch.paths_per_price / blockDim.x + (launch.paths_per_price % blockDim.x != 0U));
            if constexpr (PriceOwner) {
                if (channel == 0U) {
                    outputs.prices[row] = static_cast<float>(value);
                    outputs.price_standard_errors[row] = static_cast<float>(error);
                } else {
                    const auto index = row*total_sensitivities + first + channel-1U;
                    outputs.gradients[index] = static_cast<float>(value);
                    outputs.gradient_standard_errors[index] = static_cast<float>(error);
                }
            } else {
                const auto index = row*total_sensitivities + first + channel;
                outputs.gradients[index] = static_cast<float>(value);
                outputs.gradient_standard_errors[index] = static_cast<float>(error);
            }
        }
        __syncthreads();
    }
}

template<typename Policy>
__global__ void kernel(pg::DeviceInputs<typename Policy::InputRow> inputs, pg::LaunchConfiguration launch,
                       equity::price_gradients::TimeConfiguration time, pg::Outputs outputs, BatchGroup group) {
    constexpr unsigned int K = Policy::kSensitivities;
    static_assert(sizeof(typename Policy::PreparedRow) <= kMaximumSharedPreparedRowBytes,
        "Gradient prepared row exceeds shared budget; use a compact view.");
    __shared__ typename Policy::PreparedRow prepared;
    __shared__ philox::PhiloxKey key;
    __shared__ pg::CentralRequirement central_requirement;
    const std::size_t tasks = launch.result_count * group.batches_per_row;
    for (std::size_t index = blockIdx.x; index < tasks; index += gridDim.x) {
        const std::size_t row = launch.result_offset + index / group.batches_per_row;
        const std::size_t first = group.first_sensitivity + (index % group.batches_per_row)*K;
        if (threadIdx.x == 0U) {
            const auto* central = inputs.scenarios + row * (1U + 2U*group.sensitivity_count);
            const auto* pairs = central + 1U + 2U*first;
            auto required = first == 0U ? pg::CentralRequirement::payoff : pg::CentralRequirement::none;
            if constexpr (K > 0U) {
                #pragma unroll
                for (unsigned int i = 0; i < K; ++i)
                    if (pairs[2U*i].central_requirement > required) required = pairs[2U*i].central_requirement;
            }
            central_requirement = required;
            prepared = Policy::prepare(central,pairs,
                K == 0U ? nullptr : inputs.stencils + row*group.sensitivity_count + first,time,
                required != pg::CentralRequirement::none);
            key = philox::make_key(launch.base_seed + row);
        }
        __syncthreads();
        if constexpr (K == 0U) {
            evaluate_and_reduce<Policy,true,true,true>(prepared,key,launch,outputs,row,first,group.sensitivity_count);
        } else if (first == 0U) {
            evaluate_and_reduce<Policy,true,true,true>(prepared,key,launch,outputs,row,first,group.sensitivity_count);
        } else if (central_requirement == pg::CentralRequirement::payoff) {
            evaluate_and_reduce<Policy,false,true,true>(prepared,key,launch,outputs,row,first,group.sensitivity_count);
        } else if (central_requirement == pg::CentralRequirement::state) {
            evaluate_and_reduce<Policy,false,true,false>(prepared,key,launch,outputs,row,first,group.sensitivity_count);
        } else {
            evaluate_and_reduce<Policy,false,false,false>(prepared,key,launch,outputs,row,first,group.sensitivity_count);
        }
        // The last reduction synchronizes before another task reuses shared data.
    }
}

template<typename Policy>
void launch_group(pg::DeviceInputs<typename Policy::InputRow> inputs, const pg::LaunchConfiguration& configuration,
            equity::price_gradients::TimeConfiguration time, pg::Outputs outputs, BatchGroup group, const char* name, const char* variant) {
    const auto function = kernel<Policy>;
    const std::size_t shared = 2U * (configuration.threads_per_block / 32U) * sizeof(double);
    int active = 0;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, function, configuration.threads_per_block, shared), name);
    if (active == 0) throw std::invalid_argument("Gradient MC kernel has no resident block.");
    const std::size_t blocks = std::min(configuration.block_count,
        pg::gradient_task_count(configuration.result_count, group.batches_per_row));
    const std::string diagnostic_variant = std::string(variant) + "/B=" + std::to_string(Policy::kSensitivities);
    report_cuda_kernel_launch_if_enabled(name, diagnostic_variant.c_str(), function, dim3(blocks),
        dim3(configuration.threads_per_block), shared);
    kernel<Policy><<<blocks, configuration.threads_per_block, shared>>>(inputs, configuration, time, outputs, group);
    check_cuda(cudaGetLastError(), name);
}
// Dispatch only bounded widths, independent of the number or identity of selected
// parameters. Splitting the remainder outside the kernel avoids hot-path masks.
template<typename Dynamics, typename ProductPolicy>
void launch(pg::DeviceInputs<typename equity::price_gradients::TerminalPolicy<Dynamics,ProductPolicy,0U>::InputRow> inputs,
            const pg::LaunchConfiguration& configuration, equity::price_gradients::TimeConfiguration time,
            pg::Outputs outputs, std::size_t count, const char* name, const char* variant) {
    const auto submit = [&](auto width, BatchGroup group) {
        using Policy = equity::price_gradients::TerminalPolicy<Dynamics,ProductPolicy,decltype(width)::value>;
        launch_group<Policy>(inputs,configuration,time,outputs,group,name,variant);
    };
    if (count == 0U) {
        submit(std::integral_constant<unsigned int,0U>{},{0U,0U,1U});
        return;
    }
    const unsigned int width = configuration.sensitivity_batch_size;
    pg::validate_sensitivity_batch_size(width);
    const std::size_t full = count/width, remainder = count%width;
    if (full != 0U) {
        const BatchGroup group{count,0U,full};
        switch (width) {
        case 1U: submit(std::integral_constant<unsigned int,1U>{},group); break;
        case 2U: submit(std::integral_constant<unsigned int,2U>{},group); break;
        case 4U: submit(std::integral_constant<unsigned int,4U>{},group); break;
        }
    }
    if (remainder != 0U) {
        pg::dispatch_count<3U,1U>(remainder,[&](auto tail) {
            submit(tail,{count,full*width,1U});
        });
    }
}
}  // namespace ai_factory::workbench::monte_carlo::price_gradients
