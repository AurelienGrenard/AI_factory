// Shared host validation for compact device-prepared sensitivity launches.
#pragma once

#include "common/price_gradients/device_prepared_launch.cuh"
#include "common/price_gradients/launch.cuh"

#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace ai_factory::workbench::monte_carlo::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;

template<typename HostPlan>
DevicePreparedPlan make_device_prepared_plan(const HostPlan& host) {
    DevicePreparedPlan result{
        host.time,
        host.construction,
        host.models.size(),
        host.products.size(),
        host.result_count,
        host.sensitivity_count(),
        0U,
        host.has_maturity_sensitivity(),
    };
    if constexpr (requires { host.curves.size(); }) {
        result.curve_count = host.curves.size();
    }
    return result;
}

template<std::size_t NodeCapacity, typename Inputs>
void validate_device_prepared_launch(
    Inputs inputs,
    DevicePreparedPlan plan,
    DevicePreparedStencilOutputs<NodeCapacity> stencil_outputs,
    const pg::LaunchConfiguration& launch,
    std::span<const pg::BufferRange> numerical_outputs
) {
    if (launch.method != pg::PricingMethod::monte_carlo
        && launch.method != pg::PricingMethod::closed_form) {
        throw std::invalid_argument("Unknown device-prepared pricing method.");
    }
    if (plan.result_count == 0U
        || launch.result_offset >= plan.result_count
        || launch.result_count == 0U
        || launch.result_count > plan.result_count - launch.result_offset) {
        throw std::invalid_argument(
            "Device-prepared sensitivity launch exceeds its row range."
        );
    }
    bool insufficient_input_capacity =
        inputs.model_capacity < plan.model_count
        || inputs.product_capacity < plan.product_count
        || inputs.sensitivity_capacity < plan.sensitivity_count;
    if constexpr (requires { inputs.curve_capacity; }) {
        insufficient_input_capacity = insufficient_input_capacity
            || inputs.curve_capacity < plan.curve_count;
    }
    if (insufficient_input_capacity) {
        throw std::invalid_argument(
            "Insufficient device-prepared sensitivity input capacity."
        );
    }
    const auto sensitivity_count =
        plan.result_count * plan.sensitivity_count;
    if (plan.sensitivity_count != 0U
        && (stencil_outputs.stencils == nullptr
            || stencil_outputs.capacity < sensitivity_count)) {
        throw std::invalid_argument(
            "Insufficient represented-stencil output capacity."
        );
    }
    if (plan.sensitivity_count != 0U && stencil_outputs.error == nullptr) {
        throw std::invalid_argument("Missing device-preparation status buffer.");
    }
    if (launch.method == pg::PricingMethod::monte_carlo) {
        validate_monte_carlo_path_count(launch.paths_per_price);
        validate_reduction_block_size(launch.threads_per_block);
        validate_row_seed_range(plan.result_count, launch.base_seed);
        if (launch.sensitivity_batch_size != 1U) {
            throw std::invalid_argument(
                "Device-prepared Monte Carlo uses one sensitivity per block."
            );
        }
        const auto task_count = launch.result_count
            * (plan.sensitivity_count == 0U ? 1U : plan.sensitivity_count);
        if (launch.block_count == 0U
            || launch.block_count > task_count
            || (plan.sensitivity_count != 0U
                && launch.block_count < plan.sensitivity_count)) {
            throw std::invalid_argument(
                "Invalid device-prepared sensitivity block cap."
            );
        }
    } else {
        validate_cuda_block_size(launch.threads_per_block);
        validate_grid_x_size(launch.block_count);
    }
    if (launch.result_count
            > static_cast<std::size_t>(
                std::numeric_limits<unsigned int>::max()
            )
        || plan.sensitivity_count > 65535U) {
        throw std::invalid_argument(
            "Device-prepared sensitivity CUDA grid overflow."
        );
    }

    std::vector<pg::BufferRange> immutable_inputs{
        pg::checked_buffer_range(
            inputs.models,
            plan.model_count,
            sizeof(typename Inputs::PreparationPolicy::Model)
        ),
        pg::checked_buffer_range(
            inputs.products,
            plan.product_count,
            sizeof(typename Inputs::PreparationPolicy::Product)
        ),
    };
    if constexpr (requires { inputs.curves; }) {
        immutable_inputs.push_back(pg::checked_buffer_range(
            inputs.curves,
            plan.curve_count,
            sizeof(typename Inputs::PreparationPolicy::Curve)
        ));
    }
    if (plan.sensitivity_count != 0U) {
        immutable_inputs.push_back(pg::checked_buffer_range(
            inputs.sensitivities,
            plan.sensitivity_count,
            sizeof(pg::SensitivitySpec<
                typename Inputs::PreparationPolicy::Parameter
            >)
        ));
    }
    std::vector<pg::BufferRange> mutable_outputs(
        numerical_outputs.begin(), numerical_outputs.end()
    );
    if (plan.sensitivity_count != 0U) {
        mutable_outputs.push_back(pg::checked_buffer_range(
            stencil_outputs.stencils,
            sensitivity_count,
            sizeof(pg::SensitivityStencil<NodeCapacity>)
        ));
    }
    if (plan.sensitivity_count != 0U) {
        mutable_outputs.push_back(pg::checked_buffer_range(
            stencil_outputs.error,
            1U,
            sizeof(preparation::Error)
        ));
    }
    const auto overlaps = [](pg::BufferRange first, pg::BufferRange second) {
        return first.begin < second.end && second.begin < first.end;
    };
    for (std::size_t i = 0U; i < mutable_outputs.size(); ++i) {
        for (const auto input : immutable_inputs) {
            if (overlaps(input, mutable_outputs[i])) {
                throw std::invalid_argument(
                    "Device-prepared sensitivity outputs overlap inputs."
                );
            }
        }
        for (std::size_t j = 0U; j < i; ++j) {
            if (overlaps(mutable_outputs[i], mutable_outputs[j])) {
                throw std::invalid_argument(
                    "Device-prepared sensitivity output arrays overlap."
                );
            }
        }
    }
}

}  // namespace ai_factory::workbench::monte_carlo::price_gradients
