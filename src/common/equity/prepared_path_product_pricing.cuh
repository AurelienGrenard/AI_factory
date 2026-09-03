// Generic launcher for host-prepared fixed-step dynamics and path products.
#pragma once

#include "common/equity/path_product_monte_carlo_policy.cuh"
#include "common/monte_carlo/monte_carlo_kernel.cuh"
#include "common/price_construction.cuh"
#include "common/simulation/schedule.cuh"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::equity {

template<
    typename ModelParameters,
    typename PreparedDynamics,
    typename DynamicsPolicy,
    typename ProductPathPolicy,
    simulation::SchedulePolicy SchedulePolicy
>
requires std::same_as<typename SchedulePolicy::Dynamics, DynamicsPolicy>
void launch_prepared_path_product_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const PreparedDynamics* device_prepared_dynamics,
    std::size_t prepared_dynamics_count,
    const typename ProductPathPolicy::ProductParameters* host_products,
    const typename ProductPathPolicy::ProductParameters* device_products,
    std::size_t product_count,
    PriceConstruction construction,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    std::size_t monte_carlo_paths_per_price,
    float dt,
    std::uint32_t simulation_steps_per_day,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    float* device_prices,
    float* device_standard_errors,
    const char* diagnostic_name,
    const char* diagnostic_variant,
    const char* operation_name
) {
    using DeviceInputs = PreparedModelProductDeviceInputs<
        ModelParameters,
        typename ProductPathPolicy::ProductParameters,
        PreparedDynamics
    >;
    using PricingPolicy = PathProductMonteCarloPricingPolicy<
        SchedulePolicy,
        ProductPathPolicy,
        DeviceInputs
    >;
    static_assert(monte_carlo::ScalarMonteCarloPricingPolicy<PricingPolicy>);
    static_assert(
        sizeof(typename PricingPolicy::PreparedRow)
            <= monte_carlo::kMaximumSharedPreparedRowBytes
    );

    monte_carlo::launch_monte_carlo_cuda<PricingPolicy>(
        make_prepared_model_product_device_inputs(
            device_models,
            model_count,
            device_products,
            product_count,
            construction,
            device_prepared_dynamics,
            prepared_dynamics_count
        ),
        {host_products, product_count, construction},
        result_count,
        result_offset,
        launch_result_count,
        monte_carlo_paths_per_price,
        simulation::FixedStepTimeConfiguration{
            dt,
            simulation_steps_per_day,
        },
        threads_per_block,
        block_count,
        base_seed,
        device_prices,
        device_standard_errors,
        diagnostic_name,
        diagnostic_variant,
        operation_name
    );
}

}  // namespace ai_factory::workbench::equity
