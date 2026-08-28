// Generic product binding for the host-prepared rough-Heston Markovian lift.
#pragma once

#include "common/equity/prepared_path_product_pricing.cuh"
#include "common/price_construction.cuh"
#include "common/simulation/schedule.cuh"
#include "model/equity/rough/rough_heston/dynamics_impl.cuh"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::equity::rough_heston {

template<
    std::size_t FactorCount,
    typename ProductPathPolicy,
    simulation::SchedulePolicy SchedulePolicy
>
requires std::same_as<
    typename SchedulePolicy::Dynamics,
    DynamicsPolicy<FactorCount>
>
void launch_rough_heston_markovian_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const PreparedDynamics<FactorCount>* device_prepared_dynamics,
    std::size_t prepared_dynamics_count,
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
    ::ai_factory::workbench::equity::launch_prepared_path_product_cuda<
        ModelParameters,
        PreparedDynamics<FactorCount>,
        DynamicsPolicy<FactorCount>,
        ProductPathPolicy,
        SchedulePolicy
    >(
        device_models,
        model_count,
        device_prepared_dynamics,
        prepared_dynamics_count,
        device_products,
        product_count,
        construction,
        result_count,
        result_offset,
        launch_result_count,
        monte_carlo_paths_per_price,
        dt,
        simulation_steps_per_day,
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

}  // namespace ai_factory::workbench::model::equity::rough_heston
