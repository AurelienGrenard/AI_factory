// {model_display} {product_comment} composition over a prepared N-factor lift.
#include "model/equity/rough/{model}/{product}.cuh"

#include "common/simulation/schedule.cuh"
#include "model/equity/rough/{model}/markovian_pricing.cuh"
#include "product/{product}/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::{model} {{
namespace {{

template<std::size_t FactorCount>
using Schedule = {n_factor_schedule};

}}  // namespace

{template_declaration}void launch_{model}_{product}_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const PreparedDynamics<FactorCount>* device_prepared_dynamics,
    std::size_t prepared_dynamics_count,
    const product::{product_type}Parameters* device_products,
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
    float* device_standard_errors
) {{
    using ProductPolicy = {product_policy_expression};
    launch_{model}_markovian_cuda<
        FactorCount,
        ProductPolicy,
        Schedule<FactorCount>
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
        "{model}.{product}",
        {diagnostic_variant},
        "{model_display} {operation_product} kernel"
    );
}}

{explicit_instantiations}
}}  // namespace ai_factory::workbench::model::equity::{model}
