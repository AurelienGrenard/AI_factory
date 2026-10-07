// {model_display} {product_comment} legacy delta API backed by the rough lift graph.
#include "model/equity/rough/{model}/product/{product}_price_delta.cuh"
#include "model/equity/rough/{model}/product/{product}_price_gradients.cuh"
#include "common/volterra/price_gradients/legacy_spot_delta_graph.cuh"

namespace ai_factory::workbench::model::equity::{model} {{

{template_declaration}void launch_{model}_{product}_price_delta_cuda(
    const ModelParameters* host_models,
    const ModelParameters* device_models,
    std::size_t model_count,
    const PreparedDynamics<FactorCount>* device_prepared_dynamics,
    std::size_t prepared_dynamics_count,
    const product::{product_type}Parameters* host_products,
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
    ::ai_factory::workbench::equity::price_delta::SpotBumpConfiguration bump,
    float* device_prices,
    float* device_standard_errors,
    float* device_deltas,
    float* device_delta_standard_errors
) {{
    (void)device_models;
    (void)device_products;
    using Graph = {product_type}NodeGraph<{delta_graph_arguments}>;
    volterra::price_gradients::launch_legacy_lift_spot_delta<Graph>(
        host_models, model_count, device_prepared_dynamics,
        prepared_dynamics_count, host_products, product_count,
        construction, result_count, result_offset, launch_result_count,
        monte_carlo_paths_per_price, dt, simulation_steps_per_day,
        threads_per_block, block_count, base_seed, bump,
        device_prices, device_standard_errors, device_deltas,
        device_delta_standard_errors,
        prepare_{model}_{product}_sensitivities
    );
}}

{explicit_instantiations}
}}  // namespace ai_factory::workbench::model::equity::{model}
