// {model_display} {product_comment} composition over generic CUDA layers.
#include "model/equity/markovian/{model}/product/{product}.cuh"

#include "common/monte_carlo/monte_carlo_kernel.cuh"
#include "common/simulation/schedule.cuh"
#include "model/equity/markovian/{model}/dynamics_impl.cuh"
#include "product/{product}/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::{model} {{
namespace {{

using Schedule = {schedule_expression};
{pricing_policy_declaration}

static_assert(monte_carlo::ScalarMonteCarloPricingPolicy<{static_pricing_policy_use}>);

}}  // namespace

{template_declaration}void launch_{model}_{product}_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::{product_type}Parameters* device_products,
    std::size_t product_count,
    PriceConstruction construction,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    std::size_t monte_carlo_paths_per_price,
{time_parameter_declarations}    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    float* device_prices,
    float* device_standard_errors
) {{
{time_configuration}    monte_carlo::launch_monte_carlo_cuda<{pricing_policy_use}>(
        make_model_product_device_inputs(
            device_models,
            model_count,
            device_products,
            product_count,
            construction
        ),
        result_count,
        result_offset,
        launch_result_count,
        monte_carlo_paths_per_price,
        time_configuration,
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
