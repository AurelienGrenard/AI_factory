// {model_display} {product_comment} composition over generic CUDA layers.
#include "model/equity/{model}/{product}.cuh"

#include "common/monte_carlo/monte_carlo_kernel.cuh"
#include "common/simulation/schedule.cuh"
#include "model/equity/{model}/dynamics.cu"
#include "product/{product}/pricing_policy.cuh"

namespace ai_factory::workbench::{model} {{
namespace {{

using Schedule = {schedule}<{model}::DynamicsPolicy>;
template<OptionSide Side>
using PricingPolicy =
    product::{pricing_policy}<Schedule, Side>;

static_assert(monte_carlo::ScalarMonteCarloPricingPolicy<PricingPolicy<OptionSide::call>>);

}}  // namespace

template<OptionSide Side>
void launch_{model}_{product}_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::{product_type}Parameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
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
{time_configuration}    monte_carlo::launch_monte_carlo_cuda<PricingPolicy<Side>>(
        make_model_product_device_inputs(
            device_models,
            model_count,
            device_products,
            product_count,
            cartesian_product
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
        option_side_name(Side),
        "{model_display} {operation_product} kernel"
    );
}}

template void launch_{model}_{product}_cuda<OptionSide::call>(
    const ModelParameters*, std::size_t,
    const product::{product_type}Parameters*, std::size_t,
    bool, std::size_t, std::size_t, std::size_t, std::size_t,
{instantiation_time_signature}
    float*, float*
);
template void launch_{model}_{product}_cuda<OptionSide::put>(
    const ModelParameters*, std::size_t,
    const product::{product_type}Parameters*, std::size_t,
    bool, std::size_t, std::size_t, std::size_t, std::size_t,
{instantiation_time_signature}
    float*, float*
);

}}  // namespace ai_factory::workbench::{model}
