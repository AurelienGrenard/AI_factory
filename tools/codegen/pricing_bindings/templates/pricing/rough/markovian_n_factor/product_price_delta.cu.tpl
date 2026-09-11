// Public {model_display} {header_product_comment} N-factor price/spot-delta launcher.

#include "model/equity/rough/{model}/product/{product}_price_delta.cuh"
#include "common/equity/price_delta/path_product_policy.cuh"
#include "common/equity/price_delta/multiplicative_spot_path.cuh"
#include "common/monte_carlo/monte_carlo_price_delta_kernel.cuh"
#include "model/equity/rough/{model}/dynamics_impl.cuh"
#include "product/{product}/pricing_policy.cuh"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::equity::{model} {{
namespace {{
template<std::size_t FactorCount>
using Schedule = {n_factor_schedule};
}}  // namespace

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
    namespace delta = ::ai_factory::workbench::equity::price_delta;
    using Product = {product_policy_expression};
    using Inputs = PreparedModelProductDeviceInputs<ModelParameters,
        typename Product::ProductParameters, PreparedDynamics<FactorCount>>;
    using Policy = delta::PathProductPriceDeltaPolicy<Schedule<FactorCount>,
        Product, delta::MultiplicativeSpotPath<DynamicsPolicy<FactorCount>>, Inputs>;
    const auto primary = make_prepared_model_product_device_inputs(
        device_models, model_count, device_products, product_count, construction,
        device_prepared_dynamics, prepared_dynamics_count);
    monte_carlo::launch_monte_carlo_price_delta_cuda<Policy>(
        {{primary, bump}},
        {{host_models, model_count, host_products, product_count, construction, bump}},
        result_count, result_offset, launch_result_count, monte_carlo_paths_per_price,
        simulation::FixedStepTimeConfiguration{{dt, simulation_steps_per_day}},
        threads_per_block, block_count, base_seed, device_prices, device_standard_errors,
        device_deltas, device_delta_standard_errors,
        "{model}.{product}_price_delta", {diagnostic_variant});
}}

{explicit_instantiations}
}}  // namespace ai_factory::workbench::model::equity::{model}
