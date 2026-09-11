// {model_display} {product_comment} composition over the Volterra FFT engine.
#include "model/equity/rough/{model}/product/{product}_price_delta.cuh"

#include "common/volterra/hybrid_schedule.cuh"
#include "{kernel_header}"
#include "common/volterra/hybrid_fft_price_delta.cuh"
#include "model/equity/rough/{model}/dynamics_impl.cuh"
#include "product/{product}/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::{model} {{

{template_declaration}void launch_{model}_{product}_price_delta_cuda(
    const ModelParameters* host_models,
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::{product_type}Parameters* host_products,
    const product::{product_type}Parameters* device_products,
    std::size_t product_count,
    PriceConstruction construction,
    std::size_t result_count,
    std::size_t result_index,
    std::size_t monte_carlo_paths_per_price,
    float day_fraction,
    float target_dt,
    std::size_t step_count,
    std::size_t path_chunk_size,
    void* device_workspace,
    std::size_t workspace_bytes,
    std::uint64_t base_seed,
    ::ai_factory::workbench::equity::price_delta::SpotBumpConfiguration bump,
    float* device_prices,
    float* device_standard_errors,
    float* device_deltas,
    float* device_delta_errors
) {{
    using ProductPolicy = {product_policy_expression};
    volterra::hybrid_fft::launch_price_delta_cuda<
        {kernel}, PathPolicy, ProductPolicy, {volterra_schedule},
        ::ai_factory::workbench::equity::price_delta::{path_strategy}<PathPolicy>
    >(
        host_models,
        device_models,
        model_count,
        host_products,
        device_products,
        product_count,
        construction,
        result_count,
        result_index,
        monte_carlo_paths_per_price,
        volterra::HybridTimeConfiguration{{day_fraction, target_dt}},
        step_count,
        path_chunk_size,
        device_workspace,
        workspace_bytes,
        base_seed,
        bump,
        device_prices,
        device_standard_errors,
        device_deltas,
        device_delta_errors,
        "{model}.{product}.price_delta",
        {diagnostic_variant}
    );
}}

{explicit_instantiations}
}}  // namespace ai_factory::workbench::model::equity::{model}
