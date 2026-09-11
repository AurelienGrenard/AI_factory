// Generated Black-Scholes RangeAccrual composition over shared closed-form bumping.
#include "model/equity/markovian/black_scholes/product/range_accrual_price_delta.cuh"

#include "common/closed_form/closed_form_price_delta_kernels.cuh"
#include "model/equity/markovian/black_scholes/product/range_accrual_impl.cuh"

namespace ai_factory::workbench::model::equity::black_scholes {


void launch_black_scholes_range_accrual_price_delta_cuda(
    const ModelParameters* host_models, const ModelParameters* device_models,
    std::size_t model_count, 
    const product::RangeAccrualParameters* device_products,
    std::size_t product_count, PriceConstruction construction,
    std::size_t result_count, std::size_t result_offset, std::size_t launch_result_count,
    float day_fraction, unsigned int threads_per_block, std::size_t block_count,
    ::ai_factory::workbench::equity::price_delta::SpotBumpConfiguration bump,
    float* device_prices, float* device_deltas
) {
    
    using PricePolicy = RangeAccrualClosedFormPricingPolicy;
    using Policy = ::ai_factory::workbench::equity::price_delta::ClosedFormSpotDeltaPolicy<PricePolicy>;
    closed_form::launch_closed_form_price_delta_cuda<Policy>(
        {make_model_product_device_inputs(device_models, model_count,
            device_products, product_count, construction), bump},
        host_models, result_count, result_offset, launch_result_count,
        {day_fraction}, threads_per_block, block_count, device_prices, device_deltas,
        "black_scholes.range_accrual.price_delta", "default");
}



}  // namespace ai_factory::workbench::model::equity::black_scholes
