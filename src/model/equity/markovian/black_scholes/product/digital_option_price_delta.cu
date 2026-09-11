// Generated Black-Scholes DigitalOption composition over shared closed-form bumping.
#include "model/equity/markovian/black_scholes/product/digital_option_price_delta.cuh"

#include "common/closed_form/closed_form_price_delta_kernels.cuh"
#include "model/equity/markovian/black_scholes/product/digital_option_impl.cuh"

namespace ai_factory::workbench::model::equity::black_scholes {

template<OptionSide Side>
void launch_black_scholes_digital_option_price_delta_cuda(
    const ModelParameters* host_models, const ModelParameters* device_models,
    std::size_t model_count, 
    const product::DigitalOptionParameters* device_products,
    std::size_t product_count, PriceConstruction construction,
    std::size_t result_count, std::size_t result_offset, std::size_t launch_result_count,
    float day_fraction, unsigned int threads_per_block, std::size_t block_count,
    ::ai_factory::workbench::equity::price_delta::SpotBumpConfiguration bump,
    float* device_prices, float* device_deltas
) {
    
    using PricePolicy = DigitalOptionClosedFormPricingPolicy<Side>;
    using Policy = ::ai_factory::workbench::equity::price_delta::ClosedFormSpotDeltaPolicy<PricePolicy>;
    closed_form::launch_closed_form_price_delta_cuda<Policy>(
        {make_model_product_device_inputs(device_models, model_count,
            device_products, product_count, construction), bump},
        host_models, result_count, result_offset, launch_result_count,
        {day_fraction}, threads_per_block, block_count, device_prices, device_deltas,
        "black_scholes.digital_option.price_delta", option_side_name(Side));
}

template void launch_black_scholes_digital_option_price_delta_cuda<OptionSide::call>(
    const ModelParameters*, const ModelParameters*, std::size_t, const product::DigitalOptionParameters*, std::size_t, PriceConstruction, std::size_t, std::size_t, std::size_t, float, unsigned int, std::size_t, ::ai_factory::workbench::equity::price_delta::SpotBumpConfiguration, float*, float*);
template void launch_black_scholes_digital_option_price_delta_cuda<OptionSide::put>(
    const ModelParameters*, const ModelParameters*, std::size_t, const product::DigitalOptionParameters*, std::size_t, PriceConstruction, std::size_t, std::size_t, std::size_t, float, unsigned int, std::size_t, ::ai_factory::workbench::equity::price_delta::SpotBumpConfiguration, float*, float*);

}  // namespace ai_factory::workbench::model::equity::black_scholes
