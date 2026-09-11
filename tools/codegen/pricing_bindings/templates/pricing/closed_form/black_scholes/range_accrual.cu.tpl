// Generated closed-form Black-Scholes range-accrual composition.
#include "model/equity/markovian/black_scholes/product/range_accrual.cuh"
#include "model/equity/markovian/black_scholes/product/range_accrual_impl.cuh"

#include "common/closed_form/closed_form_kernels.cuh"
#include "common/compensated_sum.cuh"
#include "common/device_inputs.cuh"
#include "common/time_configuration.cuh"
#include "model/equity/markovian/black_scholes/analytics_impl.cuh"

namespace ai_factory::workbench::model::equity::black_scholes {

void launch_black_scholes_range_accrual_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::RangeAccrualParameters* device_products,
    std::size_t product_count,
    PriceConstruction construction,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    float day_fraction,
    unsigned int threads_per_block,
    std::size_t block_count,
    float* device_prices
) {
    using PricingPolicy = RangeAccrualClosedFormPricingPolicy;
    closed_form::launch_closed_form_cuda<PricingPolicy>(
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
        time::DayFractionTimeConfiguration{day_fraction},
        threads_per_block,
        block_count,
        device_prices,
        "black_scholes.range_accrual",
        "none",
        "Black-Scholes Range Accrual kernel"
    );
}

}  // namespace ai_factory::workbench::model::equity::black_scholes
