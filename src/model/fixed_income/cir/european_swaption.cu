// Closed-form European swaptions under the CIR short rate.
#include "model/fixed_income/cir/european_swaption.cuh"

#include "common/fixed_income/european_swaption.cuh"

#include "model/fixed_income/cir/analytics.cu"

#include <cstddef>

namespace ai_factory::workbench::model::cir {

template<SwaptionSide Side>
void launch_cir_european_swaption_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::RegularEuropeanSwaptionParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    float time_day_fraction,
    unsigned int threads_per_block,
    std::size_t block_count,
    float* device_prices
) {
    fixed_income::launch_one_factor_european_swaption<Side>(
        "cir.european_swaption",
        device_models,
        model_count,
        device_products,
        product::RegularEuropeanSwaptionScheduleSource{},
        product_count,
        cartesian_product,
        result_count,
        result_offset,
        launch_result_count,
        time_day_fraction,
        threads_per_block,
        block_count,
        device_prices
    );
}

template<SwaptionSide Side>
void launch_cir_european_swaption_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::ExplicitEuropeanSwaptionParameters* device_products,
    const std::uint32_t* device_payment_times,
    const float* device_accrual_fractions,
    std::size_t schedule_size,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    float time_day_fraction,
    unsigned int threads_per_block,
    std::size_t block_count,
    float* device_prices
) {
    fixed_income::launch_one_factor_european_swaption<Side>(
        "cir.european_swaption",
        device_models,
        model_count,
        device_products,
        product::ExplicitEuropeanSwaptionScheduleSource{
            device_payment_times,
            device_accrual_fractions,
            schedule_size,
        },
        product_count,
        cartesian_product,
        result_count,
        result_offset,
        launch_result_count,
        time_day_fraction,
        threads_per_block,
        block_count,
        device_prices
    );
}

template void launch_cir_european_swaption_cuda<SwaptionSide::payer>(
    const ModelParameters*, std::size_t,
    const product::RegularEuropeanSwaptionParameters*, std::size_t,
    bool, std::size_t, std::size_t, std::size_t, float,
    unsigned int, std::size_t, float*
);
template void launch_cir_european_swaption_cuda<SwaptionSide::receiver>(
    const ModelParameters*, std::size_t,
    const product::RegularEuropeanSwaptionParameters*, std::size_t,
    bool, std::size_t, std::size_t, std::size_t, float,
    unsigned int, std::size_t, float*
);
template void launch_cir_european_swaption_cuda<SwaptionSide::payer>(
    const ModelParameters*, std::size_t,
    const product::ExplicitEuropeanSwaptionParameters*,
    const std::uint32_t*, const float*, std::size_t, std::size_t,
    bool, std::size_t, std::size_t, std::size_t, float,
    unsigned int, std::size_t, float*
);
template void launch_cir_european_swaption_cuda<SwaptionSide::receiver>(
    const ModelParameters*, std::size_t,
    const product::ExplicitEuropeanSwaptionParameters*,
    const std::uint32_t*, const float*, std::size_t, std::size_t,
    bool, std::size_t, std::size_t, std::size_t, float,
    unsigned int, std::size_t, float*
);

}  // namespace ai_factory::workbench::model::cir
