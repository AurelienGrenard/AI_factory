// Shared one-thread-per-price European-swaption CUDA engine.
#pragma once

#include "common/check_cuda.cuh"
#include "common/cuda_kernel_diagnostics.cuh"
#include "common/fixed_income/swaption_side.cuh"
#include "common/result_index.cuh"
#include "product/european_swaption/dataset.hpp"
#include "product/european_swaption/schedule.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>

namespace ai_factory::workbench::fixed_income {

template<typename PreparedModel, typename ScheduleView>
struct PreparedEuropeanSwaptionRow {
    PreparedModel model;
    float notional;
    float strike;
    float exercise_time;
    ScheduleView schedule;
};

// Prepare one model/product row without schedule-pool allocation or copying.
template<typename Policy, typename Model, typename Product, typename Source>
__device__ __forceinline__ auto prepare_european_swaption_row(
    const Policy& policy,
    const Model& model,
    const Product& product,
    Source source,
    float time_day_fraction
) {
    const auto prepared_model = policy.prepare_model(model);
    const auto schedule = product::make_european_swaption_schedule_view(
        product, source, time_day_fraction
    );
    return PreparedEuropeanSwaptionRow<
        decltype(prepared_model), decltype(schedule)
    >{
        prepared_model,
        product.notional,
        product.strike,
        static_cast<float>(product.exercise_time) * time_day_fraction,
        schedule,
    };
}

// Prepare one fitted model/curve/product row.
template<
    typename Policy,
    typename Model,
    typename Curve,
    typename Product,
    typename Source
>
__device__ __forceinline__ auto prepare_european_swaption_row(
    const Policy& policy,
    const Model& model,
    const Curve& curve,
    const Product& product,
    Source source,
    float time_day_fraction
) {
    const auto prepared_model = policy.prepare_model(model, curve);
    const auto schedule = product::make_european_swaption_schedule_view(
        product, source, time_day_fraction
    );
    return PreparedEuropeanSwaptionRow<
        decltype(prepared_model), decltype(schedule)
    >{
        prepared_model,
        product.notional,
        product.strike,
        static_cast<float>(product.exercise_time) * time_day_fraction,
        schedule,
    };
}

template<SwaptionSide Side, typename Policy, typename Row>
__device__ __forceinline__ float evaluate_european_swaption_price(
    const Policy& policy,
    const Row& row
) {
    return row.notional * policy.template price<Side>(
        row.model,
        policy.initial_state(row.model),
        0.0f,
        row.exercise_time,
        row.strike,
        row.schedule
    );
}

// Share one kernel across standalone one-factor model policies.
template<
    SwaptionSide Side,
    typename Policy,
    typename Model,
    typename Product,
    typename Source
>
__global__ void one_factor_european_swaption_kernel(
    const Model* __restrict__ models,
    const Product* __restrict__ products,
    Source schedule_source,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_offset,
    std::size_t launch_result_count,
    float time_day_fraction,
    float* __restrict__ prices
) {
    const std::size_t launch_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (launch_index >= launch_result_count) return;

    const std::size_t result_index = result_offset + launch_index;
    const ModelProductIndices indices = decode_model_product_result_index(
        result_index, product_count, cartesian_product
    );
    const auto row = prepare_european_swaption_row(
        Policy{},
        models[indices.model_index],
        products[indices.product_index],
        schedule_source,
        time_day_fraction
    );
    prices[result_index] = evaluate_european_swaption_price<Side>(
        Policy{}, row
    );
}

// Share one kernel across fitted one-factor model/curve policies.
template<
    SwaptionSide Side,
    typename Policy,
    typename Model,
    typename Curve,
    typename Product,
    typename Source
>
__global__ void fitted_one_factor_european_swaption_kernel(
    const Model* __restrict__ models,
    const Curve* __restrict__ curves,
    const Product* __restrict__ products,
    Source schedule_source,
    std::size_t curve_count,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_offset,
    std::size_t launch_result_count,
    float time_day_fraction,
    float* __restrict__ prices
) {
    const std::size_t launch_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (launch_index >= launch_result_count) return;

    const std::size_t result_index = result_offset + launch_index;
    const ModelCurveProductIndices indices =
        decode_model_curve_product_result_index(
            result_index,
            curve_count,
            product_count,
            cartesian_product
        );
    const auto row = prepare_european_swaption_row(
        Policy{},
        models[indices.model_index],
        curves[indices.curve_index],
        products[indices.product_index],
        schedule_source,
        time_day_fraction
    );
    prices[result_index] = evaluate_european_swaption_price<Side>(
        Policy{}, row
    );
}

inline void validate_european_swaption_schedule_source(
    product::RegularEuropeanSwaptionScheduleSource
) {}

inline void validate_european_swaption_schedule_source(
    product::ExplicitEuropeanSwaptionScheduleSource source
) {
    validate_device_pointer(source.payment_times, "device_payment_times");
    validate_device_pointer(
        source.accrual_fractions, "device_accrual_fractions"
    );
    if (source.schedule_size == 0U) {
        throw std::invalid_argument(
            "The European swaption schedule pool is empty."
        );
    }
}

inline void validate_european_swaption_batch(
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    float time_day_fraction,
    unsigned int threads_per_block,
    std::size_t block_count
) {
    validate_day_fraction(time_day_fraction);
    if (result_offset >= result_count
        || launch_result_count == 0U
        || launch_result_count > result_count - result_offset) {
        throw std::invalid_argument(
            "The European swaption launch batch exceeds the result array."
        );
    }
    validate_cuda_block_size(threads_per_block);
    validate_block_count(launch_result_count, block_count);
    validate_grid_x_size(block_count);
    const std::size_t thread_count = checked_workspace_product(
        block_count,
        static_cast<std::size_t>(threads_per_block),
        "The European swaption thread count exceeds size_t."
    );
    if (thread_count < launch_result_count) {
        throw std::invalid_argument(
            "The European swaption launch requires one thread per price."
        );
    }
}

// Validate and launch one standalone model specialization.
template<
    SwaptionSide Side,
    typename Policy,
    typename Model,
    typename Product,
    typename Source
>
void launch_one_factor_european_swaption(
    const char* kernel_name,
    const Model* device_models,
    std::size_t model_count,
    const Product* device_products,
    Source schedule_source,
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
    validate_device_pointer(device_models, "device_models");
    validate_device_pointer(device_products, "device_products");
    validate_device_pointer(device_prices, "device_prices");
    validate_european_swaption_schedule_source(schedule_source);
    validate_model_product_construction(
        model_count, product_count, cartesian_product, result_count
    );
    validate_european_swaption_batch(
        result_count,
        result_offset,
        launch_result_count,
        time_day_fraction,
        threads_per_block,
        block_count
    );

    report_cuda_kernel_launch_if_enabled(
        kernel_name,
        swaption_side_name(Side),
        one_factor_european_swaption_kernel<
            Side, Policy, Model, Product, Source
        >,
        dim3(static_cast<unsigned int>(block_count)),
        dim3(threads_per_block)
    );
    one_factor_european_swaption_kernel<
        Side, Policy, Model, Product, Source
    ><<<static_cast<unsigned int>(block_count), threads_per_block>>>(
        device_models,
        device_products,
        schedule_source,
        product_count,
        cartesian_product,
        result_offset,
        launch_result_count,
        time_day_fraction,
        device_prices
    );
    check_cuda(cudaGetLastError(), "European swaption kernel");
}

// Validate and launch one fitted model/curve specialization.
template<
    SwaptionSide Side,
    typename Policy,
    typename Model,
    typename Curve,
    typename Product,
    typename Source
>
void launch_fitted_one_factor_european_swaption(
    const char* kernel_name,
    const Model* device_models,
    std::size_t model_count,
    const Curve* device_curves,
    std::size_t curve_count,
    const Product* device_products,
    Source schedule_source,
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
    validate_device_pointer(device_models, "device_models");
    validate_device_pointer(device_curves, "device_curves");
    validate_device_pointer(device_products, "device_products");
    validate_device_pointer(device_prices, "device_prices");
    validate_european_swaption_schedule_source(schedule_source);
    validate_model_curve_product_construction(
        model_count,
        curve_count,
        product_count,
        cartesian_product,
        result_count
    );
    validate_european_swaption_batch(
        result_count,
        result_offset,
        launch_result_count,
        time_day_fraction,
        threads_per_block,
        block_count
    );

    report_cuda_kernel_launch_if_enabled(
        kernel_name,
        swaption_side_name(Side),
        fitted_one_factor_european_swaption_kernel<
            Side, Policy, Model, Curve, Product, Source
        >,
        dim3(static_cast<unsigned int>(block_count)),
        dim3(threads_per_block)
    );
    fitted_one_factor_european_swaption_kernel<
        Side, Policy, Model, Curve, Product, Source
    ><<<static_cast<unsigned int>(block_count), threads_per_block>>>(
        device_models,
        device_curves,
        device_products,
        schedule_source,
        curve_count,
        product_count,
        cartesian_product,
        result_offset,
        launch_result_count,
        time_day_fraction,
        device_prices
    );
    check_cuda(cudaGetLastError(), "European swaption kernel");
}

}  // namespace ai_factory::workbench::fixed_income
