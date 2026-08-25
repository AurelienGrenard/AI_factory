// Contiguous device-array layouts shared by pricing execution methods.
#pragma once

#include "common/check_cuda.cuh"
#include "common/result_index.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <type_traits>

namespace ai_factory::workbench {

template<typename Model, typename Product>
struct ModelProductDeviceInputs {
    using ModelParameters = Model;
    using ProductParameters = Product;

    const Model* models;
    std::size_t model_count;
    const Product* products;
    std::size_t product_count;
    bool cartesian_product;

    __device__ __forceinline__ ModelProductIndices indices(
        std::size_t result_index
    ) const {
        return decode_model_product_result_index(
            result_index,
            product_count,
            cartesian_product
        );
    }

    inline void validate(std::size_t result_count) const {
        validate_device_pointer(models, "device_models");
        validate_device_pointer(products, "device_products");
        validate_model_product_construction(
            model_count,
            product_count,
            cartesian_product,
            result_count
        );
    }

    template<
        typename Pricing,
        typename TimeConfiguration,
        typename... AdditionalInputs
    >
    __device__ __forceinline__ typename Pricing::PreparedRow prepare_row(
        std::size_t result_index,
        const TimeConfiguration& time_configuration,
        const AdditionalInputs&... additional_inputs
    ) const {
        const ModelProductIndices row = indices(result_index);
        return Pricing::prepare_row(
            models[row.model_index],
            products[row.product_index],
            additional_inputs...,
            time_configuration
        );
    }
};

template<typename Model, typename Curve, typename Product>
struct ModelCurveProductDeviceInputs {
    using ModelParameters = Model;
    using CurveParameters = Curve;
    using ProductParameters = Product;

    const Model* models;
    std::size_t model_count;
    const Curve* curves;
    std::size_t curve_count;
    const Product* products;
    std::size_t product_count;
    bool cartesian_product;

    __device__ __forceinline__ ModelCurveProductIndices indices(
        std::size_t result_index
    ) const {
        return decode_model_curve_product_result_index(
            result_index,
            curve_count,
            product_count,
            cartesian_product
        );
    }

    inline void validate(std::size_t result_count) const {
        validate_device_pointer(models, "device_models");
        validate_device_pointer(curves, "device_curves");
        validate_device_pointer(products, "device_products");
        validate_model_curve_product_construction(
            model_count,
            curve_count,
            product_count,
            cartesian_product,
            result_count
        );
    }

    template<
        typename Pricing,
        typename TimeConfiguration,
        typename... AdditionalInputs
    >
    __device__ __forceinline__ typename Pricing::PreparedRow prepare_row(
        std::size_t result_index,
        const TimeConfiguration& time_configuration,
        const AdditionalInputs&... additional_inputs
    ) const {
        const ModelCurveProductIndices row = indices(result_index);
        return Pricing::prepare_row(
            models[row.model_index],
            curves[row.curve_index],
            products[row.product_index],
            additional_inputs...,
            time_configuration
        );
    }
};

template<typename PrimaryInputs, typename Context>
struct DeviceInputsWithContext {
    PrimaryInputs primary;
    Context context;

    inline void validate(std::size_t result_count) const
        requires requires(const Context& device_context) {
            validate_device_context(device_context);
        }
    {
        primary.validate(result_count);
        validate_device_context(context);
    }

    template<typename Pricing, typename TimeConfiguration>
    __device__ __forceinline__ typename Pricing::PreparedRow prepare_row(
        std::size_t result_index,
        const TimeConfiguration& time_configuration
    ) const {
        return primary.template prepare_row<Pricing>(
            result_index,
            time_configuration,
            context
        );
    }
};

template<typename Model, typename Product>
inline ModelProductDeviceInputs<Model, Product>
make_model_product_device_inputs(
    const Model* models,
    std::size_t model_count,
    const Product* products,
    std::size_t product_count,
    bool cartesian_product
) {
    return {
        models,
        model_count,
        products,
        product_count,
        cartesian_product,
    };
}

template<typename Model, typename Curve, typename Product>
inline ModelCurveProductDeviceInputs<Model, Curve, Product>
make_model_curve_product_device_inputs(
    const Model* models,
    std::size_t model_count,
    const Curve* curves,
    std::size_t curve_count,
    const Product* products,
    std::size_t product_count,
    bool cartesian_product
) {
    return {
        models,
        model_count,
        curves,
        curve_count,
        products,
        product_count,
        cartesian_product,
    };
}

template<typename PrimaryInputs, typename Context>
inline DeviceInputsWithContext<PrimaryInputs, Context>
with_device_context(PrimaryInputs primary, Context context) {
    return {primary, context};
}

static_assert(std::is_trivially_copyable_v<
    ModelProductDeviceInputs<float, float>
>);
static_assert(std::is_trivially_copyable_v<
    ModelCurveProductDeviceInputs<float, float, float>
>);

}  // namespace ai_factory::workbench
