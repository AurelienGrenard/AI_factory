// Device-safe access to a small compile-time list of float product members.
#pragma once

#include <cmath>
#include <cstdint>

namespace ai_factory::workbench::price_gradients {

template<auto First, auto... Rest, typename Product>
__host__ __device__ inline float read_product_parameter(
    std::uint8_t index,
    const Product& product
) {
    if (index == 0U) return product.*First;
    if constexpr (sizeof...(Rest) != 0U) {
        return read_product_parameter<Rest...>(index - 1U, product);
    }
    return ::nanf("");
}

template<auto First, auto... Rest, typename Product>
__host__ __device__ inline void write_product_parameter(
    std::uint8_t index,
    Product& product,
    float value
) {
    if (index == 0U) {
        product.*First = value;
    } else if constexpr (sizeof...(Rest) != 0U) {
        write_product_parameter<Rest...>(index - 1U, product, value);
    }
}

__host__ __device__ inline bool finite_value(float value) {
#if defined(__CUDA_ARCH__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
}

__host__ __device__ inline bool finite_positive(float value) {
    return finite_value(value) && value > 0.0f;
}

__host__ __device__ inline bool finite_non_negative(float value) {
    return finite_value(value) && value >= 0.0f;
}

__host__ __device__ inline bool valid_regular_calendar(
    std::uint32_t maturity_days,
    std::uint32_t observation_interval_days
) {
    return maturity_days > 0U
        && observation_interval_days > 0U
        && maturity_days % observation_interval_days == 0U;
}

}  // namespace ai_factory::workbench::price_gradients
