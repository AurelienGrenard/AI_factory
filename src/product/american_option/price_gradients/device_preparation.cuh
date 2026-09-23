// American-option parameter access for row-local device sensitivity tasks.
#pragma once

#include "product/american_option/parameter_domain.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <string_view>

namespace ai_factory::workbench::product::american_option::price_gradients {

struct DevicePreparation {
    using Product = AmericanOptionParameters;
    static constexpr std::array parameter_names{
        std::string_view{"product.strike"},
    };

    __host__ __device__ static bool valid(const Product& product) {
#if defined(__CUDA_ARCH__)
        return ::isfinite(product.strike) && product.strike > 0.0f
            && product.maturity_days != 0U
            && product.exercise_interval_days != 0U
            && product.exercise_interval_days < product.maturity_days;
#else
        return valid_parameters(product);
#endif
    }

    __host__ __device__ static float read(
        std::uint8_t index,
        const Product& product
    ) {
        return index == 0U ? product.strike : ::nanf("");
    }

    __host__ __device__ static void write(
        std::uint8_t index,
        Product& product,
        float value
    ) {
        if (index == 0U) product.strike = value;
    }
};

}  // namespace ai_factory::workbench::product::american_option::price_gradients
