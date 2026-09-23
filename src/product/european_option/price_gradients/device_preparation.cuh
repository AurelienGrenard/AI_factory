// European-option device adapter for parameter access and input validation.
#pragma once

#include "product/european_option/parameter_domain.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <string_view>

namespace ai_factory::workbench::product::european_option::price_gradients {

struct DevicePreparation {
    using Product = EuropeanOptionParameters;
    static constexpr std::array parameter_names{
        std::string_view{"product.strike"},
    };

    __host__ __device__ static bool valid(const Product& product) {
        return valid_parameters(product);
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

}  // namespace ai_factory::workbench::product::european_option::price_gradients
