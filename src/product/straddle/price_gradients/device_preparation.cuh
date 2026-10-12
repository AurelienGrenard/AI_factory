// Straddle parameter access for device-prepared sensitivities.
#pragma once

#include "common/price_gradients/product_parameter_access.cuh"
#include "product/straddle/parameters.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <string_view>

namespace ai_factory::workbench::product::straddle::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;

struct DevicePreparation {
    using Product = StraddleParameters;
    static constexpr std::array parameter_names{
        std::string_view{"product.strike"},
    };

    __host__ __device__ static bool valid(const Product& product) {
        return pg::finite_positive(product.strike)
            && product.maturity_days > 0U;
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

}  // namespace ai_factory::workbench::product::straddle::price_gradients
