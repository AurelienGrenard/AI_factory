// Rate-option device adapter for compact sensitivity preparation.
#pragma once

#include "product/rate_option/parameter_domain.hpp"

#include <cuda_runtime.h>
#include <array>
#include <cstdint>
#include <string_view>

namespace ai_factory::workbench::product::rate_option::price_gradients {

struct DevicePreparation {
    using Product = RateOptionParameters;
    static constexpr std::array parameter_names{
        std::string_view{"product.notional"},
        std::string_view{"product.strike"},
    };

    __host__ __device__ static bool valid(const Product& product) {
        return valid_parameters(product);
    }
    __host__ __device__ static float read(
        std::uint8_t index, const Product& product
    ) {
        return index == 0U ? product.notional
            : index == 1U ? product.strike : ::nanf("");
    }
    __host__ __device__ static void write(
        std::uint8_t index, Product& product, float value
    ) {
        if (index == 0U) product.notional = value;
        else if (index == 1U) product.strike = value;
    }
    __host__ __device__ static std::uint32_t payment_count(const Product&) {
        return 0U;
    }
};

}  // namespace ai_factory::workbench::product::rate_option::price_gradients
