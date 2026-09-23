// Digital-option parameter access for device-prepared sensitivities.
#pragma once

#include "product/digital_option/parameter_domain.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <string_view>

namespace ai_factory::workbench::product::digital_option::price_gradients {

struct DevicePreparation {
    using Product = DigitalOptionParameters;
    static constexpr std::array parameter_names{
        std::string_view{"product.strike"},
        std::string_view{"product.cash_payoff"},
    };

    __host__ __device__ static bool valid(const Product& product) {
        return valid_parameters(product);
    }

    __host__ __device__ static float read(
        std::uint8_t index,
        const Product& product
    ) {
        switch (index) {
        case 0U: return product.strike;
        case 1U: return product.cash_payoff;
        }
        return ::nanf("");
    }

    __host__ __device__ static void write(
        std::uint8_t index,
        Product& product,
        float value
    ) {
        switch (index) {
        case 0U: product.strike = value; break;
        case 1U: product.cash_payoff = value; break;
        }
    }
};

}  // namespace ai_factory::workbench::product::digital_option::price_gradients
