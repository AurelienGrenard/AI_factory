// Gap-option parameter access for device-prepared sensitivities.
#pragma once

#include "price_gradients/product_parameter_access.cuh"
#include "product/gap_option/parameters.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <string_view>

namespace ai_factory::workbench::product::gap_option::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;

struct DevicePreparation {
    using Product = GapOptionParameters;
    static constexpr std::array parameter_names{
        std::string_view{"product.trigger_strike"},
        std::string_view{"product.payoff_strike"},
    };

    __host__ __device__ static bool valid(const Product& product) {
        return pg::finite_positive(product.trigger_strike)
            && pg::finite_positive(product.payoff_strike)
            && product.maturity_days > 0U;
    }

    __host__ __device__ static float read(
        std::uint8_t index,
        const Product& product
    ) {
        switch (index) {
        case 0U: return product.trigger_strike;
        case 1U: return product.payoff_strike;
        }
        return ::nanf("");
    }

    __host__ __device__ static void write(
        std::uint8_t index,
        Product& product,
        float value
    ) {
        switch (index) {
        case 0U: product.trigger_strike = value; break;
        case 1U: product.payoff_strike = value; break;
        }
    }
};

}  // namespace ai_factory::workbench::product::gap_option::price_gradients
