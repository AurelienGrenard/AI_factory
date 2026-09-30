// Regular European-swaption device adapter for compact sensitivities.
#pragma once

#include "product/european_swaption/parameter_domain.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <string_view>

namespace ai_factory::workbench::product::european_swaption::price_gradients {

struct DevicePreparation {
    using Product = RegularEuropeanSwaptionParameters;
    static constexpr bool kSupportsMaturitySensitivity = true;
    __host__ __device__ static std::uint32_t terminal_maturity_days(
        const Product& product
    ) {
        return product.exercise_time_days
            + product.payment_count * product.payment_interval_days;
    }
    __host__ __device__ static std::uint32_t terminal_prefix_days(
        const Product& product
    ) {
        return product.exercise_time_days
            + (product.payment_count - 1U)
                * product.payment_interval_days;
    }
    static constexpr std::array parameter_names{
        std::string_view{"product.notional"},
        std::string_view{"product.strike"},
        std::string_view{"product.accrual_fraction"},
    };

    __host__ __device__ static bool valid(const Product& product) {
        return valid_parameters(product);
    }

    __host__ __device__ static float read(
        std::uint8_t index,
        const Product& product
    ) {
        switch (index) {
        case 0U: return product.notional;
        case 1U: return product.strike;
        case 2U: return product.accrual_fraction;
        }
        return ::nanf("");
    }

    __host__ __device__ static void write(
        std::uint8_t index,
        Product& product,
        float value
    ) {
        switch (index) {
        case 0U: product.notional = value; break;
        case 1U: product.strike = value; break;
        case 2U: product.accrual_fraction = value; break;
        }
    }

    __host__ __device__ static std::uint32_t payment_count(
        const Product& product
    ) {
        return product.payment_count;
    }
};

}  // namespace ai_factory::workbench::product::european_swaption::price_gradients
