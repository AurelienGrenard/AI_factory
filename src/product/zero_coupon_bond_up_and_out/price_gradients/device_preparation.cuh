// Continuous contract coordinates for the grid-monitored bond barrier call.
#pragma once

#include "product/zero_coupon_bond_up_and_out/parameters.hpp"

#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <string_view>

namespace ai_factory::workbench::product::zero_coupon_bond_up_and_out::price_gradients {

struct DevicePreparation {
    using Product = ZeroCouponBondUpAndOutParameters;
    static constexpr bool kSupportsMaturitySensitivity = true;
    static constexpr std::array parameter_names{
        std::string_view{"product.notional"},
        std::string_view{"product.strike"},
        std::string_view{"product.barrier"},
    };

    __host__ __device__ static std::uint32_t terminal_maturity_days(
        const Product& product
    ) {
        // The quoted maturity is the bond maturity U; the option expiry T
        // and monitoring grid are contractual calendar fields and stay fixed.
        return product.bond_maturity_days;
    }

    __host__ __device__ static std::uint32_t terminal_prefix_days(
        const Product& product
    ) {
        return product.option_expiry_days;
    }

    __host__ __device__ static bool valid(const Product& product) {
        return std::isfinite(product.notional) && product.notional > 0.0f
            && std::isfinite(product.strike) && product.strike > 0.0f
            && std::isfinite(product.barrier) && product.barrier > 0.0f
            && product.option_expiry_days > 0U
            && product.bond_maturity_days > product.option_expiry_days;
    }

    __host__ __device__ static float read(
        std::uint8_t index, const Product& product
    ) {
        return index == 0U ? product.notional
            : index == 1U ? product.strike
            : index == 2U ? product.barrier : ::nanf("");
    }

    __host__ __device__ static void write(
        std::uint8_t index, Product& product, float value
    ) {
        if (index == 0U) product.notional = value;
        else if (index == 1U) product.strike = value;
        else if (index == 2U) product.barrier = value;
    }

    __host__ __device__ static std::uint32_t payment_count(const Product&) {
        return 0U;
    }
};

}  // namespace ai_factory::workbench::product::zero_coupon_bond_up_and_out::price_gradients
