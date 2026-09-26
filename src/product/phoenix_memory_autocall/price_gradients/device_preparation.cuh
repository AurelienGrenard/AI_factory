// Phoenix Memory Autocall device adapter for continuous sensitivities.
#pragma once

#include "common/price_gradients/product_parameter_access.cuh"
#include "common/simulation/calendar.hpp"
#include "product/phoenix_memory_autocall/parameters.hpp"

#include <array>
#include <cstdint>
#include <string_view>

namespace ai_factory::workbench::product::phoenix_memory_autocall::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;

struct DevicePreparation {
    using Product = PhoenixMemoryAutocallParameters;
    static constexpr std::array parameter_names{
        std::string_view{"product.autocall_barrier"},
        std::string_view{"product.coupon_barrier"},
        std::string_view{"product.protection_barrier"},
        std::string_view{"product.annual_coupon_rate"}
    };

    using Calendar = simulation::RegularCalendar;

    __host__ __device__ static Calendar calendar(const Product& product) {
        return {
            product.observation_interval_days,
            product.observation_interval_days == 0U
                ? 0U
                : product.maturity_days / product.observation_interval_days,
        };
    }

    __host__ __device__ static bool valid(const Product& product) {
        return pg::valid_regular_calendar(
                product.maturity_days,
                product.observation_interval_days
            )
            && pg::finite_positive(product.autocall_barrier)
            && pg::finite_positive(product.coupon_barrier)
            && pg::finite_positive(product.protection_barrier)
            && pg::finite_non_negative(product.annual_coupon_rate);
    }

    __host__ __device__ static float read(
        std::uint8_t index,
        const Product& product
    ) {
        return pg::read_product_parameter<
            &Product::autocall_barrier,
            &Product::coupon_barrier,
            &Product::protection_barrier,
            &Product::annual_coupon_rate
        >(index, product);
    }

    __host__ __device__ static void write(
        std::uint8_t index,
        Product& product,
        float value
    ) {
        pg::write_product_parameter<
            &Product::autocall_barrier,
            &Product::coupon_barrier,
            &Product::protection_barrier,
            &Product::annual_coupon_rate
        >(index, product, value);
    }
};

}  // namespace ai_factory::workbench::product::phoenix_memory_autocall::price_gradients
