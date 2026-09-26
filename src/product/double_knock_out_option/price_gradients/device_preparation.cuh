// Double Knock Out Option device adapter for continuous sensitivities.
#pragma once

#include "common/price_gradients/product_parameter_access.cuh"
#include "common/simulation/calendar.hpp"
#include "product/double_knock_out_option/parameters.hpp"

#include <array>
#include <cstdint>
#include <string_view>

namespace ai_factory::workbench::product::double_knock_out_option::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;

struct DevicePreparation {
    using Product = DoubleKnockOutOptionParameters;
    static constexpr std::array parameter_names{
        std::string_view{"product.strike"},
        std::string_view{"product.lower_barrier"},
        std::string_view{"product.upper_barrier"}
    };

    using Calendar = simulation::MaturityCalendar;

    __host__ __device__ static Calendar calendar(const Product& product) {
        return {product.maturity_days};
    }

    __host__ __device__ static bool valid(const Product& product) {
        return product.maturity_days > 0U
            && pg::finite_positive(product.strike)
            && pg::finite_positive(product.lower_barrier)
            && pg::finite_positive(product.upper_barrier)
            && product.lower_barrier < product.upper_barrier;
    }

    __host__ __device__ static float read(
        std::uint8_t index,
        const Product& product
    ) {
        return pg::read_product_parameter<
            &Product::strike,
            &Product::lower_barrier,
            &Product::upper_barrier
        >(index, product);
    }

    __host__ __device__ static void write(
        std::uint8_t index,
        Product& product,
        float value
    ) {
        pg::write_product_parameter<
            &Product::strike,
            &Product::lower_barrier,
            &Product::upper_barrier
        >(index, product, value);
    }
};

}  // namespace ai_factory::workbench::product::double_knock_out_option::price_gradients
