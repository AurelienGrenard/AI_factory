// Up No Touch device adapter for continuous sensitivities.
#pragma once

#include "common/price_gradients/product_parameter_access.cuh"
#include "common/simulation/calendar.hpp"
#include "product/up_no_touch/parameters.hpp"

#include <array>
#include <cstdint>
#include <string_view>

namespace ai_factory::workbench::product::up_no_touch::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;

struct DevicePreparation {
    using Product = UpNoTouchParameters;
    static constexpr std::array parameter_names{
        std::string_view{"product.barrier"},
        std::string_view{"product.cash_payoff"}
    };

    using Calendar = simulation::MaturityCalendar;

    __host__ __device__ static Calendar calendar(const Product& product) {
        return {product.maturity_days};
    }

    __host__ __device__ static bool valid(const Product& product) {
        return product.maturity_days > 0U
            && pg::finite_positive(product.barrier)
            && pg::finite_non_negative(product.cash_payoff);
    }

    __host__ __device__ static float read(
        std::uint8_t index,
        const Product& product
    ) {
        return pg::read_product_parameter<
            &Product::barrier,
            &Product::cash_payoff
        >(index, product);
    }

    __host__ __device__ static void write(
        std::uint8_t index,
        Product& product,
        float value
    ) {
        pg::write_product_parameter<
            &Product::barrier,
            &Product::cash_payoff
        >(index, product, value);
    }
};

}  // namespace ai_factory::workbench::product::up_no_touch::price_gradients
