// Range Accrual device adapter for continuous sensitivities.
#pragma once

#include "common/price_gradients/product_parameter_access.cuh"
#include "common/simulation/calendar.hpp"
#include "product/range_accrual/parameters.hpp"

#include <array>
#include <cstdint>
#include <string_view>

namespace ai_factory::workbench::product::range_accrual::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;

struct DevicePreparation {
    using Product = RangeAccrualParameters;
    static constexpr std::array parameter_names{
        std::string_view{"product.lower_barrier"},
        std::string_view{"product.upper_barrier"},
        std::string_view{"product.coupon_rate"}
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
            && pg::finite_positive(product.lower_barrier)
            && pg::finite_positive(product.upper_barrier)
            && product.lower_barrier < product.upper_barrier
            && pg::finite_non_negative(product.coupon_rate);
    }

    __host__ __device__ static float read(
        std::uint8_t index,
        const Product& product
    ) {
        return pg::read_product_parameter<
            &Product::lower_barrier,
            &Product::upper_barrier,
            &Product::coupon_rate
        >(index, product);
    }

    __host__ __device__ static void write(
        std::uint8_t index,
        Product& product,
        float value
    ) {
        pg::write_product_parameter<
            &Product::lower_barrier,
            &Product::upper_barrier,
            &Product::coupon_rate
        >(index, product, value);
    }
};

}  // namespace ai_factory::workbench::product::range_accrual::price_gradients
