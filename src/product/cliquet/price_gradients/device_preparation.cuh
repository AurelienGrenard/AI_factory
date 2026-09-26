// Cliquet device adapter for continuous sensitivities.
#pragma once

#include "common/price_gradients/product_parameter_access.cuh"
#include "common/simulation/calendar.hpp"
#include "product/cliquet/parameters.hpp"

#include <array>
#include <cstdint>
#include <string_view>

namespace ai_factory::workbench::product::cliquet::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;

struct DevicePreparation {
    using Product = CliquetParameters;
    static constexpr std::array parameter_names{
        std::string_view{"product.participation_rate"},
        std::string_view{"product.local_floor"},
        std::string_view{"product.local_cap"},
        std::string_view{"product.global_floor"},
        std::string_view{"product.global_cap"}
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
            && pg::finite_non_negative(product.participation_rate)
            && pg::finite_value(product.local_floor)
            && pg::finite_value(product.local_cap)
            && pg::finite_value(product.global_floor)
            && pg::finite_value(product.global_cap)
            && product.local_floor <= product.local_cap
            && product.global_floor <= product.global_cap;
    }

    __host__ __device__ static float read(
        std::uint8_t index,
        const Product& product
    ) {
        return pg::read_product_parameter<
            &Product::participation_rate,
            &Product::local_floor,
            &Product::local_cap,
            &Product::global_floor,
            &Product::global_cap
        >(index, product);
    }

    __host__ __device__ static void write(
        std::uint8_t index,
        Product& product,
        float value
    ) {
        pg::write_product_parameter<
            &Product::participation_rate,
            &Product::local_floor,
            &Product::local_cap,
            &Product::global_floor,
            &Product::global_cap
        >(index, product, value);
    }
};

}  // namespace ai_factory::workbench::product::cliquet::price_gradients
