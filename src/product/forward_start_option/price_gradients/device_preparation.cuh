// Forward Start Option device adapter for continuous sensitivities.
#pragma once

#include "common/price_gradients/product_parameter_access.cuh"
#include "common/simulation/calendar.hpp"
#include "product/forward_start_option/parameters.hpp"

#include <array>
#include <cstdint>
#include <string_view>

namespace ai_factory::workbench::product::forward_start_option::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;

struct DevicePreparation {
    using Product = ForwardStartOptionParameters;
    static constexpr std::array parameter_names{
        std::string_view{"product.moneyness"}
    };

    using Calendar = simulation::StaticCalendar<2U>;

    __host__ __device__ static Calendar calendar(const Product& product) {
        return {{
            product.reset_time_days,
            product.maturity_days - product.reset_time_days,
        }};
    }

    __host__ __device__ static bool valid(const Product& product) {
        return product.reset_time_days > 0U
            && product.reset_time_days < product.maturity_days
            && pg::finite_positive(product.moneyness);
    }

    __host__ __device__ static float read(
        std::uint8_t index,
        const Product& product
    ) {
        return pg::read_product_parameter<
            &Product::moneyness
        >(index, product);
    }

    __host__ __device__ static void write(
        std::uint8_t index,
        Product& product,
        float value
    ) {
        pg::write_product_parameter<
            &Product::moneyness
        >(index, product, value);
    }
};

}  // namespace ai_factory::workbench::product::forward_start_option::price_gradients
