// Asset-or-nothing payoff composed with a terminal equity schedule.
#pragma once

#include "common/device_inputs.cuh"

#include "common/equity/concepts.cuh"
#include "common/equity/discount.cuh"
#include "common/simulation/schedule.cuh"
#include "common/equity/handlers.cuh"
#include "common/equity/path_product_policy.cuh"
#include "common/option_side.cuh"
#include "product/asset_or_nothing_option/parameters.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::product {

template<OptionSide Side>
struct AssetOrNothingOptionPathPolicy {
    using ProductParameters = AssetOrNothingOptionParameters;
    using Calendar = simulation::MaturityCalendar;
    static constexpr equity::ObservationCoordinate kObservationCoordinate =
        equity::ObservationCoordinate::spot;

    struct PreparedProduct {
        float strike;
        float discount;
    };

    struct Handler {
        __device__ __forceinline__ bool on_initial_value(float) { return true; }
        __device__ __forceinline__ bool on_observation(std::uint32_t, float) {
            return true;
        }
    };

    __host__ __device__ static Calendar calendar(
        const ProductParameters& product
    ) {
        return {product.maturity_days};
    }

    template<typename ModelParameters>
    __device__ __forceinline__ static PreparedProduct prepare_product(
        const ModelParameters& model,
        const ProductParameters& product,
        equity::ProductPreparationContext context
    ) {
        return {
            product.strike,
            expf(-model.risk_free_rate * context.maturity_years),
        };
    }

    __device__ __forceinline__ static Handler make_handler(
        const PreparedProduct&
    ) {
        return {};
    }

    template<typename StatePolicy>
    __device__ __forceinline__ static float finalize(
        const PreparedProduct& product,
        const typename StatePolicy::State& terminal,
        const Handler&
    ) {
        const float spot = StatePolicy::spot(terminal);
        const bool pays = Side == OptionSide::call
            ? spot > product.strike
            : spot < product.strike;
        return pays ? product.discount * spot : 0.0f;
    }
};

template<
    simulation::TerminalSchedulePolicy SchedulePolicy,
    OptionSide Side
>
requires equity::SpotDynamicsPolicy<typename SchedulePolicy::Dynamics>
struct AssetOrNothingOptionPricingPolicy {
    using Schedule = SchedulePolicy;
    using Dynamics = typename Schedule::Dynamics;
    using ModelParameters = typename Dynamics::Parameters;
    using ProductParameters = AssetOrNothingOptionParameters;
    using DeviceInputs =
        ModelProductDeviceInputs<ModelParameters, ProductParameters>;
    using TimeConfiguration = typename Schedule::TimeConfiguration;

    struct PreparedRow {
        typename Schedule::PreparedSchedule schedule;
        float strike;
        float discount;
    };

    __device__ __forceinline__ static PreparedRow prepare_row(
        const ModelParameters& model,
        const ProductParameters& product,
        const TimeConfiguration& time_configuration
    ) {
        const typename Schedule::Calendar calendar{product.maturity_days};
        return {
            Schedule::prepare(model, calendar, time_configuration),
            product.strike,
            equity::constant_rate_discount_factor(
                model,
                simulation::day_count_year_fraction(
                    product.maturity_days,
                    time_configuration
                )
            ),
        };
    }

    __device__ __forceinline__ static float evaluate_path(
        const PreparedRow& row,
        philox::PhiloxKey key,
        std::size_t path
    ) {
        const typename Dynamics::State terminal = Schedule::simulate_terminal(
            row.schedule,
            key,
            path
        );
        const float terminal_spot = Dynamics::spot(terminal);
        const bool pays = Side == OptionSide::call
            ? terminal_spot > row.strike
            : terminal_spot < row.strike;
        return pays ? row.discount * terminal_spot : 0.0f;
    }

};

}  // namespace ai_factory::workbench::product
