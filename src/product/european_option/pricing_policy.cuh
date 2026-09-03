// European-option payoff composed with a terminal equity schedule.
#pragma once

#include "common/device_inputs.cuh"

#include "common/equity/concepts.cuh"
#include "common/equity/discount.cuh"
#include "common/equity/path_product_policy.cuh"
#include "common/simulation/schedule.cuh"
#include "common/equity/handlers.cuh"
#include "common/payoff/vanilla_option.cuh"
#include "product/european_option/parameters.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product {

// Schedule-independent terminal payoff used by non-Markovian path engines.
// The engine owns path construction and observation dispatch; this policy
// owns only the contract calendar, prepared payoff, handler and final value.
template<OptionSide Side>
struct EuropeanOptionPathPolicy {
    using ProductParameters = EuropeanOptionParameters;
    using Calendar = simulation::MaturityCalendar;
    static constexpr equity::ObservationCoordinate kObservationCoordinate =
        equity::ObservationCoordinate::spot;

    struct PreparedProduct {
        float strike;
        float discount;
    };

    struct Handler {
        __device__ __forceinline__ bool on_initial_value(float) {
            return true;
        }

        __device__ __forceinline__ bool on_observation(
            std::uint32_t,
            float
        ) {
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
        return product.discount * payoff::vanilla_option_payoff<Side>(
            StatePolicy::spot(terminal),
            product.strike
        );
    }
};

static_assert(std::is_trivially_copyable_v<
    EuropeanOptionPathPolicy<OptionSide::call>::PreparedProduct
>);

template<
    simulation::TerminalSchedulePolicy SchedulePolicy,
    OptionSide Side
>
requires equity::SpotDynamicsPolicy<typename SchedulePolicy::Dynamics>
struct EuropeanOptionPricingPolicy {
    using Schedule = SchedulePolicy;
    using Dynamics = typename Schedule::Dynamics;
    using ModelParameters = typename Dynamics::Parameters;
    using ProductParameters = EuropeanOptionParameters;
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
    ) requires simulation::DevicePreparedSchedulePolicy<Schedule> {
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

    // Expensive numerical coefficients may be prepared once on the host and
    // supplied through a result-row device context. Product/calendar assembly
    // remains identical to the ordinary device-preparation path.
    template<typename PreparedDynamics>
    __device__ __forceinline__ static PreparedRow prepare_row(
        const ModelParameters& model,
        const ProductParameters& product,
        const PreparedDynamics& dynamics,
        const TimeConfiguration& time_configuration
    ) requires requires(
        const PreparedDynamics& prepared_dynamics,
        const typename Schedule::Calendar& calendar
    ) {
        Schedule::prepare_from_dynamics(
            prepared_dynamics,
            calendar,
            time_configuration
        );
    } {
        const typename Schedule::Calendar calendar{product.maturity_days};
        return {
            Schedule::prepare_from_dynamics(
                dynamics,
                calendar,
                time_configuration
            ),
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
        return row.discount
            * payoff::vanilla_option_payoff<Side>(Dynamics::spot(terminal), row.strike);
    }

};

}  // namespace ai_factory::workbench::product
