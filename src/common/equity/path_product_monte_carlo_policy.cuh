// Compose one model-independent equity path product with a simulation schedule.
#pragma once

#include "common/device_inputs.cuh"
#include "common/equity/path_product_policy.cuh"
#include "common/simulation/schedule.cuh"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::equity {

template<
    simulation::SchedulePolicy SchedulePolicy,
    typename ProductPathPolicy,
    typename DeviceInputsPolicy = ModelProductDeviceInputs<
        typename SchedulePolicy::Dynamics::Parameters,
        typename ProductPathPolicy::ProductParameters
    >
>
requires EquityPathProductPolicy<
    ProductPathPolicy,
    typename SchedulePolicy::Dynamics
>
struct PathProductMonteCarloPricingPolicy {
    using Schedule = SchedulePolicy;
    using Dynamics = typename Schedule::Dynamics;
    using ModelParameters = typename Dynamics::Parameters;
    using ProductParameters = typename ProductPathPolicy::ProductParameters;
    using DeviceInputs = DeviceInputsPolicy;
    using TimeConfiguration = typename Schedule::TimeConfiguration;

    struct PreparedRow {
        typename Schedule::PreparedSchedule schedule;
        typename ProductPathPolicy::PreparedProduct product;
    };

    __device__ __forceinline__ static ProductPreparationContext
    preparation_context(
        const typename ProductPathPolicy::Calendar& calendar,
        const TimeConfiguration& time_configuration
    ) {
        const std::uint64_t maturity_days =
            simulation::calendar_maturity_days(calendar);
        const float day_fraction = simulation::day_count_year_fraction(
            1U,
            time_configuration
        );
        return {
            day_fraction,
            static_cast<float>(maturity_days) * day_fraction,
        };
    }

    __device__ __forceinline__ static PreparedRow prepare_row(
        const ModelParameters& model,
        const ProductParameters& product,
        const TimeConfiguration& time_configuration
    ) requires simulation::DevicePreparedSchedulePolicy<Schedule> {
        const typename ProductPathPolicy::Calendar calendar =
            ProductPathPolicy::calendar(product);
        return {
            Schedule::prepare(model, calendar, time_configuration),
            ProductPathPolicy::prepare_product(
                model,
                product,
                preparation_context(calendar, time_configuration)
            ),
        };
    }

    template<typename PreparedInput>
    __device__ __forceinline__ static PreparedRow prepare_row(
        const ModelParameters& model,
        const ProductParameters& product,
        const PreparedInput& prepared_input,
        const TimeConfiguration& time_configuration
    ) requires requires(
        const PreparedInput& input,
        const typename ProductPathPolicy::Calendar& calendar
    ) {
        Schedule::prepare_from_input(input, calendar, time_configuration);
    } {
        const typename ProductPathPolicy::Calendar calendar =
            ProductPathPolicy::calendar(product);
        return {
            Schedule::prepare_from_input(
                prepared_input,
                calendar,
                time_configuration
            ),
            ProductPathPolicy::prepare_product(
                model,
                product,
                preparation_context(calendar, time_configuration)
            ),
        };
    }

    __device__ __forceinline__ static float evaluate_path(
        const PreparedRow& row,
        philox::PhiloxKey key,
        std::size_t path
    ) {
        auto handler = ProductPathPolicy::make_handler(row.product);
        typename Dynamics::State terminal{};
        if constexpr (simulation::TerminalSchedulePolicy<Schedule>) {
            terminal = Schedule::simulate_terminal(row.schedule, key, path);
        } else {
            PathProductObservationAdapter<
                Dynamics,
                typename ProductPathPolicy::Handler,
                ProductPathPolicy::kObservationCoordinate
            > adapter{handler};
            static_assert(simulation::ObservationHandlerFor<
                decltype(adapter),
                Dynamics
            >);
            terminal = Schedule::simulate(row.schedule, key, path, adapter);
        }
        return ProductPathPolicy::template finalize<Dynamics>(
            row.product,
            terminal,
            handler
        );
    }
};

}  // namespace ai_factory::workbench::equity
