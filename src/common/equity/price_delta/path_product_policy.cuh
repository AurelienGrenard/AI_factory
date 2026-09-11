// Compose coupled spot paths with existing product handlers and paired payoffs.
#pragma once

#include "common/device_inputs.cuh"
#include "common/equity/path_product_policy.cuh"
#include "common/equity/price_delta/spot_bump.cuh"
#include "common/simulation/schedule.cuh"

namespace ai_factory::workbench::equity::price_delta {

template<typename SchedulePolicy, typename ProductPathPolicy, typename PathPolicy>
struct PathProductPriceDeltaPolicy {
    using Schedule = SchedulePolicy;
    using Dynamics = typename Schedule::Dynamics;
    using ModelParameters = typename PathPolicy::ModelParameters;
    using ProductParameters = typename ProductPathPolicy::ProductParameters;
    using TimeConfiguration = typename Schedule::TimeConfiguration;
    using PrimaryInputs = ModelProductDeviceInputs<ModelParameters, ProductParameters>;
    using DeviceInputs = DeviceInputsWithContext<PrimaryInputs, SpotBumpConfiguration>;
    static_assert(std::is_same_v<Dynamics, typename PathPolicy::Dynamics>);

    struct HostInputs {
        const ModelParameters* models;
        std::size_t model_count;
        const ProductParameters* products;
        std::size_t product_count;
        PriceConstruction construction;
        SpotBumpConfiguration bump;

        void validate(std::size_t result_count, const TimeConfiguration& time) const {
            if (models == nullptr || products == nullptr) {
                throw std::invalid_argument("Price-delta requires host model/product mirrors.");
            }
            validate_model_product_construction(model_count, product_count,
                                               construction, result_count);
            for (std::size_t i = 0; i < model_count; ++i) {
                validate_spot_bump(models[i].spot, bump);
            }
            for (std::size_t i = 0; i < product_count; ++i) {
                simulation::validate_calendar(ProductPathPolicy::calendar(products[i]), time);
            }
        }
    };

    struct PreparedRow {
        typename Schedule::PreparedSchedule schedule;
        typename PathPolicy::Prepared path;
        typename ProductPathPolicy::PreparedProduct products[3];
        float bump_width;
    };

    __device__ __forceinline__ static PreparedRow prepare_row(
        const ModelParameters& model, const ProductParameters& product,
        SpotBumpConfiguration configuration, const TimeConfiguration& time
    ) {
        const SpotBump bump = prepare_spot_bump(model.spot, configuration);
        const auto calendar = ProductPathPolicy::calendar(product);
        const float day_fraction = simulation::day_count_year_fraction(1U, time);
        const ProductPreparationContext context{
            day_fraction,
            static_cast<float>(simulation::calendar_maturity_days(calendar)) * day_fraction
        };
        auto lower = model;
        auto upper = model;
        lower.spot = bump.lower;
        upper.spot = bump.upper;
        return {
            Schedule::prepare(PathPolicy::parameters(model, bump), calendar, time),
            PathPolicy::prepare(bump),
            {ProductPathPolicy::prepare_product(model, product, context),
             ProductPathPolicy::prepare_product(lower, product, context),
             ProductPathPolicy::prepare_product(upper, product, context)},
            bump.width
        };
    }

    struct ScenarioObserver {
        typename ProductPathPolicy::Handler handler;
        SpotObservation terminal{};
        bool active = true;

        __device__ __forceinline__ static float coordinate(SpotObservation state) {
            if constexpr (ProductPathPolicy::kObservationCoordinate
                          == ObservationCoordinate::spot) return state.spot;
            else return state.log_spot;
        }
        template<bool Initial>
        __device__ __forceinline__ void observe(
            std::uint32_t observation, SpotObservation state
        ) {
            if (!active) return;
            terminal = state;
            if constexpr (Initial) active = handler.on_initial_value(coordinate(state));
            else active = handler.on_observation(observation, coordinate(state));
        }
    };

    struct Observer {
        const typename PathPolicy::Prepared& path;
        ScenarioObserver (&scenarios)[3];

        template<bool Initial>
        __device__ __forceinline__ bool observe(
            std::uint32_t observation, const typename Dynamics::State& state
        ) {
            scenarios[0].template observe<Initial>(observation,
                PathPolicy::template observe<0U>(path, state));
            scenarios[1].template observe<Initial>(observation,
                PathPolicy::template observe<1U>(path, state));
            scenarios[2].template observe<Initial>(observation,
                PathPolicy::template observe<2U>(path, state));
            return scenarios[0].active || scenarios[1].active || scenarios[2].active;
        }
        __device__ __forceinline__ bool on_initial_state(const typename Dynamics::State& state) {
            return observe<true>(0U, state);
        }
        __device__ __forceinline__ bool on_observation(
            std::uint32_t observation, const typename Dynamics::State& state
        ) {
            return observe<false>(observation, state);
        }
    };

    __device__ __forceinline__ static PairedPayoff evaluate_path(
        const PreparedRow& row, philox::PhiloxKey key, std::size_t path
    ) {
        // The schedule adapter borrows three independent product states. Bound
        // each scenario explicitly; compiled resources still need inspection.
        static_assert(sizeof(ScenarioObserver) <= simulation::kMaximumObservationHandlerBytes);
        ScenarioObserver scenarios[3]{
            {ProductPathPolicy::make_handler(row.products[0])},
            {ProductPathPolicy::make_handler(row.products[1])},
            {ProductPathPolicy::make_handler(row.products[2])}
        };
        Observer observer{row.path, scenarios};
        if constexpr (simulation::TerminalSchedulePolicy<Schedule>) {
            const auto terminal = Schedule::simulate_terminal(row.schedule, key, path);
            observer.scenarios[0].terminal = PathPolicy::template observe<0U>(row.path, terminal);
            observer.scenarios[1].terminal = PathPolicy::template observe<1U>(row.path, terminal);
            observer.scenarios[2].terminal = PathPolicy::template observe<2U>(row.path, terminal);
        } else {
            // Each observer freezes its own terminal state on early termination.
            // The schedule stops only when all three contracts have terminated.
            Schedule::simulate(row.schedule, key, path, observer);
        }
        float payoffs[3];
        #pragma unroll
        for (unsigned int scenario = 0; scenario < 3U; ++scenario) {
            payoffs[scenario] = ProductPathPolicy::template finalize<SpotObservationPolicy>(
                row.products[scenario], observer.scenarios[scenario].terminal,
                observer.scenarios[scenario].handler);
        }
        return {payoffs[0], (payoffs[2] - payoffs[1]) / row.bump_width};
    }
};

}  // namespace ai_factory::workbench::equity::price_delta
