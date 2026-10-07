// Shared path-product evaluation for rough sensitivity nodes.
#pragma once

#include "common/equity/path_product_policy.cuh"
#include "common/philox.cuh"
#include "common/price_gradients/device_prepared_launch.cuh"
#include "common/price_gradients/sensitivity_parameter.cuh"
#include "common/volterra/hybrid_schedule.cuh"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::volterra::price_gradients {

template<typename Dynamics, typename ProductPolicy,
         typename HybridSchedule>
struct RoughPathPayoffNodePolicy {
    using NodeValue = float;
    struct Metadata {
        typename ProductPolicy::PreparedProduct product;
        float spot_scale;
        float log_scale;
        typename HybridSchedule::PreparedSchedule schedule;
    };

    template<typename Scenario>
    __device__ static Metadata prepare_metadata(
        const Scenario& scenario,
        ::ai_factory::workbench::price_gradients::TimeConfiguration time
    ) {
        const auto day_fraction =
            static_cast<float>(time.simulation_steps_per_day) * time.dt;
        auto schedule = [&] {
            const auto calendar = ProductPolicy::calendar(scenario.product);
            const HybridTimeConfiguration configuration{day_fraction, time.dt};
            if constexpr (requires {
                HybridSchedule::prepare(
                    calendar, configuration,
                    scenario.step_count, scenario.central_step_count
                );
            }) {
                // Keep contractual prefix observations on the central grid.
                // The maturity node changes only the terminal observation.
                return HybridSchedule::prepare(
                    calendar, configuration,
                    scenario.step_count, scenario.central_step_count
                );
            } else {
                return HybridSchedule::prepare(
                    calendar, configuration, scenario.step_count
                );
            }
        }();
        schedule.maturity_years = scenario.maturity_years;
        schedule.time_step = time.dt;
        return {
            ProductPolicy::prepare_product(
                scenario.model, scenario.product,
                {day_fraction, scenario.maturity_years}
            ),
            scenario.spot_scale,
            logf(scenario.spot_scale),
            schedule
        };
    }

    __device__ static float payoff(const Metadata&, float value) {
        return value;
    }

    __device__ static float centered_first(
        const Metadata&, const Metadata&,
        float lower, float upper, float width
    ) {
        return (upper - lower) / width;
    }
};

template<typename Dynamics, typename Handler,
         equity::ObservationCoordinate Coordinate>
struct ScaledRoughPathObserver {
    Handler& handler;
    float spot_scale;
    float log_spot_scale;

    __device__ __forceinline__ float value(
        const typename Dynamics::State& state
    ) const {
        if constexpr (Coordinate == equity::ObservationCoordinate::spot)
            return Dynamics::spot(state) * spot_scale;
        else
            return Dynamics::log_spot(state) + log_spot_scale;
    }

    __device__ __forceinline__ bool on_initial_state(
        const typename Dynamics::State& state
    ) { return handler.on_initial_value(value(state)); }

    __device__ __forceinline__ bool on_observation(
        std::uint32_t index, const typename Dynamics::State& state
    ) { return handler.on_observation(index, value(state)); }
};

// A bounded group shares one stochastic trajectory while retaining separate
// payoff handlers. Eight handlers cap per-thread register/local-memory usage.
template<typename Dynamics, typename ProductPolicy,
         typename HybridSchedule, std::size_t Capacity = 8U>
struct RoughPathPayoffBatch {
    static constexpr std::size_t kCapacity = Capacity;
    using Product = typename ProductPolicy::PreparedProduct;
    using Metadata = typename RoughPathPayoffNodePolicy<
        Dynamics, ProductPolicy, HybridSchedule
    >::Metadata;
    using Handler = typename ProductPolicy::Handler;
    struct Slot {
        Product product{};
        Handler handler{};
        float spot_scale = 1.0f;
        float log_scale = 0.0f;
        float exit_log_spot = 0.0f;
        bool active = true;
    };
    Slot slots[Capacity]{};
    std::size_t count = 0U;
    std::size_t active_count = 0U;

    __device__ void add(const Metadata& metadata) {
        auto& slot = slots[count++];
        slot.product = metadata.product;
        slot.handler = ProductPolicy::make_handler(slot.product);
        slot.spot_scale = metadata.spot_scale;
        slot.log_scale = metadata.log_scale;
        slot.active = true;
        ++active_count;
    }

    __device__ bool observe(
        std::uint32_t observation,
        const typename Dynamics::State& state,
        bool initial
    ) {
        const float log_value = Dynamics::log_spot(state);
        const float spot_value = Dynamics::spot(state);
        for (std::size_t i = 0U; i < count; ++i) {
            auto& slot = slots[i];
            if (!slot.active) continue;
            float coordinate = 0.0f;
            if constexpr (ProductPolicy::kObservationCoordinate
                == equity::ObservationCoordinate::spot) {
                coordinate = spot_value * slot.spot_scale;
            } else {
                coordinate = log_value + slot.log_scale;
            }
            const bool keep = initial
                ? slot.handler.on_initial_value(coordinate)
                : slot.handler.on_observation(observation, coordinate);
            if (!keep) {
                slot.active = false;
                slot.exit_log_spot = log_value;
                --active_count;
            }
        }
        return active_count != 0U;
    }

    __device__ bool on_initial_state(
        const typename Dynamics::State& state
    ) { return observe(0U, state, true); }

    __device__ bool on_observation(
        std::uint32_t observation,
        const typename Dynamics::State& state
    ) { return observe(observation, state, false); }

    __device__ void finalize(
        const typename Dynamics::State& state,
        float* values
    ) const {
        static_assert(Dynamics::kNativeLogSpot);
        for (std::size_t i = 0U; i < count; ++i) {
            const auto& slot = slots[i];
            auto terminal = state;
            terminal.log_spot = (slot.active
                ? Dynamics::log_spot(state)
                : slot.exit_log_spot) + slot.log_scale;
            values[i] = ProductPolicy::template finalize<Dynamics>(
                slot.product, terminal, slot.handler
            );
        }
    }
};

template<typename Dynamics, typename ProductPolicy,
         typename HybridSchedule>
struct PreparedLiftPathEvaluation {
    static constexpr bool kReuseCentralValue = false;
    static constexpr bool kBatchSharedPath = true;
    using NodePolicy = RoughPathPayoffNodePolicy<
        Dynamics, ProductPolicy, HybridSchedule
    >;
    using Metadata = typename NodePolicy::Metadata;

    template<typename Scenario>
    __device__ static void evaluate_batch(
        const typename Dynamics::PreparedDynamics& dynamics,
        const Scenario* scenarios,
        const Metadata* metadata,
        const std::uint16_t* indices,
        std::size_t count,
        ::ai_factory::workbench::price_gradients::TimeConfiguration time,
        philox::PhiloxKey key,
        std::size_t path,
        float* values
    ) {
        RoughPathPayoffBatch<
            Dynamics, ProductPolicy, HybridSchedule
        > batch{};
        for (std::size_t i = 0U; i < count; ++i)
            batch.add(metadata[indices[i]]);
        const auto& first = scenarios[indices[0U]];
        const auto& schedule = metadata[indices[0U]].schedule;
        auto cursor = HybridSchedule::make_cursor(schedule);
        auto state = Dynamics::initial_state(dynamics);
        bool running = HybridSchedule::on_initial_state(
            schedule, cursor, state, batch
        );
        typename Dynamics::RandomContext random(key, path);
        for (std::uint32_t step = 0U;
             running && step < first.step_count; ++step) {
            Dynamics::simulate_one_step(dynamics, random, state);
            running = HybridSchedule::on_step(
                schedule, cursor, step, state, batch
            );
        }
        batch.finalize(state, values);
    }

    template<typename Scenario>
    __device__ static float evaluate(
        const typename Dynamics::PreparedDynamics& dynamics,
        const Scenario& scenario,
        const Metadata& metadata,
        ::ai_factory::workbench::price_gradients::TimeConfiguration time,
        philox::PhiloxKey key,
        std::size_t path
    ) {
        auto handler = ProductPolicy::make_handler(metadata.product);
        ScaledRoughPathObserver<
            Dynamics, decltype(handler),
            ProductPolicy::kObservationCoordinate
        > observer{
            handler, metadata.spot_scale, metadata.log_scale
        };
        const auto& schedule = metadata.schedule;
        auto cursor = HybridSchedule::make_cursor(schedule);
        auto state = Dynamics::initial_state(dynamics);
        bool running = HybridSchedule::on_initial_state(
            schedule, cursor, state, observer
        );
        typename Dynamics::RandomContext random(key, path);
        for (std::uint32_t step = 0U;
             running && step < scenario.step_count; ++step) {
            Dynamics::simulate_one_step(dynamics, random, state);
            running = HybridSchedule::on_step(
                schedule, cursor, step, state, observer
            );
        }
        state.log_spot += metadata.log_scale;
        return ProductPolicy::template finalize<Dynamics>(
            metadata.product, state, handler
        );
    }
};

}  // namespace ai_factory::workbench::volterra::price_gradients
