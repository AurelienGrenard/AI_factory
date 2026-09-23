// Terminal product composition over prepared scenarios and common model innovations.
#pragma once
#include "common/equity/price_gradients/scenarios.hpp"
#include "common/price_gradients/result.cuh"
#include "common/equity/path_product_policy.cuh"

namespace ai_factory::workbench::equity::price_gradients {

template<typename Dynamics, typename ProductPolicy, unsigned int K>
struct TerminalPolicy {
    using Model = typename Dynamics::ModelParameters;
    using Product = typename ProductPolicy::ProductParameters;
    using InputRow = Scenario<Model, Product>;
    static constexpr unsigned int kSensitivities = K;
    static constexpr unsigned int kScenarios = 1U + 2U * K;
    struct PreparedRow {
        typename Dynamics::Prepared dynamics[kScenarios];
        typename ProductPolicy::PreparedProduct products[kScenarios];
        const InputRow* inputs;
        const InputRow* pairs;
        const pg::Stencil* stencils;
        std::uint32_t maximum_steps;
        __device__ __forceinline__ const InputRow& input(unsigned int scenario) const {
            return scenario == 0U ? *inputs : pairs[scenario-1U];
        }
    };
    struct Observation {
        struct State { float unscaled_spot, scale; };
        // Keep scaling adjacent to the consuming payoff, as in the historical
        // multiplicative observer; its native contraction is part of parity.
        __device__ __forceinline__ static float spot(const State& state) { return state.unscaled_spot * state.scale; }
    };
    template<bool IncludeCentral>
    __device__ __forceinline__ static float terminal_spot(
        const PreparedRow& row, const typename Dynamics::State* states, unsigned int scenario
    ) {
        if constexpr (IncludeCentral)
            return Dynamics::spot(row.input(scenario).reuse_central ? states[0] : states[scenario]);
        else return Dynamics::spot(states[scenario-1U]);
    }
    template<bool IncludeCentral>
    __device__ __forceinline__ static float finalize(
        const PreparedRow& row, const typename Dynamics::State* states, unsigned int scenario
    ) {
        const auto handler = ProductPolicy::make_handler(row.products[scenario]);
        const typename Observation::State terminal{terminal_spot<IncludeCentral>(row, states, scenario), row.input(scenario).spot_scale};
        return ProductPolicy::template finalize<Observation>(row.products[scenario], terminal, handler);
    }
    __device__ __forceinline__ static PreparedRow prepare(const InputRow* inputs, const InputRow* pairs,
                                                         const pg::Stencil* stencils, TimeConfiguration time, bool include_central) {
        PreparedRow row{};
        row.inputs = inputs;
        row.pairs = pairs;
        row.stencils = stencils;
        #pragma unroll
        for (unsigned int scenario = 0; scenario < kScenarios; ++scenario) {
            if (scenario == 0U && !include_central) {
                if constexpr (requires { Dynamics::kDrawRequiresCentralPrepared; }) {
                    if constexpr (!Dynamics::kDrawRequiresCentralPrepared) continue;
                } else continue;
            }
            const auto& input = row.input(scenario);
            auto model = input.model;
            model.spot = input.simulation_spot;
            const float horizon = Dynamics::kExactTerminal ? input.maturity_years : time.dt;
            row.dynamics[scenario] = Dynamics::prepare(model, horizon);
            row.products[scenario] = ProductPolicy::prepare_product(input.model, input.product,
                ProductPreparationContext{static_cast<float>(time.simulation_steps_per_day)*time.dt, input.maturity_years});
            row.maximum_steps = max(row.maximum_steps, input.step_count);
        }
        return row;
    }
    template<bool IncludeCentral, bool IncludePayoff>
    __device__ __forceinline__ static pg::Result<K> evaluate_path(const PreparedRow& row, philox::PhiloxKey key, std::size_t path) {
        static_assert(IncludeCentral || !IncludePayoff);
        constexpr unsigned int first_scenario = IncludeCentral ? 0U : 1U;
        typename Dynamics::State states[kScenarios-first_scenario];
        #pragma unroll
        for (unsigned int scenario = first_scenario; scenario < kScenarios; ++scenario)
            states[scenario-first_scenario] = Dynamics::initial(row.dynamics[scenario]);
        typename Dynamics::RandomContext random(key, path);
        if constexpr (Dynamics::kExactTerminal) {
            const auto innovations = [&] {
                if constexpr (requires { Dynamics::draw(random, row.dynamics[0]); })
                    return Dynamics::draw(random, row.dynamics[0]);
                else
                    return Dynamics::draw(random);
            }();
            if constexpr (IncludeCentral)
                Dynamics::transition(row.dynamics[0], innovations, row.input(0).normal_weights, states[0]);
            #pragma unroll
            for (unsigned int scenario = 1; scenario < kScenarios; ++scenario)
                if (!row.input(scenario).reuse_central)
                    Dynamics::transition(row.dynamics[scenario], innovations, row.input(scenario).normal_weights, states[scenario-first_scenario]);
        } else {
            for (std::uint32_t step = 0; step < row.maximum_steps; ++step) {
                const auto innovations = Dynamics::draw(random);
                if constexpr (IncludeCentral)
                    if (step < row.input(0).step_count)
                        Dynamics::transition(row.dynamics[0], innovations, nullptr, states[0]);
                #pragma unroll
                for (unsigned int scenario = 1; scenario < kScenarios; ++scenario) {
                    // Freeze each terminal state on its own grid date.
                    if (!row.input(scenario).reuse_central && step < row.input(scenario).step_count)
                        Dynamics::transition(row.dynamics[scenario], innovations, nullptr, states[scenario-first_scenario]);
                }
            }
        }
        pg::Result<K> result{};
        if constexpr (IncludePayoff) result.price = finalize<IncludeCentral>(row, states, 0U);
        if constexpr (K > 0U) {
            #pragma unroll
            for (unsigned int i = 0; i < K; ++i) {
                const auto& stencil = row.stencils[i];
                if (stencil.kind == pg::StencilKind::centered) {
                    const unsigned int first = 2U*i+1U, second = 2U*i+2U;
                    result.gradients[i] = ProductPolicy::template centered_difference<Observation>(
                        row.products[first], row.products[second],
                        {terminal_spot<IncludeCentral>(row, states, first), row.input(first).spot_scale},
                        {terminal_spot<IncludeCentral>(row, states, second), row.input(second).spot_scale},
                        stencil.represented_width);
                } else if constexpr (IncludePayoff) {
                    const float first = finalize<IncludeCentral>(row, states, 2U*i+1U);
                    const float second = finalize<IncludeCentral>(row, states, 2U*i+2U);
                    result.gradients[i] = pg::difference(stencil, result.price, first, second);
                }
            }
        }
        return result;
    }
};

}  // namespace ai_factory::workbench::equity::price_gradients
