// Product payoff policy shared by terminal rough lift and Gaussian FFT graphs.
#pragma once

#include "common/monte_carlo/price_gradients/terminal_node_graph/reconstruction.cuh"
#include "common/price_gradients/time_configuration.hpp"

#include <cuda_runtime.h>

namespace ai_factory::workbench::volterra::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace mcpg = ::ai_factory::workbench::monte_carlo::price_gradients;

template<typename Dynamics, typename ProductPolicy>
struct PreparedLiftTerminalNodePolicy {
    using NodeValue = float;
    using PreparedProduct = typename ProductPolicy::PreparedProduct;
    using Observation = mcpg::TerminalSpotObservation;
    struct Metadata {
        PreparedProduct product;
        float spot_scale;
    };

    template<typename Scenario>
    __device__ static Metadata prepare_metadata(
        const Scenario& scenario, pg::TimeConfiguration time
    ) {
        return {
            ProductPolicy::prepare_product(
                scenario.model, scenario.product,
                {static_cast<float>(time.simulation_steps_per_day) * time.dt,
                 scenario.maturity_years}
            ),
            scenario.spot_scale
        };
    }

    __device__ static float payoff(const Metadata& metadata, float value) {
        const auto handler = ProductPolicy::make_handler(metadata.product);
        return ProductPolicy::template finalize<Observation>(
            metadata.product, {value, metadata.spot_scale}, handler
        );
    }

    __device__ static float centered_first(
        const Metadata& lower, const Metadata& upper,
        float lower_value, float upper_value, float width
    ) {
        if constexpr (requires(
            PreparedProduct first,
            PreparedProduct second,
            typename Observation::State first_state,
            typename Observation::State second_state
        ) {
            ProductPolicy::template centered_difference<Observation>(
                first, second, first_state, second_state, 1.0f
            );
        }) {
            return ProductPolicy::template centered_difference<Observation>(
                lower.product, upper.product,
                {lower_value, lower.spot_scale},
                {upper_value, upper.spot_scale}, width
            );
        } else {
            return (payoff(upper, upper_value)
                    - payoff(lower, lower_value)) / width;
        }
    }
};

}  // namespace ai_factory::workbench::volterra::price_gradients
