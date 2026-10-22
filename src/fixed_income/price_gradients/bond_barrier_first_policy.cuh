// First-order CRN sensitivities of a grid-monitored zero-coupon bond barrier.
#pragma once

#include "philox/distributions.cuh"
#include "price_gradients/reconstruction.cuh"
#include "price_gradients/sensitivity_task.cuh"
#include "price_gradients/time_configuration.hpp"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::fixed_income::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;

// Each node replays the same Philox path. This keeps the stochastic discount,
// bond monitoring, and barrier event paired across the finite-difference nodes.
template<typename Pricing, typename Preparation>
struct BondBarrierFirstPolicy {
    static constexpr std::size_t kNodeCapacity = 3U;
    using Scenario = typename Preparation::Scenario;
    using Nodes = pg::SensitivityNodes<Scenario, kNodeCapacity>;
    using Stencil = pg::SensitivityStencil<kNodeCapacity>;

    struct PreparedRow {
        typename Pricing::PreparedRow pricing[kNodeCapacity];
        const Stencil* stencil;
        std::uint8_t node_count;
    };

    __device__ __forceinline__ static PreparedRow prepare(
        const Nodes* nodes,
        const Stencil* stencil,
        pg::TimeConfiguration time,
        bool
    ) {
        PreparedRow row{};
        row.stencil = stencil;
        row.node_count = static_cast<std::uint8_t>(
            pg::active_node_count(*stencil)
        );
        const typename Pricing::TimeConfiguration schedule_time{
            time.dt, time.simulation_steps_per_day
        };
        for (unsigned int node = 0U; node < row.node_count; ++node) {
            const auto& scenario = (*nodes)[node];
            if constexpr (requires { scenario.curve; }) {
                row.pricing[node] = Pricing::prepare_row(
                    scenario.model, scenario.curve, scenario.product,
                    schedule_time
                );
            } else {
                row.pricing[node] = Pricing::prepare_row(
                    scenario.model, scenario.product, schedule_time
                );
            }
            // U is the sole continuously bumped date. T and the observation
            // calendar remain frozen in the product contract.
            row.pricing[node].maturity_years = scenario.maturity_years;
        }
        return row;
    }

    struct PathResult {
        float price;
        float sensitivity;
    };

    template<bool IncludeCentral, bool IncludePayoff>
    __device__ __forceinline__ static PathResult evaluate_path(
        const PreparedRow& row,
        philox::PhiloxKey key,
        std::size_t path
    ) {
        static_assert(IncludeCentral || !IncludePayoff);
        pg::SensitivityValues<kNodeCapacity> values{};
        for (unsigned int node = 0U; node < row.node_count; ++node) {
            values[node] = Pricing::evaluate_path(
                row.pricing[node], key, path
            );
        }
        PathResult result{};
        if constexpr (IncludePayoff) result.price = values[0U];
        result.sensitivity = pg::reconstruct_first_sensitivity(
            *row.stencil, values
        );
        return result;
    }
};

}  // namespace ai_factory::workbench::fixed_income::price_gradients
