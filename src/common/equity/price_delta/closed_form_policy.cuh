// Centered spot bumping of an existing closed-form policy, without new analytics.
#pragma once

#include "common/device_inputs.cuh"
#include "common/equity/price_delta/spot_bump.cuh"

namespace ai_factory::workbench::equity::price_delta {

template<typename PricePolicy>
struct ClosedFormSpotDeltaPolicy {
    using PrimaryInputs = typename PricePolicy::DeviceInputs;
    using ModelParameters = typename PrimaryInputs::ModelParameters;
    using ProductParameters = typename PrimaryInputs::ProductParameters;
    using DeviceInputs = DeviceInputsWithContext<PrimaryInputs, SpotBumpConfiguration>;
    using TimeConfiguration = typename PricePolicy::TimeConfiguration;
    struct PreparedRow {
        typename PricePolicy::PreparedRow central, lower, upper;
        float width;
    };
    __device__ __forceinline__ static PreparedRow prepare_row(
        const ModelParameters& model, const ProductParameters& product,
        SpotBumpConfiguration configuration, const TimeConfiguration& time
    ) {
        const auto bump = prepare_spot_bump(model.spot, configuration);
        auto lower = model;
        auto upper = model;
        lower.spot = bump.lower;
        upper.spot = bump.upper;
        return {PricePolicy::prepare_row(model, product, time),
                PricePolicy::prepare_row(lower, product, time),
                PricePolicy::prepare_row(upper, product, time), bump.width};
    }
    __device__ __forceinline__ static float evaluate_price(const PreparedRow& row) {
        return PricePolicy::evaluate_price(row.central);
    }
    __device__ __forceinline__ static PairedPayoff evaluate_price_delta(const PreparedRow& row) {
        const float lower = PricePolicy::evaluate_price(row.lower);
        const float upper = PricePolicy::evaluate_price(row.upper);
        return {evaluate_price(row), (upper - lower) / row.width};
    }
};

}  // namespace ai_factory::workbench::equity::price_delta
