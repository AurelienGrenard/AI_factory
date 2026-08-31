// Constant-rate discounting independent from payoff and path simulation.
#pragma once

#include <cmath>
#include <concepts>

namespace ai_factory::workbench::equity {

template<typename Parameters>
concept ConstantRateParameters =
    requires(const Parameters& parameters) {
        { parameters.risk_free_rate } -> std::convertible_to<float>;
    };

// Current equity datasets carry one continuously compounded constant rate in
// the model parameters.
template<ConstantRateParameters Parameters>
__device__ __forceinline__ float constant_rate_discount_factor(
    const Parameters& parameters,
    float time_years
) {
    return expf(-static_cast<float>(parameters.risk_free_rate) * time_years);
}

}  // namespace ai_factory::workbench::equity
