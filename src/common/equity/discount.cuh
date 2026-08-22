// Discount policies independent from product payoff and path simulation.
#pragma once

#include "common/equity/concepts.cuh"
#include "common/equity/pricing_inputs.cuh"

#include <cmath>
#include <concepts>
#include <type_traits>

namespace ai_factory::workbench::equity {

template<typename Discount, typename Dynamics>
concept DiscountPolicyFor =
    EquityDynamicsPolicy<Dynamics>
    && std::is_trivially_copyable_v<typename Discount::DeviceInputs>
    && requires(
        const typename Dynamics::Parameters& parameters,
        const typename Discount::DeviceInputs& inputs,
        float time
    ) {
        typename Discount::DeviceInputs;
        {
            Discount::discount_factor(parameters, inputs, time)
        } -> std::same_as<float>;
        {
            Discount::validate_inputs(inputs)
        } -> std::same_as<void>;
    };

// Current equity datasets carry one continuously compounded constant rate in
// the model parameters. A future curve policy can replace this composition.
template<SupportsRiskFreeRate DynamicsPolicy>
struct ConstantRateDiscountPolicy {
    using Dynamics = DynamicsPolicy;
    using DeviceInputs = EmptyDeviceInputs;

    __device__ __forceinline__ static float discount_factor(
        const typename Dynamics::Parameters& parameters,
        const DeviceInputs&,
        float time
    ) {
        return expf(-Dynamics::risk_free_rate(parameters) * time);
    }

    static void validate_inputs(const DeviceInputs&) {}
};

}  // namespace ai_factory::workbench::equity
