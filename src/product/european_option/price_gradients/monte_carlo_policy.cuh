// European terminal gradients with an explicit discounted-payoff contraction.
#pragma once
#include "product/european_option/pricing_policy.cuh"

namespace ai_factory::workbench::product {
template<OptionSide Side>
struct EuropeanOptionGradientPathPolicy : EuropeanOptionPathPolicy<Side> {
    using Base = EuropeanOptionPathPolicy<Side>;
    using PreparedProduct = typename Base::PreparedProduct;

    template<typename Observation>
    __device__ __forceinline__ static float centered_difference(
        PreparedProduct first, PreparedProduct second,
        const typename Observation::State& first_state,
        const typename Observation::State& second_state, float width
    ) {
        const float first_discount = first.discount, second_discount = second.discount;
        const float first_payoff = payoff::vanilla_option_payoff<Side>(
            fmaf(first_state.unscaled_spot, first_state.scale, -first.strike), 0.f);
        const float second_payoff = payoff::vanilla_option_payoff<Side>(
            fmaf(second_state.unscaled_spot, second_state.scale, -second.strike), 0.f);
        // Historical CUDA delta contracts the upper discount with subtraction.
        // Spell this out: optimization of unrelated scenarios must not change it.
        return fmaf(second_discount, second_payoff, -__fmul_rn(first_discount, first_payoff))/width;
    }
};
}  // namespace ai_factory::workbench::product
