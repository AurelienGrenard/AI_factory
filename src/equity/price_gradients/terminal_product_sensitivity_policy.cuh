// Add finite-difference contraction to an existing terminal product policy.
#pragma once

namespace ai_factory::workbench::equity::price_gradients {

template<typename ProductPathPolicy>
struct TerminalProductSensitivityPolicy : ProductPathPolicy {
    using Base = ProductPathPolicy;
    using PreparedProduct = typename Base::PreparedProduct;

    template<typename Observation>
    __device__ __forceinline__ static float centered_difference(
        const PreparedProduct& first,
        const PreparedProduct& second,
        const typename Observation::State& first_state,
        const typename Observation::State& second_state,
        float width
    ) {
        const auto first_handler = Base::make_handler(first);
        const auto second_handler = Base::make_handler(second);
        const float first_value = Base::template finalize<Observation>(
            first, first_state, first_handler
        );
        const float second_value = Base::template finalize<Observation>(
            second, second_state, second_handler
        );
        return (second_value - first_value) / width;
    }
};

}  // namespace ai_factory::workbench::equity::price_gradients
