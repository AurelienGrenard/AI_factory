// Minimal compile-time contracts for reusable fixed-income analytics.
#pragma once

#include "common/fixed_income/one_factor_affine.cuh"

#include <concepts>

namespace ai_factory::workbench::fixed_income {

template<typename Provider, typename Parameters, typename State>
concept ZeroCouponBondProvider = requires(
    const Provider& provider,
    const Parameters& parameters,
    const State& state,
    float valuation_time,
    float maturity
) {
    {
        provider.zero_coupon_bond(
            parameters, state, valuation_time, maturity
        )
    } -> std::same_as<float>;
};

template<typename Provider, typename Parameters, typename State>
concept OneFactorAffineBondProvider =
    ZeroCouponBondProvider<Provider, Parameters, State>
    && requires(
        const Provider& provider,
        const Parameters& parameters,
        float valuation_time,
        float maturity
    ) {
        {
            provider.affine_bond_coefficients(
                parameters, valuation_time, maturity
            )
        } -> std::same_as<OneFactorAffineBondCoefficients>;
    };

template<typename Provider, typename Parameters, typename State>
concept BondOptionProvider = requires(
    const Provider& provider,
    const Parameters& parameters,
    const State& state,
    float option_sign,
    float valuation_time,
    float option_expiry,
    float bond_maturity,
    float strike
) {
    provider.prepare_bond_option_context(
        parameters, state, valuation_time, option_expiry
    );
    {
        provider.bond_option_price(
            provider.prepare_bond_option_context(
                parameters, state, valuation_time, option_expiry
            ),
            parameters,
            state,
            option_sign,
            valuation_time,
            option_expiry,
            bond_maturity,
            strike
        )
    } -> std::same_as<float>;
};

template<typename Provider, typename Parameters, typename State>
concept JamshidianAnalyticsProvider =
    OneFactorAffineBondProvider<Provider, Parameters, State>
    && BondOptionProvider<Provider, Parameters, State>;

template<typename Provider, typename CurveParameters>
concept ParametricCurveProvider = requires(
    const Provider& provider,
    const CurveParameters& curve,
    float time
) {
    { provider.log_discount_factor(curve, time) } -> std::same_as<float>;
    { provider.instantaneous_forward(curve, time) } -> std::same_as<float>;
};

}  // namespace ai_factory::workbench::fixed_income
