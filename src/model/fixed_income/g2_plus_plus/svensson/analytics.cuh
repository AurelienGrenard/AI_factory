// G2++ analytics fitted to a Svensson curve.
#pragma once

#include "curve/svensson/term_structure.cuh"
#include "model/fixed_income/g2_plus_plus/fitted_analytics.cuh"

#include <cuda_runtime.h>

namespace ai_factory::workbench::model::fixed_income::g2_plus_plus::svensson {

using CurveAnalyticsProvider = curve::svensson::AnalyticsProvider;
using G2PlusPlusFittedParameters =
    fitted::FittedParameters<CurveAnalyticsProvider>;
using FittedAnalyticsProvider =
    fitted::AnalyticsProvider<CurveAnalyticsProvider>;

static_assert(
    ::ai_factory::workbench::fixed_income::ParametricCurveProvider<
        CurveAnalyticsProvider,
        curve::svensson::SvenssonParameters
    >
);
static_assert(
    ::ai_factory::workbench::fixed_income::ZeroCouponBondProvider<
        FittedAnalyticsProvider,
        G2PlusPlusFittedParameters,
        model::fixed_income::g2::State
    >
);
static_assert(
    ::ai_factory::workbench::fixed_income::BondOptionProvider<
        FittedAnalyticsProvider,
        G2PlusPlusFittedParameters,
        model::fixed_income::g2::State
    >
);

__device__ __forceinline__ G2PlusPlusFittedParameters compose_fitted_model(
    const ModelParameters& model,
    const curve::svensson::SvenssonParameters& initial_curve
) {
    return fitted::compose_fitted_model<CurveAnalyticsProvider>(
        model, initial_curve
    );
}

struct FittedModelComposition {
    using ModelParameters =
        ::ai_factory::workbench::model::fixed_income::g2_plus_plus::ModelParameters;
    using CurveParameters =
        ::ai_factory::workbench::curve::svensson::SvenssonParameters;
    using FittedModel = G2PlusPlusFittedParameters;

    __device__ __forceinline__ static model::fixed_income::g2::State
    initial_state() {
        return {0.0f, 0.0f};
    }

    __device__ __forceinline__ static FittedModel compose(
        const ModelParameters& model,
        const CurveParameters& initial_curve
    ) {
        return compose_fitted_model(model, initial_curve);
    }
};

using BermudanSwaptionAnalyticsPolicy =
    fitted::BermudanSwaptionAnalyticsPolicy<FittedModelComposition>;

using fitted::A;
using fitted::B;
using fitted::discount_factor;
using fitted::forward_rate;
using fitted::log_A;
using fitted::log_discount_factor;
using fitted::log_zero_coupon_bond;
using fitted::payer_swap_value;
using fitted::short_rate;
using fitted::short_rate_shift;
using fitted::swap_rate;
using fitted::zero_coupon_bond;
using fitted::zero_coupon_bond_call_price;
using fitted::zero_coupon_bond_put_price;

}  // namespace ai_factory::workbench::model::fixed_income::g2_plus_plus::svensson
