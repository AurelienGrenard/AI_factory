// Rate-option policies assembled from model bond-option analytics.
#pragma once

#include "common/device_inputs.cuh"
#include "common/option_side.cuh"
#include "common/time_configuration.cuh"
#include "product/rate_option/parameters.hpp"

#include <cuda_runtime.h>

namespace ai_factory::workbench::fixed_income {

template<typename Model, OptionSide Side>
struct StandaloneRateOptionClosedFormPricingPolicy {
    using DeviceInputs = ModelProductDeviceInputs<
        Model,
        product::RateOptionParameters
    >;
    using TimeConfiguration = time::DayFractionTimeConfiguration;

    struct PreparedRow {
        Model model;
        float bond_option_scale;
        float bond_strike;
        float fixing_time_years;
        float payment_time_years;
    };

    __device__ __forceinline__ static PreparedRow prepare_row(
        const Model& model,
        const product::RateOptionParameters& product,
        const TimeConfiguration& time_configuration
    ) {
        const float strike_factor = fmaf(
            time::year_fraction(product.accrual_period_days, time_configuration),
            product.strike,
            1.0f
        );
        return {
            model,
            product.notional * strike_factor,
            1.0f / strike_factor,
            time::year_fraction(product.fixing_time_days, time_configuration),
            time::year_fraction(product.payment_time_days, time_configuration),
        };
    }

    __device__ __forceinline__ static float evaluate_price(
        const PreparedRow& row
    ) {
        if constexpr (Side == OptionSide::call) {
            return row.bond_option_scale * zero_coupon_bond_put_price(
                row.model,
                row.model.initial_state,
                0.0f,
                row.fixing_time_years,
                row.payment_time_years,
                row.bond_strike
            );
        } else {
            return row.bond_option_scale * zero_coupon_bond_call_price(
                row.model,
                row.model.initial_state,
                0.0f,
                row.fixing_time_years,
                row.payment_time_years,
                row.bond_strike
            );
        }
    }
};

template<typename Composition, OptionSide Side>
struct FittedRateOptionClosedFormPricingPolicy {
    using Model = typename Composition::ModelParameters;
    using Curve = typename Composition::CurveParameters;
    using FittedModel = typename Composition::FittedModel;
    using DeviceInputs = ModelCurveProductDeviceInputs<
        Model,
        Curve,
        product::RateOptionParameters
    >;
    using TimeConfiguration = time::DayFractionTimeConfiguration;

    struct PreparedRow {
        FittedModel model;
        float bond_option_scale;
        float bond_strike;
        float fixing_time_years;
        float payment_time_years;
    };

    __device__ __forceinline__ static PreparedRow prepare_row(
        const Model& model,
        const Curve& initial_curve,
        const product::RateOptionParameters& product,
        const TimeConfiguration& time_configuration
    ) {
        const float strike_factor = fmaf(
            time::year_fraction(product.accrual_period_days, time_configuration),
            product.strike,
            1.0f
        );
        return {
            Composition::compose(model, initial_curve),
            product.notional * strike_factor,
            1.0f / strike_factor,
            time::year_fraction(product.fixing_time_days, time_configuration),
            time::year_fraction(product.payment_time_days, time_configuration),
        };
    }

    __device__ __forceinline__ static float evaluate_price(
        const PreparedRow& row
    ) {
        if constexpr (Side == OptionSide::call) {
            return row.bond_option_scale * zero_coupon_bond_put_price(
                row.model,
                Composition::initial_state(),
                0.0f,
                row.fixing_time_years,
                row.payment_time_years,
                row.bond_strike
            );
        } else {
            return row.bond_option_scale * zero_coupon_bond_call_price(
                row.model,
                Composition::initial_state(),
                0.0f,
                row.fixing_time_years,
                row.payment_time_years,
                row.bond_strike
            );
        }
    }
};

}  // namespace ai_factory::workbench::fixed_income
