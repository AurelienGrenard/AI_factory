// Reusable closed-form pricing policies built from model bond-option analytics.
#pragma once

#include "common/device_inputs.cuh"
#include "common/option_side.cuh"
#include "common/time_configuration.cuh"
#include "product/rate_option/parameters.hpp"
#include "product/zero_coupon_bond_option/parameters.hpp"

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
        float fixing_time;
        float payment_time;
    };

    __device__ __forceinline__ static PreparedRow prepare_row(
        const Model& model,
        const product::RateOptionParameters& product,
        const TimeConfiguration& time_configuration
    ) {
        const float strike_factor = fmaf(
            time::year_fraction(product.accrual_period, time_configuration),
            product.strike,
            1.0f
        );
        return {
            model,
            product.notional * strike_factor,
            1.0f / strike_factor,
            time::year_fraction(product.fixing_time, time_configuration),
            time::year_fraction(product.payment_time, time_configuration),
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
                row.fixing_time,
                row.payment_time,
                row.bond_strike
            );
        } else {
            return row.bond_option_scale * zero_coupon_bond_call_price(
                row.model,
                row.model.initial_state,
                0.0f,
                row.fixing_time,
                row.payment_time,
                row.bond_strike
            );
        }
    }
};

template<typename Model, OptionSide Side>
struct StandaloneZeroCouponBondOptionClosedFormPricingPolicy {
    using DeviceInputs = ModelProductDeviceInputs<
        Model,
        product::ZeroCouponBondOptionParameters
    >;
    using TimeConfiguration = time::DayFractionTimeConfiguration;

    struct PreparedRow {
        Model model;
        float notional;
        float strike;
        float option_expiry;
        float bond_maturity;
    };

    __device__ __forceinline__ static PreparedRow prepare_row(
        const Model& model,
        const product::ZeroCouponBondOptionParameters& product,
        const TimeConfiguration& time_configuration
    ) {
        return {
            model,
            product.notional,
            product.strike,
            time::year_fraction(product.option_expiry, time_configuration),
            time::year_fraction(product.bond_maturity, time_configuration),
        };
    }

    __device__ __forceinline__ static float evaluate_price(
        const PreparedRow& row
    ) {
        if constexpr (Side == OptionSide::call) {
            return row.notional * zero_coupon_bond_call_price(
                row.model,
                row.model.initial_state,
                0.0f,
                row.option_expiry,
                row.bond_maturity,
                row.strike
            );
        } else {
            return row.notional * zero_coupon_bond_put_price(
                row.model,
                row.model.initial_state,
                0.0f,
                row.option_expiry,
                row.bond_maturity,
                row.strike
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
        float fixing_time;
        float payment_time;
    };

    __device__ __forceinline__ static PreparedRow prepare_row(
        const Model& model,
        const Curve& initial_curve,
        const product::RateOptionParameters& product,
        const TimeConfiguration& time_configuration
    ) {
        const float strike_factor = fmaf(
            time::year_fraction(product.accrual_period, time_configuration),
            product.strike,
            1.0f
        );
        return {
            Composition::compose(model, initial_curve),
            product.notional * strike_factor,
            1.0f / strike_factor,
            time::year_fraction(product.fixing_time, time_configuration),
            time::year_fraction(product.payment_time, time_configuration),
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
                row.fixing_time,
                row.payment_time,
                row.bond_strike
            );
        } else {
            return row.bond_option_scale * zero_coupon_bond_call_price(
                row.model,
                Composition::initial_state(),
                0.0f,
                row.fixing_time,
                row.payment_time,
                row.bond_strike
            );
        }
    }
};

template<typename Composition, OptionSide Side>
struct FittedZeroCouponBondOptionClosedFormPricingPolicy {
    using Model = typename Composition::ModelParameters;
    using Curve = typename Composition::CurveParameters;
    using FittedModel = typename Composition::FittedModel;
    using DeviceInputs = ModelCurveProductDeviceInputs<
        Model,
        Curve,
        product::ZeroCouponBondOptionParameters
    >;
    using TimeConfiguration = time::DayFractionTimeConfiguration;

    struct PreparedRow {
        FittedModel model;
        float notional;
        float strike;
        float option_expiry;
        float bond_maturity;
    };

    __device__ __forceinline__ static PreparedRow prepare_row(
        const Model& model,
        const Curve& initial_curve,
        const product::ZeroCouponBondOptionParameters& product,
        const TimeConfiguration& time_configuration
    ) {
        return {
            Composition::compose(model, initial_curve),
            product.notional,
            product.strike,
            time::year_fraction(product.option_expiry, time_configuration),
            time::year_fraction(product.bond_maturity, time_configuration),
        };
    }

    __device__ __forceinline__ static float evaluate_price(
        const PreparedRow& row
    ) {
        if constexpr (Side == OptionSide::call) {
            return row.notional * zero_coupon_bond_call_price(
                row.model,
                Composition::initial_state(),
                0.0f,
                row.option_expiry,
                row.bond_maturity,
                row.strike
            );
        } else {
            return row.notional * zero_coupon_bond_put_price(
                row.model,
                Composition::initial_state(),
                0.0f,
                row.option_expiry,
                row.bond_maturity,
                row.strike
            );
        }
    }
};

}  // namespace ai_factory::workbench::fixed_income
