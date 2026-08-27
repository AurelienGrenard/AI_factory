// Zero-coupon-bond-option policies assembled from model analytics.
#pragma once

#include "common/device_inputs.cuh"
#include "common/option_side.cuh"
#include "common/time_configuration.cuh"
#include "product/zero_coupon_bond_option/parameters.hpp"

#include <cuda_runtime.h>

namespace ai_factory::workbench::fixed_income {

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
        float option_expiry_years;
        float bond_maturity_years;
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
            time::year_fraction(product.option_expiry_days, time_configuration),
            time::year_fraction(product.bond_maturity_days, time_configuration),
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
                row.option_expiry_years,
                row.bond_maturity_years,
                row.strike
            );
        } else {
            return row.notional * zero_coupon_bond_put_price(
                row.model,
                row.model.initial_state,
                0.0f,
                row.option_expiry_years,
                row.bond_maturity_years,
                row.strike
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
        float option_expiry_years;
        float bond_maturity_years;
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
            time::year_fraction(product.option_expiry_days, time_configuration),
            time::year_fraction(product.bond_maturity_days, time_configuration),
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
                row.option_expiry_years,
                row.bond_maturity_years,
                row.strike
            );
        } else {
            return row.notional * zero_coupon_bond_put_price(
                row.model,
                Composition::initial_state(),
                0.0f,
                row.option_expiry_years,
                row.bond_maturity_years,
                row.strike
            );
        }
    }
};

}  // namespace ai_factory::workbench::fixed_income
