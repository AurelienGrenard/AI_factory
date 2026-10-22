// Resolve model-independent fixed-income quotes against the unbumped t=0 term structure.
#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace ai_factory::workbench::fixed_income::quotes {

template<typename Product>
struct ResolvedProduct {
    Product absolute;
    float strike_reference = 0.0f;
    float barrier_reference = 0.0f;
    bool quoted = false;
};

template<typename Model>
struct StandaloneBonds {
    const Model& model;
    __device__ __forceinline__ float operator()(float maturity) const {
        // ADL selects the analytics belonging to this standalone model.
        return zero_coupon_bond(model, model.initial_state, 0.0f, maturity);
    }
};

template<typename Curve>
struct InitialCurveBonds {
    const Curve& curve;
    __device__ __forceinline__ float operator()(float maturity) const {
        // Fitted ++ models reproduce their supplied initial discount curve.
        return discount_factor(curve, maturity);
    }
};

template<typename Product, typename Bonds>
__device__ __forceinline__ ResolvedProduct<Product> resolve_with_bonds(
    const Product& quote, float day_fraction, const Bonds& bond
) {
    ResolvedProduct<Product> result{quote};
    if (!quote.quote_is_relative) return result;
    result.quoted = true;
    if constexpr (requires { quote.option_expiry_days; quote.bond_maturity_days; }) {
        const float expiry = quote.option_expiry_days * day_fraction;
        const float maturity = quote.bond_maturity_days * day_fraction;
        const float p0u = bond(maturity);
        result.strike_reference = p0u / bond(expiry);
        result.absolute.strike = quote.strike * result.strike_reference;
        if constexpr (requires { quote.barrier; }) {
            result.barrier_reference = p0u;
            result.absolute.barrier = quote.barrier * p0u;
        }
    } else if constexpr (requires { quote.fixing_time_days; }) {
        const float fixing = quote.fixing_time_days * day_fraction;
        const float payment = quote.payment_time_days * day_fraction;
        const float accrual = quote.accrual_period_days * day_fraction;
        result.strike_reference =
            (bond(fixing) / bond(payment) - 1.0f) / accrual;
        result.absolute.strike = result.strike_reference + quote.strike;
    } else if constexpr (requires { quote.payment_count; quote.payment_interval_days; }) {
        const float start = [&]() {
            if constexpr (requires { quote.first_exercise_time_days; })
                return quote.first_exercise_time_days * day_fraction;
            else return quote.exercise_time_days * day_fraction;
        }();
        const float interval = quote.payment_interval_days * day_fraction;
        float annuity = 0.0f;
        float last_bond = 0.0f;
        for (std::uint32_t payment = 1U; payment <= quote.payment_count; ++payment) {
            last_bond = bond(fmaf(static_cast<float>(payment), interval, start));
            annuity = fmaf(quote.accrual_fraction, last_bond, annuity);
        }
        result.strike_reference = (bond(start) - last_bond) / annuity;
        result.absolute.strike = result.strike_reference + quote.strike;
    }
    result.absolute.quote_is_relative = false;
    return result;
}

template<typename Model, typename Product, typename Time>
__host__ __device__ inline ResolvedProduct<Product> resolve(
    const Model& model, const Product& quote, const Time& time
) {
    if (!quote.quote_is_relative) return {quote};
#if defined(__CUDA_ARCH__)
    const float day_fraction = [&]() {
        if constexpr (requires { time.day_fraction; }) return time.day_fraction;
        else return time.dt * time.simulation_steps_per_day;
    }();
    return resolve_with_bonds(quote, day_fraction, StandaloneBonds<Model>{model});
#else
    // The host preflight validates quote coordinates. Device preparation
    // computes the true model-specific reference before constructing a task.
    ResolvedProduct<Product> result{quote};
    result.quoted = true;
    result.strike_reference = 0.05f;
    result.barrier_reference = 1.0f;
    if constexpr (requires { quote.option_expiry_days; }) {
        result.strike_reference = 1.0f;
        result.absolute.strike = quote.strike;
    } else result.absolute.strike = result.strike_reference + quote.strike;
    result.absolute.quote_is_relative = false;
    return result;
#endif
}

template<typename Model, typename Curve, typename Product, typename Time>
__host__ __device__ inline ResolvedProduct<Product> resolve(
    const Model&, const Curve& curve, const Product& quote, const Time& time
) {
    if (!quote.quote_is_relative) return {quote};
#if defined(__CUDA_ARCH__)
    const float day_fraction = [&]() {
        if constexpr (requires { time.day_fraction; }) return time.day_fraction;
        else return time.dt * time.simulation_steps_per_day;
    }();
    return resolve_with_bonds(quote, day_fraction, InitialCurveBonds<Curve>{curve});
#else
    return resolve(curve, quote, time);
#endif
}

template<typename Product>
__host__ __device__ inline float quoted_coordinate(
    const Product& absolute, std::uint8_t product_parameter,
    float strike_reference, float barrier_reference, bool quoted
) {
    if (product_parameter == 2U) {
        if constexpr (requires { absolute.barrier; }) {
            return quoted ? absolute.barrier / barrier_reference : absolute.barrier;
        }
    }
    if (!quoted) return absolute.strike;
    if constexpr (requires { absolute.option_expiry_days; })
        return absolute.strike / strike_reference;
    else return absolute.strike - strike_reference;
}

template<typename Product>
__host__ __device__ inline float absolute_coordinate(
    float quote, std::uint8_t product_parameter,
    float strike_reference, float barrier_reference, bool quoted
) {
    if (!quoted) return quote;
    if (product_parameter == 2U) {
        if constexpr (requires { Product{}.barrier; })
            return quote * barrier_reference;
    }
    if constexpr (requires { Product{}.option_expiry_days; })
        return quote * strike_reference;
    else return quote + strike_reference;
}

}  // namespace ai_factory::workbench::fixed_income::quotes
