// Device-side schedule views shared by every European-swaption model.
#pragma once

#include "common/check_cuda.cuh"
#include "common/fixed_income/cashflows.cuh"
#include "product/european_swaption/parameters.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::product {

// Empty source tag for schedules carried entirely by regular product rows.
struct RegularEuropeanSwaptionScheduleSource {};

// Dataset-level device pools shared by explicitly scheduled product rows.
struct ExplicitEuropeanSwaptionScheduleSource {
    const std::uint32_t* payment_times_days;
    const float* accrual_fractions;
    std::size_t schedule_size;
    std::size_t schedule_stride = 1U;
};

inline void validate_device_context(
    RegularEuropeanSwaptionScheduleSource
) {}

inline void validate_device_context(
    ExplicitEuropeanSwaptionScheduleSource source
) {
    validate_device_pointer(
        source.payment_times_days,
        "device_payment_times_days"
    );
    validate_device_pointer(
        source.accrual_fractions,
        "device_accrual_fractions"
    );
    if (source.schedule_size == 0U) {
        throw std::invalid_argument(
            "The European swaption schedule pool is empty."
        );
    }
    if (source.schedule_stride == 0U) {
        throw std::invalid_argument(
            "The European swaption schedule stride is zero."
        );
    }
}

// Reconstruct an equally spaced fixed leg without schedule-pool reads.
struct RegularEuropeanSwaptionScheduleView {
    float first_payment_time;
    float payment_interval;
    float coupon_accrual_fraction;
    std::uint32_t count;

    __device__ __forceinline__ bool valid() const {
        return count > 0U
            && isfinite(first_payment_time)
            && first_payment_time > 0.0f
            && isfinite(payment_interval)
            && payment_interval > 0.0f
            && isfinite(coupon_accrual_fraction)
            && coupon_accrual_fraction > 0.0f;
    }

    __device__ __forceinline__ float payment_time(
        std::uint32_t payment
    ) const {
        return fmaf(
            static_cast<float>(payment),
            payment_interval,
            first_payment_time
        );
    }

    __device__ __forceinline__ float accrual_fraction(
        std::uint32_t
    ) const {
        return coupon_accrual_fraction;
    }

    __device__ __forceinline__ std::uint32_t payment_count() const {
        return count;
    }
};

// The product maps its pools onto the common business-day fixed-leg view.
using ExplicitEuropeanSwaptionScheduleView =
    fixed_income::BusinessDayFixedLegScheduleView;

static_assert(sizeof(RegularEuropeanSwaptionScheduleView) == 16U);
static_assert(sizeof(ExplicitEuropeanSwaptionScheduleView) == 32U);
static_assert(sizeof(ExplicitEuropeanSwaptionScheduleSource) == 32U);

// Materialize the register-only view carried by one regular product row.
__device__ __forceinline__ RegularEuropeanSwaptionScheduleView
make_european_swaption_schedule_view(
    const RegularEuropeanSwaptionParameters& product,
    RegularEuropeanSwaptionScheduleSource,
    float time_day_fraction
) {
    const float payment_interval =
        static_cast<float>(product.payment_interval_days) * time_day_fraction;
    return {
        static_cast<float>(product.exercise_time_days) * time_day_fraction
            + payment_interval,
        payment_interval,
        product.accrual_fraction,
        product.payment_count,
    };
}

// Resolve one explicit row into its shared dataset-level schedule pools.
__device__ __forceinline__ ExplicitEuropeanSwaptionScheduleView
make_european_swaption_schedule_view(
    const ExplicitEuropeanSwaptionParameters& product,
    ExplicitEuropeanSwaptionScheduleSource source,
    float time_day_fraction
) {
    const std::size_t payment_count = product.payment_count;
    const bool valid_schedule = product.schedule_offset < source.schedule_size
        && payment_count > 0U
        && payment_count - 1U
            <= (source.schedule_size - 1U - product.schedule_offset)
                / source.schedule_stride;
    return {
        valid_schedule
            ? source.payment_times_days + product.schedule_offset : nullptr,
        valid_schedule
            ? source.accrual_fractions + product.schedule_offset : nullptr,
        product.payment_count,
        time_day_fraction,
        source.schedule_stride,
    };
}

}  // namespace ai_factory::workbench::product
