// Register-only co-terminal fixed-leg views for Bermudan swaptions.
#pragma once

#include "product/bermudan_swaption/parameters.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace ai_factory::workbench::product {

struct BermudanSwaptionScheduleView {
    float first_exercise_time;
    float payment_interval;
    float coupon_accrual_fraction;
    std::uint32_t total_payment_count;
    std::uint32_t exercise;

    __device__ __forceinline__ float payment_time(
        std::uint32_t payment
    ) const {
        return fmaf(
            static_cast<float>(exercise + payment + 1U),
            payment_interval,
            first_exercise_time
        );
    }

    __device__ __forceinline__ float accrual_fraction(
        std::uint32_t
    ) const {
        return coupon_accrual_fraction;
    }

    __device__ __forceinline__ std::uint32_t payment_count() const {
        return total_payment_count - exercise;
    }
};

static_assert(sizeof(BermudanSwaptionScheduleView) == 20U);

__device__ __forceinline__ BermudanSwaptionScheduleView
make_bermudan_swaption_schedule_view(
    const BermudanSwaptionParameters& product,
    std::uint32_t exercise,
    float time_day_fraction
) {
    return {
        static_cast<float>(product.first_exercise_time_days) * time_day_fraction,
        static_cast<float>(product.payment_interval_days) * time_day_fraction,
        product.accrual_fraction,
        product.payment_count,
        exercise,
    };
}

}  // namespace ai_factory::workbench::product
