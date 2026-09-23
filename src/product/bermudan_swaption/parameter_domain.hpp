// Bermudan-swaption parameter-domain predicates shared by host and device preparation.
#pragma once

#include "product/bermudan_swaption/parameters.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::product::bermudan_swaption {

__host__ __device__ inline bool finite_parameter(float value) {
#if defined(__CUDA_ARCH__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
}

__host__ __device__ inline bool valid_parameters(
    const BermudanSwaptionParameters& product
) {
    if (!finite_parameter(product.notional) || !(product.notional > 0.0f)
        || !finite_parameter(product.strike) || product.strike < 0.0f
        || !finite_parameter(product.accrual_fraction)
        || !(product.accrual_fraction > 0.0f)
        || product.first_exercise_time_days == 0U
        || product.payment_interval_days == 0U
        || product.exercise_count < 2U
        || product.payment_count < product.exercise_count) {
        return false;
    }
    const std::uint64_t final_payment_time =
        static_cast<std::uint64_t>(product.first_exercise_time_days)
        + static_cast<std::uint64_t>(product.payment_count)
            * product.payment_interval_days;
    return final_payment_time
        <= static_cast<std::uint64_t>(0xffffffffU);
}

inline const char* parameter_domain_error(
    const BermudanSwaptionParameters& product
) {
    if (!std::isfinite(product.notional) || !(product.notional > 0.0f))
        return "notional must be finite and positive.";
    if (!std::isfinite(product.strike) || product.strike < 0.0f)
        return "strike must be finite and non-negative.";
    if (!std::isfinite(product.accrual_fraction)
        || !(product.accrual_fraction > 0.0f))
        return "accrual_fraction must be finite and positive.";
    if (product.first_exercise_time_days == 0U
        || product.payment_interval_days == 0U)
        return "exercise and payment day counts must be positive.";
    if (product.exercise_count < 2U)
        return "exercise_count must be at least two.";
    if (product.payment_count < product.exercise_count)
        return "payment_count must be at least exercise_count for a co-terminal swaption.";
    const std::uint64_t final_payment_time =
        static_cast<std::uint64_t>(product.first_exercise_time_days)
        + static_cast<std::uint64_t>(product.payment_count)
            * product.payment_interval_days;
    if (final_payment_time > std::numeric_limits<std::uint32_t>::max())
        return "final payment time exceeds uint32_t.";
    return nullptr;
}

inline void validate_parameters(
    const BermudanSwaptionParameters& product,
    const std::string& prefix
) {
    if (const char* error = parameter_domain_error(product))
        throw std::invalid_argument(prefix + error);
}

}  // namespace ai_factory::workbench::product::bermudan_swaption
