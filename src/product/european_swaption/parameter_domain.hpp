// Regular European-swaption parameter-domain predicates and host validation diagnostics.
#pragma once

#include "product/european_swaption/parameters.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::product::european_swaption {

__host__ __device__ inline bool finite_parameter(float value) {
#if defined(__CUDA_ARCH__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
}

__host__ __device__ inline bool valid_parameters(
    const RegularEuropeanSwaptionParameters& parameters
) {
    if (!finite_parameter(parameters.notional)
        || !(parameters.notional > 0.0f)
        || !finite_parameter(parameters.strike)
        || parameters.strike < 0.0f
        || !finite_parameter(parameters.accrual_fraction)
        || !(parameters.accrual_fraction > 0.0f)
        || parameters.exercise_time_days == 0U
        || parameters.payment_interval_days == 0U
        || parameters.payment_count == 0U) {
        return false;
    }
    const std::uint64_t final_payment_time =
        static_cast<std::uint64_t>(parameters.exercise_time_days)
        + static_cast<std::uint64_t>(parameters.payment_count)
            * parameters.payment_interval_days;
    return final_payment_time
        <= static_cast<std::uint64_t>(0xffffffffU);
}

inline const char* contract_parameter_domain_error(
    float notional,
    float strike,
    std::uint32_t exercise_time_days
) {
    if (!std::isfinite(notional) || !(notional > 0.0f))
        return "notional must be finite and positive.";
    if (!std::isfinite(strike) || strike < 0.0f)
        return "strike must be finite and non-negative for Jamshidian.";
    if (exercise_time_days == 0U)
        return "exercise_time must be a positive day count.";
    return nullptr;
}

inline void validate_contract_parameters(
    float notional,
    float strike,
    std::uint32_t exercise_time_days,
    const std::string& prefix
) {
    if (const char* error = contract_parameter_domain_error(
            notional,
            strike,
            exercise_time_days
        ))
        throw std::invalid_argument(prefix + error);
}

inline const char* parameter_domain_error(
    const RegularEuropeanSwaptionParameters& parameters
) {
    if (valid_parameters(parameters)) return nullptr;
    if (const char* error = contract_parameter_domain_error(
            parameters.notional,
            parameters.strike,
            parameters.exercise_time_days
        ))
        return error;
    if (!std::isfinite(parameters.accrual_fraction)
        || !(parameters.accrual_fraction > 0.0f))
        return "accrual_fraction must be finite and positive.";
    if (parameters.payment_interval_days == 0U)
        return "payment_interval must be a positive day count.";
    if (parameters.payment_count == 0U)
        return "payment_count must be positive.";
    const std::uint64_t final_payment_time =
        static_cast<std::uint64_t>(parameters.exercise_time_days)
        + static_cast<std::uint64_t>(parameters.payment_count)
            * parameters.payment_interval_days;
    if (final_payment_time > std::numeric_limits<std::uint32_t>::max())
        return "final payment time exceeds uint32_t.";
    return nullptr;
}

inline void validate_parameters(
    const RegularEuropeanSwaptionParameters& parameters,
    const std::string& prefix
) {
    if (const char* error = parameter_domain_error(parameters))
        throw std::invalid_argument(prefix + error);
}

}  // namespace ai_factory::workbench::product::european_swaption
