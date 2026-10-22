// Typed regression outcomes and per-row diagnostics for Longstaff-Schwartz.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace ai_factory::workbench::longstaff_schwartz {

enum class RegressionStatus : std::uint32_t {
    success = 0U,
    no_candidates = 1U,
    insufficient_candidates = 2U,
    non_finite_statistics = 3U,
    factorization_failure = 4U,
    non_finite_coefficients = 5U,
};

__host__ __device__ constexpr bool regression_status_is_fatal(
    RegressionStatus status
) {
    return status == RegressionStatus::non_finite_statistics
        || status == RegressionStatus::factorization_failure
        || status == RegressionStatus::non_finite_coefficients;
}

constexpr const char* regression_status_name(RegressionStatus status) {
    switch (status) {
        case RegressionStatus::success: return "success";
        case RegressionStatus::no_candidates: return "no_candidates";
        case RegressionStatus::insufficient_candidates:
            return "insufficient_candidates";
        case RegressionStatus::non_finite_statistics:
            return "non_finite_statistics";
        case RegressionStatus::factorization_failure:
            return "factorization_failure";
        case RegressionStatus::non_finite_coefficients:
            return "non_finite_coefficients";
    }
    return "unknown";
}

struct RegressionDiagnostics {
    std::uint32_t successful_regression_count;
    std::uint32_t no_candidate_count;
    std::uint32_t insufficient_candidate_count;
    std::uint32_t non_finite_statistics_count;
    std::uint32_t factorization_failure_count;
    std::uint32_t non_finite_coefficient_count;
    std::uint32_t fatal_failure_count;
};

__device__ __forceinline__ void record_regression_status(
    RegressionStatus status,
    RegressionDiagnostics& diagnostics
) {
    switch (status) {
        case RegressionStatus::success:
            ++diagnostics.successful_regression_count;
            break;
        case RegressionStatus::no_candidates:
            ++diagnostics.no_candidate_count;
            break;
        case RegressionStatus::insufficient_candidates:
            ++diagnostics.insufficient_candidate_count;
            break;
        case RegressionStatus::non_finite_statistics:
            ++diagnostics.non_finite_statistics_count;
            ++diagnostics.fatal_failure_count;
            break;
        case RegressionStatus::factorization_failure:
            ++diagnostics.factorization_failure_count;
            ++diagnostics.fatal_failure_count;
            break;
        case RegressionStatus::non_finite_coefficients:
            ++diagnostics.non_finite_coefficient_count;
            ++diagnostics.fatal_failure_count;
            break;
    }
}

__device__ __forceinline__ bool invalidate_regression_result_if_fatal(
    const RegressionDiagnostics& diagnostics,
    std::size_t result_index,
    float* prices,
    float* standard_errors
) {
    if (diagnostics.fatal_failure_count == 0U) return false;
#if defined(__CUDA_ARCH__)
    const float invalid = __int_as_float(0x7fffffff);
#else
    const float invalid = std::numeric_limits<float>::quiet_NaN();
#endif
    prices[result_index] = invalid;
    standard_errors[result_index] = invalid;
    return true;
}

}  // namespace ai_factory::workbench::longstaff_schwartz
