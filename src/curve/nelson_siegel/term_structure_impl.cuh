// CUDA implementation of the analytical Nelson-Siegel curve.
#include "curve/nelson_siegel/term_structure.cuh"

#include "curve/nelson_siegel/instantaneous_forward.cuh"

#include <cuda_runtime.h>

namespace ai_factory::workbench::curve::nelson_siegel {
namespace {

// Evaluate (1 - exp(-x)) / x stably near x = 0.
__device__ __forceinline__ float level_loading(float x) {
    if (fabsf(x) < 1.0e-4f) {
        const float x2 = x * x;
        return 1.0f - 0.5f * x + x2 / 6.0f - x2 * x / 24.0f;
    }
    return -expm1f(-x) / x;
}

}  // namespace

// Evaluate the standard continuously compounded zero curve.
__device__ __forceinline__ float zero_rate(
    const NelsonSiegelParameters& parameters,
    float maturity_years
) {
    const float x = maturity_years / parameters.tau;
    const float decay = expf(-x);
    const float loading = level_loading(x);
    return parameters.beta0
        + parameters.beta1 * loading
        + parameters.beta2 * (loading - decay);
}

// Evaluate log P(0,T) directly for stable discount-ratio calculations.
__device__ __forceinline__ float log_discount_factor(
    const NelsonSiegelParameters& parameters,
    float maturity_years
) {
    return -maturity_years * zero_rate(parameters, maturity_years);
}

// Convert the zero rate into its continuously compounded discount factor.
__device__ __forceinline__ float discount_factor(
    const NelsonSiegelParameters& parameters,
    float maturity_years
) {
    return expf(log_discount_factor(parameters, maturity_years));
}

// Evaluate the analytical instantaneous forward implied by Nelson-Siegel.
__device__ __forceinline__ float instantaneous_forward(
    const NelsonSiegelParameters& parameters,
    float maturity_years
) {
    const float x = maturity_years / parameters.tau;
    return instantaneous_forward_formula(
        parameters.beta0,
        parameters.beta1,
        parameters.beta2,
        x
    );
}

// Differentiate the analytical instantaneous forward with respect to time_years.
__device__ __forceinline__ float forward_derivative(
    const NelsonSiegelParameters& parameters,
    float maturity_years
) {
    const float x = maturity_years / parameters.tau;
    return expf(-x)
        * (-parameters.beta1 + parameters.beta2 * (1.0f - x))
        / parameters.tau;
}

// Derive one finite-period forward directly from two log-discount factors.
__device__ __forceinline__ float forward_rate(
    const NelsonSiegelParameters& parameters,
    float start_years,
    float end_years
) {
    return (
        log_discount_factor(parameters, start_years)
        - log_discount_factor(parameters, end_years)
    ) / (end_years - start_years);
}

}  // namespace ai_factory::workbench::curve::nelson_siegel
