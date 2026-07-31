// CUDA implementation of the analytical Nelson-Siegel curve.
#include "curve/nelson_siegel/term_structure.cuh"

#include <cuda_runtime.h>

namespace ai_factory::workbench::curve {
namespace {

// Evaluate (1 - exp(-x)) / x stably near x = 0.
__device__ __forceinline__ float nelson_siegel_level_loading(float x) {
    if (fabsf(x) < 1.0e-4f) {
        const float x2 = x * x;
        return 1.0f - 0.5f * x + x2 / 6.0f - x2 * x / 24.0f;
    }
    return -expm1f(-x) / x;
}

}  // namespace

// Evaluate the standard continuously compounded zero curve.
__device__ __forceinline__ float nelson_siegel_zero_rate(
    const NelsonSiegelParameters& parameters,
    float maturity
) {
    const float x = maturity / parameters.tau;
    const float decay = expf(-x);
    const float loading = nelson_siegel_level_loading(x);
    return parameters.beta0
        + parameters.beta1 * loading
        + parameters.beta2 * (loading - decay);
}

// Evaluate log P(0,T) directly for stable discount-ratio calculations.
__device__ __forceinline__ float nelson_siegel_log_discount(
    const NelsonSiegelParameters& parameters,
    float maturity
) {
    return -maturity * nelson_siegel_zero_rate(parameters, maturity);
}

// Convert the zero rate into its continuously compounded discount factor.
__device__ __forceinline__ float nelson_siegel_discount_factor(
    const NelsonSiegelParameters& parameters,
    float maturity
) {
    return expf(nelson_siegel_log_discount(parameters, maturity));
}

// Evaluate the analytical instantaneous forward implied by Nelson-Siegel.
__device__ __forceinline__ float nelson_siegel_instantaneous_forward(
    const NelsonSiegelParameters& parameters,
    float maturity
) {
    const float x = maturity / parameters.tau;
    return parameters.beta0
        + expf(-x) * (parameters.beta1 + parameters.beta2 * x);
}

// Differentiate the analytical instantaneous forward with respect to time.
__device__ __forceinline__ float nelson_siegel_forward_derivative(
    const NelsonSiegelParameters& parameters,
    float maturity
) {
    const float x = maturity / parameters.tau;
    return expf(-x)
        * (-parameters.beta1 + parameters.beta2 * (1.0f - x))
        / parameters.tau;
}

// Derive one finite-period forward directly from two log-discount factors.
__device__ __forceinline__ float nelson_siegel_forward_rate(
    const NelsonSiegelParameters& parameters,
    float start,
    float end
) {
    return (
        nelson_siegel_log_discount(parameters, start)
        - nelson_siegel_log_discount(parameters, end)
    ) / (end - start);
}

}  // namespace ai_factory::workbench::curve
