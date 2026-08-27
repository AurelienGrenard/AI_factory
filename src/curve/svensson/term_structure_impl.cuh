// CUDA implementation of the analytical Svensson curve.
#include "curve/svensson/term_structure.cuh"

#include "curve/svensson/instantaneous_forward.cuh"

#include <cuda_runtime.h>

namespace ai_factory::workbench::curve::svensson {
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
    const SvenssonParameters& parameters,
    float maturity
) {
    const float x1 = maturity / parameters.tau1;
    const float x2 = maturity / parameters.tau2;
    const float decay1 = expf(-x1);
    const float decay2 = expf(-x2);
    const float loading1 = level_loading(x1);
    const float loading2 = level_loading(x2);
    return parameters.beta0
        + parameters.beta1 * loading1
        + parameters.beta2 * (loading1 - decay1)
        + parameters.beta3 * (loading2 - decay2);
}

// Evaluate log P(0,T) directly for stable discount-ratio calculations.
__device__ __forceinline__ float log_discount_factor(
    const SvenssonParameters& parameters,
    float maturity
) {
    return -maturity * zero_rate(parameters, maturity);
}

// Convert the zero rate into its continuously compounded discount factor.
__device__ __forceinline__ float discount_factor(
    const SvenssonParameters& parameters,
    float maturity
) {
    return expf(log_discount_factor(parameters, maturity));
}

// Evaluate the analytical instantaneous forward implied by Svensson.
__device__ __forceinline__ float instantaneous_forward(
    const SvenssonParameters& parameters,
    float maturity
) {
    return instantaneous_forward_formula(
        parameters.beta0,
        parameters.beta1,
        parameters.beta2,
        parameters.beta3,
        parameters.tau1,
        parameters.tau2,
        maturity
    );
}

// Differentiate the analytical instantaneous forward with respect to time.
__device__ __forceinline__ float forward_derivative(
    const SvenssonParameters& parameters,
    float maturity
) {
    const float x1 = maturity / parameters.tau1;
    const float x2 = maturity / parameters.tau2;
    const float first_component = expf(-x1)
        * (-parameters.beta1 + parameters.beta2 * (1.0f - x1))
        / parameters.tau1;
    const float second_component = parameters.beta3 * expf(-x2)
        * (1.0f - x2) / parameters.tau2;
    return first_component + second_component;
}

// Derive one finite-period forward directly from two log-discount factors.
__device__ __forceinline__ float forward_rate(
    const SvenssonParameters& parameters,
    float start,
    float end
) {
    return (
        log_discount_factor(parameters, start)
        - log_discount_factor(parameters, end)
    ) / (end - start);
}

}  // namespace ai_factory::workbench::curve::svensson
