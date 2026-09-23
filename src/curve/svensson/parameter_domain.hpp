// Svensson parameter domain shared by host loaders and device preparation.
#pragma once

#include "curve/svensson/parameters.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::curve::svensson {

__host__ __device__ inline bool finite_parameter(float value) {
#if defined(__CUDA_ARCH__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
}

__host__ __device__ inline bool valid_parameters(const SvenssonParameters& curve) {
    return finite_parameter(curve.beta0)
        && finite_parameter(curve.beta1)
        && finite_parameter(curve.beta2)
        && finite_parameter(curve.beta3)
        && finite_parameter(curve.tau1)
        && curve.tau1 > 0.0f
        && finite_parameter(curve.tau2)
        && curve.tau2 > curve.tau1;
}

inline void validate_parameters(
    const SvenssonParameters& curve,
    const std::string& prefix
) {
    if (!std::isfinite(curve.beta0))
        throw std::invalid_argument(prefix + "beta0 must be finite.");
    if (!std::isfinite(curve.beta1))
        throw std::invalid_argument(prefix + "beta1 must be finite.");
    if (!std::isfinite(curve.beta2))
        throw std::invalid_argument(prefix + "beta2 must be finite.");
    if (!std::isfinite(curve.beta3))
        throw std::invalid_argument(prefix + "beta3 must be finite.");
    if (!std::isfinite(curve.tau1) || !(curve.tau1 > 0.0f))
        throw std::invalid_argument(
            prefix + "tau1 must be finite and positive."
        );
    if (!std::isfinite(curve.tau2) || !(curve.tau2 > 0.0f))
        throw std::invalid_argument(
            prefix + "tau2 must be finite and positive."
        );
    if (!(curve.tau2 > curve.tau1))
        throw std::invalid_argument(prefix + "tau2 must exceed tau1.");
}

}  // namespace ai_factory::workbench::curve::svensson
