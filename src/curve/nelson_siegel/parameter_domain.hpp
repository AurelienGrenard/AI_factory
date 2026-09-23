// Nelson-Siegel parameter domain shared by host loaders and device preparation.
#pragma once

#include "curve/nelson_siegel/parameters.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::curve::nelson_siegel {

__host__ __device__ inline bool finite_parameter(float value) {
#if defined(__CUDA_ARCH__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
}

__host__ __device__ inline bool valid_parameters(
    const NelsonSiegelParameters& curve
) {
    return finite_parameter(curve.beta0)
        && finite_parameter(curve.beta1)
        && finite_parameter(curve.beta2)
        && finite_parameter(curve.tau)
        && curve.tau > 0.0f;
}

inline void validate_parameters(
    const NelsonSiegelParameters& curve,
    const std::string& prefix
) {
    if (!std::isfinite(curve.beta0))
        throw std::invalid_argument(prefix + "beta0 must be finite.");
    if (!std::isfinite(curve.beta1))
        throw std::invalid_argument(prefix + "beta1 must be finite.");
    if (!std::isfinite(curve.beta2))
        throw std::invalid_argument(prefix + "beta2 must be finite.");
    if (!std::isfinite(curve.tau) || !(curve.tau > 0.0f))
        throw std::invalid_argument(
            prefix + "tau must be finite and positive."
        );
}

}  // namespace ai_factory::workbench::curve::nelson_siegel
