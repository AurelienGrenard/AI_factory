// G2 parameter-domain predicates shared by host loaders and device preparation.
#pragma once

#include "model/fixed_income/g2/parameters.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::model::fixed_income::g2 {

__host__ __device__ inline bool finite_parameter(float value) {
#if defined(__CUDA_ARCH__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
}

__host__ __device__ inline bool valid_process_parameters(
    const ProcessParameters& process
) {
    return finite_parameter(process.mean_reversion_x)
        && process.mean_reversion_x > 0.0f
        && finite_parameter(process.volatility_x)
        && process.volatility_x >= 0.0f
        && finite_parameter(process.mean_reversion_y)
        && process.mean_reversion_y > 0.0f
        && finite_parameter(process.volatility_y)
        && process.volatility_y >= 0.0f
        && finite_parameter(process.correlation)
        && process.correlation >= -1.0f
        && process.correlation <= 1.0f;
}

__host__ __device__ inline bool valid_parameters(const ModelParameters& model) {
    return valid_process_parameters(model.process)
        && finite_parameter(model.initial_state.state_x)
        && finite_parameter(model.initial_state.state_y);
}

inline const char* process_parameter_domain_error(
    const ProcessParameters& process
) {
    if (!std::isfinite(process.mean_reversion_x)
        || !(process.mean_reversion_x > 0.0f))
        return "mean_reversion_x must be finite and positive.";
    if (!std::isfinite(process.mean_reversion_y)
        || !(process.mean_reversion_y > 0.0f))
        return "mean_reversion_y must be finite and positive.";
    if (!std::isfinite(process.volatility_x)
        || !(process.volatility_x >= 0.0f))
        return "volatility_x must be finite and non-negative.";
    if (!std::isfinite(process.volatility_y)
        || !(process.volatility_y >= 0.0f))
        return "volatility_y must be finite and non-negative.";
    if (!std::isfinite(process.correlation)
        || process.correlation < -1.0f
        || process.correlation > 1.0f)
        return "correlation must be finite and lie in [-1, 1].";
    return nullptr;
}

inline const char* parameter_domain_error(const ModelParameters& model) {
    if (const char* error = process_parameter_domain_error(model.process))
        return error;
    if (!std::isfinite(model.initial_state.state_x)
        || !std::isfinite(model.initial_state.state_y))
        return "initial states must be finite.";
    return nullptr;
}

inline void validate_parameters(
    const ModelParameters& model,
    const std::string& prefix
) {
    if (const char* error = parameter_domain_error(model))
        throw std::invalid_argument(prefix + error);
}

}  // namespace ai_factory::workbench::model::fixed_income::g2
