// G2++ parameter domain shared by host loaders and device preparation.
#pragma once

#include "model/fixed_income/g2/parameter_domain.hpp"
#include "model/fixed_income/g2_plus_plus/parameters.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::model::fixed_income::g2_plus_plus {

__host__ __device__ inline bool valid_parameters(const ModelParameters& model) {
    return ::ai_factory::workbench::model::fixed_income::g2::valid_process_parameters(
        model.process
    );
}

inline void validate_parameters(
    const ModelParameters& model,
    const std::string& prefix
) {
    const auto& process = model.process;
    if (!std::isfinite(process.mean_reversion_x)
        || process.mean_reversion_x <= 0.0f
        || !std::isfinite(process.mean_reversion_y)
        || process.mean_reversion_y <= 0.0f) {
        throw std::invalid_argument(
            prefix + "mean reversions must be finite and positive."
        );
    }
    if (!std::isfinite(process.volatility_x)
        || process.volatility_x < 0.0f
        || !std::isfinite(process.volatility_y)
        || process.volatility_y < 0.0f) {
        throw std::invalid_argument(
            prefix + "volatilities must be finite and non-negative."
        );
    }
    if (!std::isfinite(process.correlation)
        || process.correlation < -1.0f
        || process.correlation > 1.0f) {
        throw std::invalid_argument(
            prefix + "correlation must be finite and lie in [-1, 1]."
        );
    }
}

}  // namespace ai_factory::workbench::model::fixed_income::g2_plus_plus
