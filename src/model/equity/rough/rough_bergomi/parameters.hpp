// Rough-Bergomi parameters shared by host loaders and CUDA dynamics.
#pragma once

#include <type_traits>

namespace ai_factory::workbench::model::equity::rough_bergomi {

// Flat-forward-variance rough-Bergomi inputs under the risk-neutral measure.
struct ModelParameters {
    float spot;
    float risk_free_rate;
    float dividend_yield;
    float xi_0;
    float eta;
    float hurst_exponent;
    float rho;
};

static_assert(std::is_trivially_copyable_v<ModelParameters>);

}  // namespace ai_factory::workbench::model::equity::rough_bergomi
