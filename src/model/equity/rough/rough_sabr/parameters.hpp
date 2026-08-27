// Rough-SABR parameters shared by host loaders and CUDA path policies.
#pragma once

#include <type_traits>

namespace ai_factory::workbench::model::equity::rough_sabr {

// The lognormal Volterra volatility uses the same eta convention as rough
// Bergomi. beta controls the CEV elasticity of the equity diffusion.
struct ModelParameters {
    float spot;
    float risk_free_rate;
    float dividend_yield;
    float xi_0;
    float eta;
    float hurst_exponent;
    float rho;
    float beta;
};

static_assert(std::is_trivially_copyable_v<ModelParameters>);

}  // namespace ai_factory::workbench::model::equity::rough_sabr
