// Variance-Gamma parameters shared by host loaders and CUDA dynamics.
#pragma once

#include <type_traits>

namespace ai_factory::workbench::variance_gamma {

// Risk-neutral exponential Variance-Gamma inputs in the Madan-Carr-Chang form.
struct ModelParameters {
    float spot;
    float risk_free_rate;
    float dividend_yield;
    float sigma;
    float nu;
    float theta;
};

static_assert(std::is_trivially_copyable_v<ModelParameters>);

}  // namespace ai_factory::workbench::variance_gamma
