// Black-Scholes model parameters shared by host loaders and CUDA dynamics.
#pragma once

#include <type_traits>

namespace ai_factory::workbench::black_scholes {

// Compact FP32 model row transferred from host memory to CUDA.
struct ModelParameters {
    float spot;
    float risk_free_rate;
    float dividend_yield;
    float volatility;
};

static_assert(std::is_trivially_copyable_v<ModelParameters>);

}  // namespace ai_factory::workbench::black_scholes
