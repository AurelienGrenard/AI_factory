// Black-Scholes dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::black_scholes {

// Compact FP32 model row transferred from host memory to CUDA.
struct BlackScholesModelParameters {
    float spot;
    float risk_free_rate;
    float dividend_yield;
    float volatility;
};

static_assert(std::is_trivially_copyable_v<BlackScholesModelParameters>);

// Load every model row from JSON into one contiguous FP32 vector.
std::vector<BlackScholesModelParameters> load_models(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::black_scholes
