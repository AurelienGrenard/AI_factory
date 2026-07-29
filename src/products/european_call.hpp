// Model-independent European-call parameters and registry loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::products {

// EuropeanCallInput is the compact FP32 product row transferred to CUDA.
struct EuropeanCallInput {
    float strike;
    float maturity;
};

static_assert(std::is_trivially_copyable_v<EuropeanCallInput>);

// Load and convert every row from a European-call registry JSON database.
std::vector<EuropeanCallInput> load_european_calls(
    const std::filesystem::path& json_path
);

}  // namespace ai_factory::workbench::products
