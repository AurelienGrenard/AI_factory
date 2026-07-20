// Model-independent European-call input type and its registry loader.
// Any dynamics capable of producing a terminal spot can reuse this product
// representation without depending on Heston-specific code.
#pragma once

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::products {

// EuropeanCallInput is the compact FP32 product row transferred to CUDA.
struct EuropeanCallInput {
    float strike;
    float maturity;
};

// Load and convert every row from a European-call registry JSON database.
std::vector<EuropeanCallInput> load_european_calls(
    const std::filesystem::path& json_path
);

}  // namespace ai_factory::workbench::products
