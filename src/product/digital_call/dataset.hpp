// Digital-call dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// Cash-or-nothing call parameters stored contiguously for CUDA.
struct DigitalCallParameters {
    float strike;
    float maturity;
    float cash_payoff;
};

static_assert(std::is_trivially_copyable_v<DigitalCallParameters>);

// Load every digital-call row into one contiguous FP32 vector.
std::vector<DigitalCallParameters> load_digital_calls(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
