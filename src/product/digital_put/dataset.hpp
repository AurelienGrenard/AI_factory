// Digital-put dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// Cash-or-nothing put parameters stored contiguously for CUDA.
struct DigitalPutParameters {
    float strike;
    float maturity;
    float cash_payoff;
};

static_assert(std::is_trivially_copyable_v<DigitalPutParameters>);

// Load every digital-put row into one contiguous FP32 vector.
std::vector<DigitalPutParameters> load_digital_puts(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
