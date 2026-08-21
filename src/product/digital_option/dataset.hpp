// Digital-option dataset row and host-side JSON loader.
#pragma once

#include <cstdint>
#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// Cash-or-nothing option parameters stored contiguously for CUDA.
struct DigitalOptionParameters {
    float strike;
    std::uint32_t maturity;
    float cash_payoff;
};

static_assert(std::is_trivially_copyable_v<DigitalOptionParameters>);

// Load every digital-option row into one contiguous vector.
std::vector<DigitalOptionParameters> load_digital_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
