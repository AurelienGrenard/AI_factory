// European-option dataset row and host-side JSON loader.
#pragma once

#include <cstdint>
#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// EuropeanOptionParameters is the compact product row transferred to CUDA.
struct EuropeanOptionParameters {
    float strike;
    std::uint32_t maturity;
};

static_assert(std::is_trivially_copyable_v<EuropeanOptionParameters>);

// Load every European-option row into one contiguous vector.
std::vector<EuropeanOptionParameters> load_european_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
