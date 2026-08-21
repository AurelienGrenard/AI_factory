// Asian-option dataset row and host-side JSON loader.
#pragma once

#include <cstdint>
#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// AsianOptionParameters is the compact product row transferred to CUDA.
struct AsianOptionParameters {
    float strike;
    std::uint32_t maturity;
};

static_assert(std::is_trivially_copyable_v<AsianOptionParameters>);

// Load every Asian-option row into one contiguous vector.
std::vector<AsianOptionParameters> load_asian_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
