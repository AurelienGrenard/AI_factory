// European-swaption dataset row and host-side JSON loader.
#pragma once

#include "product/european_swaption/parameters.hpp"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace ai_factory::workbench::product {

// Own regular product rows; no separate schedule allocation is required.
struct RegularEuropeanSwaptionDataset {
    std::vector<RegularEuropeanSwaptionParameters> products;
};

// Own explicit product rows and their two parallel fixed-leg schedule pools.
struct ExplicitEuropeanSwaptionDataset {
    std::vector<ExplicitEuropeanSwaptionParameters> products;
    std::vector<std::uint32_t> payment_times;
    std::vector<float> accrual_fractions;
};

// Load one regular-schedule European-swaption dataset.
RegularEuropeanSwaptionDataset load_european_swaptions(
    const std::filesystem::path& dataset_path
);

// Load arbitrary schedules and flatten them into contiguous pools.
ExplicitEuropeanSwaptionDataset load_explicit_european_swaptions(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
