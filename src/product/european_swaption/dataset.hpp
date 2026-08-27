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
    std::uint32_t maximum_payment_count = 0U;
};

// Own explicit rows and payment-major ELLPACK pools. For row r and payment p,
// each value lives at p * products.size() + r; unused tail cells are padding.
struct ExplicitEuropeanSwaptionDataset {
    std::vector<ExplicitEuropeanSwaptionParameters> products;
    std::vector<std::uint32_t> payment_times_days;
    std::vector<float> accrual_fractions;
    std::uint32_t maximum_payment_count = 0U;
};

// Load one regular-schedule European-swaption dataset.
RegularEuropeanSwaptionDataset load_european_swaptions(
    const std::filesystem::path& dataset_path
);

// Load arbitrary schedules into coalesced payment-major pools.
ExplicitEuropeanSwaptionDataset load_explicit_european_swaptions(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
