// European-swaption dataset row and host-side JSON loader.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// Regular fixed-leg schedule with exercise equal to the underlying swap start.
struct RegularEuropeanSwaptionParameters {
    float notional;
    float strike;
    float accrual_fraction;
    std::uint32_t exercise_time;
    std::uint32_t payment_interval;
    std::uint32_t payment_count;
};

// Explicit fixed-leg schedule stored in two contiguous dataset pools.
struct ExplicitEuropeanSwaptionParameters {
    float notional;
    float strike;
    std::uint32_t exercise_time;
    std::uint32_t payment_count;
    std::size_t schedule_offset;
};

static_assert(
    std::is_trivially_copyable_v<RegularEuropeanSwaptionParameters>
);
static_assert(
    std::is_trivially_copyable_v<ExplicitEuropeanSwaptionParameters>
);
static_assert(sizeof(RegularEuropeanSwaptionParameters) == 24U);
static_assert(sizeof(ExplicitEuropeanSwaptionParameters) == 24U);

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
