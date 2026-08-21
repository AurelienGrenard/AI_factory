// European-swaption dataset row and host-side JSON loader.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// Keep one complete fixed-leg schedule in the trivially copyable CUDA row.
inline constexpr std::size_t kMaximumEuropeanSwaptionPayments = 64U;

// Physical-settlement swaption with exercise equal to the underlying swap start.
struct EuropeanSwaptionParameters {
    float notional;
    float strike;
    std::uint32_t exercise_time;
    std::uint32_t payment_count;
    std::uint32_t payment_times[kMaximumEuropeanSwaptionPayments];
    std::uint32_t accrual_periods[kMaximumEuropeanSwaptionPayments];
};

static_assert(std::is_trivially_copyable_v<EuropeanSwaptionParameters>);

// Load every European-swaption row into one contiguous CUDA-ready vector.
std::vector<EuropeanSwaptionParameters> load_european_swaptions(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
