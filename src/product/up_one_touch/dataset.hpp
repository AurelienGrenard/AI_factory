// Up-one-touch dataset row and host-side JSON loader.
#pragma once

#include <cstdint>
#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// UpOneTouchParameters is the compact product row transferred to CUDA.
struct UpOneTouchParameters {
    float barrier;
    float cash_payoff;
    std::uint32_t maturity;
};

static_assert(std::is_trivially_copyable_v<UpOneTouchParameters>);

// Load every Up-one-touch row into one contiguous vector.
std::vector<UpOneTouchParameters> load_up_one_touches(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
