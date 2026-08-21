// Cliquet dataset row and host-side JSON loader.
#pragma once

#include <cstdint>
#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// Compact periodic-return terms transferred to the CUDA pricer.
struct CliquetParameters {
    std::uint32_t maturity;
    std::uint32_t observation_interval;
    float participation_rate;
    float local_floor;
    float local_cap;
    float global_floor;
    float global_cap;
};

static_assert(std::is_trivially_copyable_v<CliquetParameters>);

// Load every Cliquet row into one contiguous vector.
std::vector<CliquetParameters> load_cliquets(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
