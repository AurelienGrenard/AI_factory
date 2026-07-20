// Declaration of the canonical Monte Carlo result JSON/YAML writer.
// Serialization lives in result_output.cpp so generators include a small API.
#pragma once

#include "tools/registry/common.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ai_factory::workbench::result_output {

// Write prices, standard errors, provenance, numerical choices, and timings.
void write_monte_carlo_result(
    const std::filesystem::path& model_json_path,
    const std::filesystem::path& product_json_path,
    ConstructionMethod construction,
    const std::vector<float>& prices,
    const std::vector<float>& standard_errors,
    const std::string& engine,
    const std::string& result_version,
    const std::filesystem::path& output_root,
    const std::filesystem::path& generation_script,
    const std::filesystem::path& source_file,
    const std::string& numerical_method,
    std::size_t monte_carlo_paths_per_price,
    float target_dt,
    unsigned int threads_per_block,
    std::uint64_t first_seed,
    double wall_seconds,
    double kernel_seconds
);

}  // namespace ai_factory::workbench::result_output
