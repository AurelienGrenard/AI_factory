// Model/product row construction and result database writing API.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ai_factory::workbench::registry {

// Select aligned row pairing or a full model/product Cartesian product.
enum class ResultConstruction : unsigned int {
    Aligned,
    CartesianProduct,
};

// Return the number of prices implied by the selected construction.
std::size_t result_row_count(
    std::size_t model_count,
    std::size_t product_count,
    ResultConstruction construction
);

// Write prices, errors, provenance, numerical choices, and timings.
void write_monte_carlo_result_database(
    const std::filesystem::path& model_json_path,
    const std::filesystem::path& product_json_path,
    ResultConstruction construction,
    const std::vector<float>& prices,
    const std::vector<float>& standard_errors,
    const std::string& engine,
    const std::filesystem::path& output_root,
    const std::filesystem::path& generation_script,
    const std::vector<std::filesystem::path>& source_files,
    const std::string& numerical_method,
    std::size_t monte_carlo_paths_per_price,
    float target_dt,
    unsigned int threads_per_block,
    std::uint64_t first_seed,
    double wall_seconds,
    double kernel_seconds
);

}  // namespace ai_factory::workbench::registry
