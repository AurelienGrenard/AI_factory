// Minimal parameter-database API for model and product registry generators.
// It creates uniform samples or grids, then writes canonical JSON and YAML.
#pragma once

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ai_factory::workbench::registry {

using ParameterRow = nlohmann::ordered_json;

// UniformParameter describes one independently sampled floating-point field.
struct UniformParameter {
    std::string name;
    float minimum;
    float maximum;
};

// GridParameter associates one field with its values and their spacing rule.
struct GridParameter {
    std::string name;
    std::vector<float> values;
    std::string spacing = "explicit values";
};

// GeneratedRows keeps parameter rows together with their YAML construction note.
struct GeneratedRows {
    std::vector<ParameterRow> rows;
    nlohmann::ordered_json construction;
};

// Draw independent uniform parameters with a reproducible standard C++ RNG.
GeneratedRows uniform_rows(
    std::size_t row_count,
    std::uint64_t seed,
    const std::vector<UniformParameter>& parameters
);

// Align grid values by index; every parameter must contain the same row count.
GeneratedRows aligned_grid(const std::vector<GridParameter>& parameters);

// Build the full Cartesian product of the supplied parameter value vectors.
GeneratedRows cartesian_grid(const std::vector<GridParameter>& parameters);

// Return an inclusive linear grid with a fixed number of points.
std::vector<float> linear_grid(float minimum, float maximum, std::size_t count);

// Write one model JSON/YAML pair using the standard workbench registry layout.
void write_model_database(
    const std::string& database_id,
    const std::string& model_family,
    const std::filesystem::path& json_path,
    const std::filesystem::path& generation_script,
    const nlohmann::ordered_json& parameter_descriptions,
    const nlohmann::ordered_json& dynamics,
    const GeneratedRows& generated
);

// Write one product JSON/YAML pair using the standard workbench registry layout.
void write_product_database(
    const std::string& database_id,
    const std::string& product_family,
    const std::filesystem::path& json_path,
    const std::filesystem::path& generation_script,
    const nlohmann::ordered_json& parameter_descriptions,
    const nlohmann::ordered_json& payoff,
    const GeneratedRows& generated
);

}  // namespace ai_factory::workbench::registry
