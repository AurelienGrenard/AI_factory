// Parameter generation and model/product database writing API.
#pragma once

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ai_factory::workbench::registry {

// One generated model or product parameter object.
using ParameterRow = nlohmann::ordered_json;

// Describe one independently sampled floating-point field.
struct UniformParameter {
    std::string name;
    float minimum;
    float maximum;
};

// Associate one parameter field with grid values and their spacing rule.
struct GridParameter {
    std::string name;
    std::vector<float> values;
    std::string spacing = "explicit values";
};

// Keep generated rows together with their construction metadata.
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

// Align equally sized parameter vectors by their row index.
GeneratedRows aligned_grid(const std::vector<GridParameter>& parameters);

// Build the full Cartesian product of the supplied parameter vectors.
GeneratedRows cartesian_grid(const std::vector<GridParameter>& parameters);

// Return an inclusive linearly spaced FP32 grid.
std::vector<float> linear_grid(float minimum, float maximum, std::size_t count);

// Write one model JSON/YAML pair using the workbench registry layout.
void write_model_database(
    const std::string& database_id,
    const std::string& model_family,
    const std::filesystem::path& json_path,
    const std::filesystem::path& generation_script,
    const nlohmann::ordered_json& parameter_descriptions,
    const nlohmann::ordered_json& dynamics,
    const GeneratedRows& generated
);

// Write one product JSON/YAML pair using the workbench registry layout.
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
