// Complete parameter generation and dataset serialization API.
#pragma once

#include <nlohmann/json.hpp>

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

// Describe one human-readable exercise interval measured in years.
struct ExerciseInterval {
    float years;
    std::string label;
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

// Build K = exp(x), with x linearly spaced on [-aT, aT] for every maturity T.
GeneratedRows maturity_dependent_exponential_strike_grid(
    const std::vector<float>& maturities,
    std::size_t strikes_per_maturity,
    float log_moneyness_slope
);

// Assign a feasible interval with a minimum number of dates including maturity.
void assign_uniform_exercise_intervals(
    GeneratedRows& generated,
    const std::vector<ExerciseInterval>& intervals,
    std::size_t minimum_exercise_count,
    std::uint64_t seed
);

// Return an inclusive linearly spaced FP32 grid.
std::vector<float> linear_grid(float minimum, float maximum, std::size_t count);

// Return the number of prices implied by the selected construction.
std::size_t result_row_count(
    std::size_t model_count,
    std::size_t product_count,
    ResultConstruction construction
);

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

// Write prices, errors, provenance, numerical choices, and timings.
void write_monte_carlo_result_database(
    const std::filesystem::path& model_json_path,
    const std::filesystem::path& product_json_path,
    ResultConstruction construction,
    const std::vector<float>& prices,
    const std::vector<float>& standard_errors,
    const std::string& random_generator,
    const std::filesystem::path& output_root,
    const std::filesystem::path& generation_script,
    const std::vector<std::filesystem::path>& source_files,
    const std::string& numerical_method,
    std::size_t monte_carlo_paths_per_price,
    const std::string& target_dt,
    const nlohmann::ordered_json& cuda_execution,
    const nlohmann::ordered_json& specification_sections,
    std::uint64_t first_seed,
    double wall_seconds,
    double kernel_seconds
);

}  // namespace ai_factory::workbench::registry
