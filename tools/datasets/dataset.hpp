// Generate parameter rows and serialize complete datasets and catalog YAML.
#pragma once

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ai_factory::workbench::datasets {

// Select aligned pricing rows or a full model/product Cartesian product.
enum class PriceConstruction : unsigned int {
    Aligned,
    CartesianProduct,
};

// One generated model or product parameter object.
using ParameterRow = nlohmann::ordered_json;

// Serialize an arbitrary catalog metadata tree with the repository YAML
// subset. Specialized generators use this after their executable has fixed
// the generation recipe; YAML is documentation, never generator input.
void write_catalog_yaml(
    const std::filesystem::path& path,
    const nlohmann::ordered_json& document
);

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

inline constexpr std::uint32_t kBusinessDaysPerYear = 252U;

// Convert an integer business-day count to the repository year fraction.
constexpr float business_days_to_years(std::uint32_t business_days) {
    return static_cast<float>(business_days)
        / static_cast<float>(kBusinessDaysPerYear);
}

// Describe one human-readable exercise interval measured in business days.
struct ExerciseInterval {
    std::uint32_t business_days;
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

// Concatenate a 90% core regime and a 10% stress regime with explicit YAML
// metadata. The caller remains responsible for choosing the regime sizes.
GeneratedRows core_stress_rows(
    GeneratedRows core,
    GeneratedRows stress
);

// Align equally sized parameter vectors by their row index.
GeneratedRows aligned_grid(const std::vector<GridParameter>& parameters);

// Build the full Cartesian product of the supplied parameter vectors.
GeneratedRows cartesian_grid(const std::vector<GridParameter>& parameters);

// Build K = exp(x) from annualized log-moneyness at each business-day maturity.
GeneratedRows maturity_dependent_exponential_strike_grid(
    const std::vector<std::uint32_t>& maturities,
    std::size_t strikes_per_maturity,
    float log_moneyness_slope
);

// Build traceable core and stress strike grids and concatenate them in that
// order. This is the common 90/10 construction for normalized equity terms.
GeneratedRows core_stress_exponential_strike_grid(
    const std::vector<std::uint32_t>& core_maturities,
    std::size_t core_strikes_per_maturity,
    float core_log_moneyness_slope,
    const std::vector<std::uint32_t>& stress_maturities,
    std::size_t stress_strikes_per_maturity,
    float stress_log_moneyness_slope
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

// Return a strictly increasing inclusive grid of integer business days.
std::vector<std::uint32_t> linear_business_day_grid(
    std::uint32_t minimum,
    std::uint32_t maximum,
    std::size_t count
);

// Return the number of prices implied by the selected construction.
std::size_t price_row_count(
    std::size_t model_count,
    std::size_t product_count,
    PriceConstruction construction
);

// Return the price count for one model, curve, and product construction.
std::size_t price_row_count(
    std::size_t model_count,
    std::size_t curve_count,
    std::size_t product_count,
    PriceConstruction construction
);

// Write a full model dataset and its catalog entry.
void write_model_dataset(
    const std::string& database_id,
    const std::string& model_family,
    const std::filesystem::path& dataset_path,
    const std::filesystem::path& catalog_path,
    const std::string& url,
    const nlohmann::ordered_json& parameter_descriptions,
    const nlohmann::ordered_json& dynamics,
    const GeneratedRows& generated
);

// Write a full curve dataset and its catalog entry.
void write_curve_dataset(
    const std::string& database_id,
    const std::string& curve_family,
    const std::filesystem::path& dataset_path,
    const std::filesystem::path& catalog_path,
    const std::string& url,
    const nlohmann::ordered_json& parameter_descriptions,
    const nlohmann::ordered_json& curve_definition,
    const GeneratedRows& generated
);

// Write a full product dataset and its catalog entry.
void write_product_dataset(
    const std::string& database_id,
    const std::string& product_family,
    const std::filesystem::path& dataset_path,
    const std::filesystem::path& catalog_path,
    const std::string& url,
    const nlohmann::ordered_json& parameter_descriptions,
    const nlohmann::ordered_json& payoff,
    const GeneratedRows& generated
);

// Write a full price dataset and its catalog entry.
void write_monte_carlo_price_dataset(
    const std::filesystem::path& model_dataset_path,
    const std::filesystem::path& product_dataset_path,
    PriceConstruction construction,
    const std::vector<float>& prices,
    const std::vector<float>& standard_errors,
    const std::string& random_generator,
    const std::filesystem::path& dataset_path,
    const std::filesystem::path& catalog_path,
    const std::string& url,
    const std::string& numerical_method,
    std::size_t monte_carlo_paths_per_price,
    const std::string& delta_t,
    const nlohmann::ordered_json& cuda_execution,
    const nlohmann::ordered_json& catalog_sections,
    std::uint64_t first_seed,
    double wall_seconds,
    double kernel_seconds
);

// Write closed-form prices built from model, curve, and product datasets.
void write_analytical_price_dataset(
    const std::filesystem::path& model_dataset_path,
    const std::filesystem::path& product_dataset_path,
    PriceConstruction construction,
    const std::vector<float>& prices,
    const std::filesystem::path& dataset_path,
    const std::filesystem::path& catalog_path,
    const std::string& url,
    const std::string& numerical_method,
    const nlohmann::ordered_json& cuda_execution,
    double wall_seconds,
    double kernel_seconds
);

// Write closed-form prices built from model, curve, and product datasets.
void write_analytical_price_dataset(
    const std::filesystem::path& model_dataset_path,
    const std::filesystem::path& curve_dataset_path,
    const std::filesystem::path& product_dataset_path,
    PriceConstruction construction,
    const std::vector<float>& prices,
    const std::filesystem::path& dataset_path,
    const std::filesystem::path& catalog_path,
    const std::string& url,
    const std::string& numerical_method,
    const nlohmann::ordered_json& cuda_execution,
    double wall_seconds,
    double kernel_seconds
);

}  // namespace ai_factory::workbench::datasets
