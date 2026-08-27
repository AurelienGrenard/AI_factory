// Pure, reproducible parameter sampling and grid construction.
#pragma once

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ai_factory::workbench::datasets {

using ParameterRow = nlohmann::ordered_json;

struct UniformParameter {
    std::string name;
    float minimum;
    float maximum;
};

struct GridParameter {
    std::string name;
    std::vector<float> values;
    std::string spacing = "explicit values";
};

inline constexpr std::uint32_t kBusinessDaysPerYear = 252U;

constexpr float business_days_to_years(std::uint32_t business_days) {
    return static_cast<float>(business_days)
        / static_cast<float>(kBusinessDaysPerYear);
}

struct ExerciseInterval {
    std::uint32_t business_days;
    std::string label;
};

struct GeneratedRows {
    std::vector<ParameterRow> rows;
    nlohmann::ordered_json construction;
};

GeneratedRows uniform_rows(
    std::size_t row_count,
    std::uint64_t seed,
    const std::vector<UniformParameter>& parameters
);

GeneratedRows core_stress_rows(GeneratedRows core, GeneratedRows stress);
GeneratedRows aligned_grid(const std::vector<GridParameter>& parameters);
GeneratedRows cartesian_grid(const std::vector<GridParameter>& parameters);

GeneratedRows maturity_dependent_exponential_strike_grid(
    const std::vector<std::uint32_t>& maturities,
    std::size_t strikes_per_maturity,
    float log_moneyness_slope
);

GeneratedRows core_stress_exponential_strike_grid(
    const std::vector<std::uint32_t>& core_maturities,
    std::size_t core_strikes_per_maturity,
    float core_log_moneyness_slope,
    const std::vector<std::uint32_t>& stress_maturities,
    std::size_t stress_strikes_per_maturity,
    float stress_log_moneyness_slope
);

void assign_uniform_exercise_intervals(
    GeneratedRows& generated,
    const std::vector<ExerciseInterval>& intervals,
    std::size_t minimum_exercise_count,
    std::uint64_t seed
);

std::vector<float> linear_grid(float minimum, float maximum, std::size_t count);

std::vector<std::uint32_t> linear_business_day_grid(
    std::uint32_t minimum,
    std::uint32_t maximum,
    std::size_t count
);

}  // namespace ai_factory::workbench::datasets
