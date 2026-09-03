// Pure parameter sampling and grid construction.
#include "tools/datasets/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <random>
#include <stdexcept>

namespace ai_factory::workbench::datasets {
namespace {
double readable_grid_bound(float value) {
    constexpr double scale = 10'000'000.0;
    return std::round(static_cast<double>(value) * scale) / scale;
}

void validate_names(const std::vector<std::string>& names) {
    for (std::size_t index = 0; index < names.size(); ++index) {
        if (names[index].empty()) {
            throw std::invalid_argument("Parameter names cannot be empty.");
        }
        for (std::size_t previous = 0; previous < index; ++previous) {
            if (names[index] == names[previous]) {
                throw std::invalid_argument(
                    "Duplicate parameter name: " + names[index]
                );
            }
        }
    }
}

}  // namespace

GeneratedRows uniform_rows(
    std::size_t row_count,
    std::uint64_t seed,
    const std::vector<UniformParameter>& parameters
) {
    if (row_count == 0U || parameters.empty()) {
        throw std::invalid_argument(
            "Uniform generation requires rows and parameters."
        );
    }
    std::vector<std::string> names;
    nlohmann::ordered_json bounds;
    for (const UniformParameter& parameter : parameters) {
        if (!(parameter.minimum <= parameter.maximum)) {
            throw std::invalid_argument(
                "Invalid uniform bounds for " + parameter.name
            );
        }
        names.push_back(parameter.name);
        bounds[parameter.name] = {
            readable_grid_bound(parameter.minimum),
            readable_grid_bound(parameter.maximum),
        };
    }
    validate_names(names);

    std::mt19937_64 generator(seed);
    std::vector<ParameterRow> rows(row_count);
    for (const UniformParameter& parameter : parameters) {
        std::uniform_real_distribution<float> distribution(
            parameter.minimum, parameter.maximum
        );
        for (ParameterRow& row : rows) {
            row[parameter.name] = distribution(generator);
        }
    }
    return {
        std::move(rows),
        {
            {"method", "independent uniform sample"},
            {"bounds", bounds},
        },
    };
}

GeneratedRows core_stress_rows(
    GeneratedRows core,
    GeneratedRows stress
) {
    if (core.rows.empty() || stress.rows.empty()) {
        throw std::invalid_argument(
            "Core/stress generation requires two non-empty regimes."
        );
    }
    const std::size_t core_count = core.rows.size();
    const std::size_t stress_count = stress.rows.size();
    const std::size_t total_count = core_count + stress_count;
    if (core_count * 10U != total_count * 9U
        || stress_count * 10U != total_count) {
        throw std::invalid_argument(
            "Core/stress generation requires an exact 90/10 split."
        );
    }

    nlohmann::ordered_json core_generation = std::move(core.construction);
    nlohmann::ordered_json stress_generation = std::move(stress.construction);
    core.rows.reserve(total_count);
    core.rows.insert(
        core.rows.end(),
        std::make_move_iterator(stress.rows.begin()),
        std::make_move_iterator(stress.rows.end())
    );
    core.construction = {
        {"method", "ordered 90/10 core-stress construction"},
        {"row_order", {
            {"core", "rows 1-" + std::to_string(core_count)},
            {
                "stress",
                "rows " + std::to_string(core_count + 1U)
                    + "-" + std::to_string(total_count)
            },
        }},
        {"core_share", 0.9},
        {"stress_share", 0.1},
        {"core", {
            {"row_count", core_count},
            {"generation", std::move(core_generation)},
        }},
        {"stress", {
            {"row_count", stress_count},
            {"generation", std::move(stress_generation)},
        }},
    };
    return core;
}

GeneratedRows aligned_grid(const std::vector<GridParameter>& parameters) {
    if (parameters.empty() || parameters.front().values.empty()) {
        throw std::invalid_argument("An aligned grid cannot be empty.");
    }
    const std::size_t row_count = parameters.front().values.size();
    std::vector<std::string> names;
    nlohmann::ordered_json parameter_counts;
    nlohmann::ordered_json values;
    for (const GridParameter& parameter : parameters) {
        if (parameter.values.size() != row_count) {
            throw std::invalid_argument(
                "Aligned grids require equal parameter lengths."
            );
        }
        names.push_back(parameter.name);
        parameter_counts[parameter.name] = parameter.values.size();
        values[parameter.name] = parameter.values;
    }
    validate_names(names);

    std::vector<ParameterRow> rows(row_count);
    for (std::size_t row = 0; row < row_count; ++row) {
        for (const GridParameter& parameter : parameters) {
            rows[row][parameter.name] = parameter.values[row];
        }
    }
    return {
        std::move(rows),
        {
            {"method", "aligned grid"},
            {"rule", "values with the same index form one row"},
            {"parameter_counts", parameter_counts},
            {"values", values},
        },
    };
}

GeneratedRows cartesian_grid(const std::vector<GridParameter>& parameters) {
    if (parameters.empty()) {
        throw std::invalid_argument("A Cartesian grid cannot be empty.");
    }
    std::vector<std::string> names;
    nlohmann::ordered_json grid;
    std::size_t row_count = 1U;
    for (const GridParameter& parameter : parameters) {
        if (parameter.values.empty()) {
            throw std::invalid_argument("Cartesian grid values cannot be empty.");
        }
        if (row_count > std::numeric_limits<std::size_t>::max()
                / parameter.values.size()) {
            throw std::overflow_error("Cartesian grid row count exceeds size_t.");
        }
        row_count *= parameter.values.size();
        names.push_back(parameter.name);
        const auto [minimum, maximum] = std::minmax_element(
            parameter.values.begin(), parameter.values.end()
        );
        grid[parameter.name] = {
            {"minimum", readable_grid_bound(*minimum)},
            {"maximum", readable_grid_bound(*maximum)},
            {"count", parameter.values.size()},
            {"spacing", parameter.spacing},
        };
    }
    validate_names(names);

    std::vector<ParameterRow> rows;
    rows.reserve(row_count);
    // Decode each row index as a mixed-radix Cartesian coordinate.
    for (std::size_t row_index = 0; row_index < row_count; ++row_index) {
        std::size_t remaining = row_index;
        ParameterRow row;
        for (std::size_t offset = parameters.size(); offset-- > 0U;) {
            const GridParameter& parameter = parameters[offset];
            const std::size_t value_index = remaining % parameter.values.size();
            remaining /= parameter.values.size();
            row[parameter.name] = parameter.values[value_index];
        }
        ParameterRow ordered_row;
        for (const GridParameter& parameter : parameters) {
            ordered_row[parameter.name] = row.at(parameter.name);
        }
        rows.push_back(std::move(ordered_row));
    }
    return {
        std::move(rows),
        {
            {"method", "Cartesian grid"},
            {"grid", grid},
        },
    };
}

GeneratedRows maturity_dependent_exponential_strike_grid(
    const std::vector<std::uint32_t>& maturities,
    std::size_t strikes_per_maturity,
    float log_moneyness_slope
) {
    if (maturities.empty() || strikes_per_maturity == 0U) {
        throw std::invalid_argument(
            "An exponential strike grid requires maturities and strikes."
        );
    }
    if (!(log_moneyness_slope >= 0.0f)
        || !std::isfinite(log_moneyness_slope)) {
        throw std::invalid_argument(
            "The log-moneyness slope must be finite and non-negative."
        );
    }
    for (std::uint32_t maturity : maturities) {
        if (maturity == 0U) {
            throw std::invalid_argument(
                "Exponential strike-grid maturities must be positive."
            );
        }
    }
    if (maturities.size() > std::numeric_limits<std::size_t>::max()
            / strikes_per_maturity) {
        throw std::overflow_error(
            "Exponential strike-grid row count exceeds size_t."
        );
    }

    const std::size_t row_count =
        maturities.size() * strikes_per_maturity;
    std::vector<ParameterRow> rows;
    rows.reserve(row_count);
    // Each maturity receives its own symmetric log-moneyness interval.
    for (std::uint32_t maturity : maturities) {
        const float maturity_years = business_days_to_years(maturity);
        const float radius = log_moneyness_slope * maturity_years;
        const std::vector<float> log_strikes =
            linear_grid(-radius, radius, strikes_per_maturity);
        for (float log_strike : log_strikes) {
            rows.push_back({
                {"strike", std::exp(log_strike)},
                {"maturity", maturity},
            });
        }
    }

    const auto [minimum_maturity, maximum_maturity] = std::minmax_element(
        maturities.begin(), maturities.end()
    );
    return {
        std::move(rows),
        {
            {"method", "maturity-dependent exponential grid"},
            {
                "rule",
                "For each T in business days, x is linearly spaced on "
                "[-aT/252, aT/252] and K = exp(x)."
            },
            {"grid", {
                {"maturity", {
                    {"minimum", *minimum_maturity},
                    {"maximum", *maximum_maturity},
                    {"count", maturities.size()},
                    {"spacing", "linear"},
                }},
                {"strike", {
                    {"count_per_maturity", strikes_per_maturity},
                    {"spacing", "linear in log-strike"},
                    {"conditional_bounds", "[exp(-aT/252), exp(aT/252)]"},
                    {"a", readable_grid_bound(log_moneyness_slope)},
                }},
            }},
        },
    };
}

GeneratedRows core_stress_exponential_strike_grid(
    const std::vector<std::uint32_t>& core_maturities,
    std::size_t core_strikes_per_maturity,
    float core_log_moneyness_slope,
    const std::vector<std::uint32_t>& stress_maturities,
    std::size_t stress_strikes_per_maturity,
    float stress_log_moneyness_slope
) {
    GeneratedRows core = maturity_dependent_exponential_strike_grid(
        core_maturities,
        core_strikes_per_maturity,
        core_log_moneyness_slope
    );
    GeneratedRows stress = maturity_dependent_exponential_strike_grid(
        stress_maturities,
        stress_strikes_per_maturity,
        stress_log_moneyness_slope
    );
    GeneratedRows combined = core_stress_rows(
        std::move(core), std::move(stress)
    );
    const auto [core_minimum, core_maximum] = std::minmax_element(
        core_maturities.begin(), core_maturities.end()
    );
    const auto [stress_minimum, stress_maximum] = std::minmax_element(
        stress_maturities.begin(), stress_maturities.end()
    );
    combined.construction["grid"] = {
        {"maturity", {
            {"core", {
                {"minimum", *core_minimum},
                {"maximum", *core_maximum},
                {"count", core_maturities.size()},
                {"spacing", "linear"},
            }},
            {"stress", {
                {"minimum", *stress_minimum},
                {"maximum", *stress_maximum},
                {"count", stress_maturities.size()},
                {"spacing", "linear"},
            }},
        }},
        {"strike", {
            {"core", {
                {"count_per_maturity", core_strikes_per_maturity},
                {"conditional_bounds", "[exp(-aT/252), exp(aT/252)]"},
                {"a", readable_grid_bound(core_log_moneyness_slope)},
            }},
            {"stress", {
                {"count_per_maturity", stress_strikes_per_maturity},
                {"conditional_bounds", "[exp(-aT/252), exp(aT/252)]"},
                {"a", readable_grid_bound(stress_log_moneyness_slope)},
            }},
            {"spacing", "linear in log-strike"},
        }},
    };
    return combined;
}

void assign_uniform_exercise_intervals(
    GeneratedRows& generated,
    const std::vector<ExerciseInterval>& intervals,
    std::size_t minimum_exercise_count,
    std::uint64_t seed
) {
    if (generated.rows.empty() || intervals.empty()
        || minimum_exercise_count < 2U) {
        throw std::invalid_argument(
            "Exercise-interval sampling requires rows, choices, and dates."
        );
    }
    for (const ExerciseInterval& interval : intervals) {
        if (interval.business_days == 0U || interval.label.empty()) {
            throw std::invalid_argument(
                "Exercise intervals require positive values and labels."
            );
        }
    }

    std::mt19937_64 generator(seed);
    const std::size_t required_pre_maturity_dates =
        minimum_exercise_count - 1U;
    for (ParameterRow& row : generated.rows) {
        const std::uint32_t maturity =
            row.at("maturity").get<std::uint32_t>();
        std::vector<std::uint32_t> feasible_intervals;
        for (const ExerciseInterval& interval : intervals) {
            if (required_pre_maturity_dates * interval.business_days
                    < maturity) {
                feasible_intervals.push_back(interval.business_days);
            }
        }
        if (feasible_intervals.empty()) {
            throw std::invalid_argument(
                "No exercise interval satisfies the maturity constraint."
            );
        }
        std::uniform_int_distribution<std::size_t> selection(
            0U, feasible_intervals.size() - 1U
        );
        row["exercise_interval"] = feasible_intervals[selection(generator)];
    }

    nlohmann::ordered_json conventions = nlohmann::ordered_json::array();
    for (const ExerciseInterval& interval : intervals) {
        conventions.push_back(interval.label);
    }
    generated.construction["exercise_interval_sampling"] = {
        {"method", "uniform sample among feasible conventions"},
        {"conventions", conventions},
        {
            "exercise_dates",
            "maturity - n * exercise_interval, ..., maturity"
        },
        {
            "constraint",
            std::to_string(required_pre_maturity_dates)
                + " * exercise_interval < maturity"
        },
        {
            "minimum_exercise_dates",
            minimum_exercise_count
        },
    };
}

std::vector<float> linear_grid(float minimum, float maximum, std::size_t count) {
    if (count == 0U || !(minimum <= maximum)) {
        throw std::invalid_argument("A linear grid requires valid bounds and count.");
    }
    if (count == 1U) return {minimum};
    std::vector<float> values(count);
    const float denominator = static_cast<float>(count - 1U);
    for (std::size_t index = 0; index < count; ++index) {
        const float weight = static_cast<float>(index) / denominator;
        values[index] = minimum + weight * (maximum - minimum);
    }
    return values;
}

std::vector<std::uint32_t> linear_business_day_grid(
    std::uint32_t minimum,
    std::uint32_t maximum,
    std::size_t count
) {
    if (count == 0U || minimum == 0U || minimum > maximum) {
        throw std::invalid_argument(
            "A business-day grid requires positive valid bounds and count."
        );
    }
    if (count == 1U) return {minimum};
    if (static_cast<std::uint64_t>(maximum) - minimum + 1U < count) {
        throw std::invalid_argument(
            "A business-day grid cannot contain duplicate days."
        );
    }
    std::vector<std::uint32_t> values(count);
    const double denominator = static_cast<double>(count - 1U);
    for (std::size_t index = 0U; index < count; ++index) {
        const double weight = static_cast<double>(index) / denominator;
        values[index] = static_cast<std::uint32_t>(std::llround(
            static_cast<double>(minimum)
                + weight * static_cast<double>(maximum - minimum)
        ));
        if (index > 0U && values[index] <= values[index - 1U]) {
            throw std::runtime_error(
                "A rounded business-day grid is not strictly increasing."
            );
        }
    }
    return values;
}

}  // namespace ai_factory::workbench::datasets
