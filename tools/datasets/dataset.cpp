// Parameter generation and complete dataset and catalog serialization.
#include "tools/datasets/dataset.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <ostream>
#include <random>
#include <sstream>
#include <stdexcept>

namespace ai_factory::workbench::datasets {
namespace {

// Create parent directories and write one complete text artifact.
void write_text_file(
    const std::filesystem::path& path,
    const std::string& contents
) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Cannot open output file: " + path.string());
    }
    output << contents;
    if (!output) {
        throw std::runtime_error("Cannot write output file: " + path.string());
    }
}

// Recursively serialize the JSON-shaped metadata tree as simple YAML.
void write_yaml_value(
    std::ostream& output,
    const nlohmann::ordered_json& value,
    std::size_t indentation
) {
    const std::string spaces(indentation, ' ');
    if (value.is_object()) {
        for (const auto& [key, child] : value.items()) {
            output << spaces << key << ':';
            if (child.is_object() || child.is_array()) {
                output << '\n';
                write_yaml_value(output, child, indentation + 2U);
            } else {
                output << ' ' << child.dump() << '\n';
            }
        }
        return;
    }
    if (value.is_array()) {
        for (const auto& child : value) {
            output << spaces << '-';
            if (child.is_object() || child.is_array()) {
                output << '\n';
                write_yaml_value(output, child, indentation + 2U);
            } else {
                output << ' ' << child.dump() << '\n';
            }
        }
    }
}

// Convert a zero-based row index to a one-based padded identifier.
std::string format_row_id(std::size_t index) {
    std::ostringstream stream;
    stream << std::setw(6) << std::setfill('0') << index + 1U;
    return stream.str();
}

// Preserve useful precision and add readable minute or hour components.
std::string format_duration(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0.0) {
        throw std::invalid_argument(
            "A formatted duration must be finite and non-negative."
        );
    }
    const auto decimal = [](double value) {
        std::ostringstream stream;
        stream << std::fixed << std::setprecision(9) << value;
        std::string text = stream.str();
        text.erase(text.find_last_not_of('0') + 1U);
        if (!text.empty() && text.back() == '.') text.pop_back();
        return text;
    };

    std::string result = decimal(seconds) + " s";
    if (seconds < 60.0) return result;

    const auto rounded_seconds =
        static_cast<std::uint64_t>(std::llround(seconds));
    if (rounded_seconds < 3'600U) {
        const std::uint64_t minutes = rounded_seconds / 60U;
        const std::uint64_t remaining_seconds = rounded_seconds % 60U;
        return result + " (" + std::to_string(minutes)
            + " min, " + std::to_string(remaining_seconds) + " s)";
    }

    const std::uint64_t hours = rounded_seconds / 3'600U;
    const std::uint64_t minutes = (rounded_seconds % 3'600U) / 60U;
    const std::uint64_t remaining_seconds = rounded_seconds % 60U;
    return result + " (" + std::to_string(hours) + " h, "
        + std::to_string(minutes) + " min, "
        + std::to_string(remaining_seconds) + " s)";
}

// Read one ordered JSON document from disk.
nlohmann::ordered_json read_json_file(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open JSON file: " + path.string());
    }
    nlohmann::ordered_json document;
    input >> document;
    return document;
}

// Write stable, two-space-indented JSON.
void write_json_file(
    const std::filesystem::path& path,
    const nlohmann::ordered_json& document
) {
    write_text_file(path, document.dump(2) + "\n");
}

// Write the metadata tree using the workbench YAML subset.
void write_yaml_file(
    const std::filesystem::path& path,
    const nlohmann::ordered_json& document
) {
    std::ostringstream output;
    write_yaml_value(output, document, 0U);
    write_text_file(path, output.str());
}

// Require one public HTTP(S) location for every complete dataset.
void validate_dataset_url(const std::string& url) {
    if (!(url.rfind("https://", 0U) == 0U
          || url.rfind("http://", 0U) == 0U)) {
        throw std::invalid_argument(
            "A dataset URL must start with http:// or https://."
        );
    }
}

// Round FP32 grid bounds for readable YAML metadata.
double readable_grid_bound(float value) {
    constexpr double scale = 10'000'000.0;
    return std::round(static_cast<double>(value) * scale) / scale;
}

// Reject empty or duplicate parameter names.
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

// Attach stable row identifiers to generated parameter objects.
nlohmann::ordered_json database_rows(
    const std::vector<ParameterRow>& parameters
) {
    nlohmann::ordered_json rows = nlohmann::ordered_json::array();
    for (std::size_t index = 0; index < parameters.size(); ++index) {
        rows.push_back({
            {"id", format_row_id(index)},
            {"parameters", parameters[index]},
        });
    }
    return rows;
}

// Write the common artifacts shared by curve, model, and product datasets.
void write_parameter_dataset(
    const std::string& database_id,
    const std::string& family,
    const std::string& family_key,
    const std::string& row_key,
    const std::string& definition_key,
    const std::filesystem::path& dataset_path,
    const std::filesystem::path& catalog_path,
    const std::string& url,
    const nlohmann::ordered_json& parameter_descriptions,
    const nlohmann::ordered_json& definition,
    const GeneratedRows& generated
) {
    if (generated.rows.empty()) {
        throw std::invalid_argument("A parameter dataset cannot be empty.");
    }
    validate_dataset_url(url);

    const nlohmann::ordered_json dataset = {
        {"database_id", database_id},
        {family_key, family},
        {"catalog", catalog_path.parent_path().generic_string()},
        {"url", url},
        {"row_count", generated.rows.size()},
        {row_key, database_rows(generated.rows)},
    };
    write_json_file(dataset_path, dataset);

    write_yaml_file(catalog_path, {
        {"title", family + " parameter dataset " + database_id},
        {"database_id", database_id},
        {family_key, family},
        {"catalog", catalog_path.parent_path().generic_string()},
        {"url", url},
        {"row_count", generated.rows.size()},
        {"parameters", parameter_descriptions},
        {definition_key, definition},
        {"construction", generated.construction},
    });
}

}  // namespace

// Sample every declared parameter independently across all rows.
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

// Preserve regime order and expose both generation recipes in catalog YAML.
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

// Combine parameter values that share the same vector index.
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

// Enumerate every combination of the supplied parameter grids.
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

// Generate a conditional log-strike grid for every maturity.
GeneratedRows maturity_dependent_exponential_strike_grid(
    const std::vector<float>& maturities,
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
    for (float maturity : maturities) {
        if (!(maturity > 0.0f) || !std::isfinite(maturity)) {
            throw std::invalid_argument(
                "Exponential strike-grid maturities must be positive and finite."
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
    for (float maturity : maturities) {
        const float radius = log_moneyness_slope * maturity;
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
                "For each T, x is linearly spaced on [-aT, aT] and K = exp(x)."
            },
            {"grid", {
                {"maturity", {
                    {"minimum", readable_grid_bound(*minimum_maturity)},
                    {"maximum", readable_grid_bound(*maximum_maturity)},
                    {"count", maturities.size()},
                    {"spacing", "linear"},
                }},
                {"strike", {
                    {"count_per_maturity", strikes_per_maturity},
                    {"spacing", "linear in log-strike"},
                    {"conditional_bounds", "[exp(-aT), exp(aT)]"},
                    {"a", readable_grid_bound(log_moneyness_slope)},
                }},
            }},
        },
    };
}

// Combine a representative strike/maturity grid with a wider stress grid.
GeneratedRows core_stress_exponential_strike_grid(
    const std::vector<float>& core_maturities,
    std::size_t core_strikes_per_maturity,
    float core_log_moneyness_slope,
    const std::vector<float>& stress_maturities,
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
                {"minimum", readable_grid_bound(*core_minimum)},
                {"maximum", readable_grid_bound(*core_maximum)},
                {"count", core_maturities.size()},
                {"spacing", "linear"},
            }},
            {"stress", {
                {"minimum", readable_grid_bound(*stress_minimum)},
                {"maximum", readable_grid_bound(*stress_maximum)},
                {"count", stress_maturities.size()},
                {"spacing", "linear"},
            }},
        }},
        {"strike", {
            {"core", {
                {"count_per_maturity", core_strikes_per_maturity},
                {"conditional_bounds", "[exp(-aT), exp(aT)]"},
                {"a", readable_grid_bound(core_log_moneyness_slope)},
            }},
            {"stress", {
                {"count_per_maturity", stress_strikes_per_maturity},
                {"conditional_bounds", "[exp(-aT), exp(aT)]"},
                {"a", readable_grid_bound(stress_log_moneyness_slope)},
            }},
            {"spacing", "linear in log-strike"},
        }},
    };
    return combined;
}

// Sample a convention with enough exercise dates including maturity.
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
        if (!(interval.years > 0.0f) || !std::isfinite(interval.years)
            || interval.label.empty()) {
            throw std::invalid_argument(
                "Exercise intervals require positive values and labels."
            );
        }
    }

    std::mt19937_64 generator(seed);
    const std::size_t required_pre_maturity_dates =
        minimum_exercise_count - 1U;
    for (ParameterRow& row : generated.rows) {
        const float maturity = row.at("maturity").get<float>();
        std::vector<float> feasible_intervals;
        for (const ExerciseInterval& interval : intervals) {
            if (static_cast<float>(required_pre_maturity_dates)
                    * interval.years < maturity) {
                feasible_intervals.push_back(interval.years);
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

// Generate an inclusive linear grid with stable FP32 arithmetic.
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

// Write one model dataset and its versioned metadata artifacts.
void write_model_dataset(
    const std::string& database_id,
    const std::string& model_family,
    const std::filesystem::path& dataset_path,
    const std::filesystem::path& catalog_path,
    const std::string& url,
    const nlohmann::ordered_json& parameter_descriptions,
    const nlohmann::ordered_json& dynamics,
    const GeneratedRows& generated
) {
    write_parameter_dataset(
        database_id, model_family, "model_family", "models", "dynamics",
        dataset_path, catalog_path, url, parameter_descriptions, dynamics,
        generated
    );
}

// Write one curve dataset and its versioned metadata artifacts.
void write_curve_dataset(
    const std::string& database_id,
    const std::string& curve_family,
    const std::filesystem::path& dataset_path,
    const std::filesystem::path& catalog_path,
    const std::string& url,
    const nlohmann::ordered_json& parameter_descriptions,
    const nlohmann::ordered_json& curve_definition,
    const GeneratedRows& generated
) {
    write_parameter_dataset(
        database_id, curve_family, "curve_family", "curves", "curve",
        dataset_path, catalog_path, url, parameter_descriptions,
        curve_definition, generated
    );
}

// Write one product dataset and its versioned metadata artifacts.
void write_product_dataset(
    const std::string& database_id,
    const std::string& product_family,
    const std::filesystem::path& dataset_path,
    const std::filesystem::path& catalog_path,
    const std::string& url,
    const nlohmann::ordered_json& parameter_descriptions,
    const nlohmann::ordered_json& payoff,
    const GeneratedRows& generated
) {
    write_parameter_dataset(
        database_id, product_family, "product_family", "products", "payoff",
        dataset_path, catalog_path, url, parameter_descriptions, payoff,
        generated
    );
}

namespace {

// Locate the source model and product rows of one result.
struct PriceIndices {
    std::size_t model;
    std::size_t product;
};

// Decode aligned or model-major Cartesian result indices.
PriceIndices price_indices(
    std::size_t result_index,
    std::size_t product_count,
    PriceConstruction construction
) {
    if (construction == PriceConstruction::Aligned) {
        return {result_index, result_index};
    }
    return {
        result_index / product_count,
        result_index % product_count,
    };
}

// Validate the construction and return its final row count.
std::size_t price_row_count_impl(
    std::size_t model_count,
    std::size_t product_count,
    PriceConstruction construction
) {
    if (model_count == 0U || product_count == 0U) {
        throw std::invalid_argument(
            "Price construction requires non-empty model and product datasets."
        );
    }
    if (construction == PriceConstruction::Aligned) {
        if (model_count != product_count) {
            throw std::invalid_argument(
                "Aligned construction requires equal model and product counts."
            );
        }
        return model_count;
    }
    if (model_count > std::numeric_limits<std::size_t>::max() / product_count) {
        throw std::overflow_error("Cartesian result count exceeds size_t.");
    }
    return model_count * product_count;
}

// Validate a model/curve/product construction and return its row count.
std::size_t price_row_count_impl(
    std::size_t model_count,
    std::size_t curve_count,
    std::size_t product_count,
    PriceConstruction construction
) {
    if (model_count == 0U || curve_count == 0U || product_count == 0U) {
        throw std::invalid_argument(
            "Price construction requires non-empty input datasets."
        );
    }
    if (construction == PriceConstruction::Aligned) {
        if (model_count != curve_count || model_count != product_count) {
            throw std::invalid_argument(
                "Aligned construction requires equal input row counts."
            );
        }
        return model_count;
    }
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    if (model_count > maximum / curve_count
        || model_count * curve_count > maximum / product_count) {
        throw std::overflow_error("Cartesian result count exceeds size_t.");
    }
    return model_count * curve_count * product_count;
}

// Write analytical prices without Monte Carlo-only fields.
void write_analytical_price_dataset_impl(
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
) {
    validate_dataset_url(url);
    const nlohmann::ordered_json model_document =
        read_json_file(model_dataset_path);
    const nlohmann::ordered_json curve_document =
        read_json_file(curve_dataset_path);
    const nlohmann::ordered_json product_document =
        read_json_file(product_dataset_path);
    const auto& model_rows = model_document.at("models");
    const auto& curve_rows = curve_document.at("curves");
    const auto& product_rows = product_document.at("products");
    const std::size_t row_count = price_row_count_impl(
        model_rows.size(),
        curve_rows.size(),
        product_rows.size(),
        construction
    );
    if (prices.size() != row_count) {
        throw std::invalid_argument(
            "Prices must match the constructed input row count."
        );
    }
    for (std::size_t index = 0U; index < row_count; ++index) {
        if (!std::isfinite(prices[index])) {
            throw std::runtime_error(
                "Price row id '" + format_row_id(index)
                + "': price must be finite."
            );
        }
    }

    const std::string database_id = dataset_path.stem().string();
    if (database_id.empty()) {
        throw std::invalid_argument(
            "The price dataset must have a non-empty basename."
        );
    }

    // Reference source identifiers without duplicating their parameters.
    nlohmann::ordered_json rows = nlohmann::ordered_json::array();
    for (std::size_t index = 0U; index < row_count; ++index) {
        std::size_t model_index = index;
        std::size_t curve_index = index;
        std::size_t product_index = index;
        if (construction == PriceConstruction::CartesianProduct) {
            const std::size_t curve_product_count =
                curve_rows.size() * product_rows.size();
            model_index = index / curve_product_count;
            const std::size_t remainder = index % curve_product_count;
            curve_index = remainder / product_rows.size();
            product_index = remainder % product_rows.size();
        }
        rows.push_back({
            {"id", format_row_id(index)},
            {"model_id", model_rows.at(model_index).at("id")},
            {"curve_id", curve_rows.at(curve_index).at("id")},
            {"product_id", product_rows.at(product_index).at("id")},
            {"outputs", {{"price", prices[index]}}},
        });
    }

    const auto dataset_reference = [](
        const nlohmann::ordered_json& document
    ) {
        return nlohmann::ordered_json{
            {"id", document.at("database_id")},
            {"catalog", document.at("catalog")},
            {"url", document.at("url")},
        };
    };
    const nlohmann::ordered_json json_document = {
        {"database_id", database_id},
        {"catalog", catalog_path.parent_path().generic_string()},
        {"url", url},
        {"row_count", row_count},
        {"model_dataset", dataset_reference(model_document)},
        {"curve_dataset", dataset_reference(curve_document)},
        {"product_dataset", dataset_reference(product_document)},
        {"timing", {
            {"wall_seconds", wall_seconds},
            {"kernel_seconds", kernel_seconds},
        }},
        {"results", rows},
    };
    write_json_file(dataset_path, json_document);

    if (!cuda_execution.is_object() || cuda_execution.empty()) {
        throw std::invalid_argument(
            "An analytical CUDA result requires execution metadata."
        );
    }
    nlohmann::ordered_json summary = {
        {"pricing_method", "Closed-form"},
        {"model", model_document.at("model_family")},
        {"curve", curve_document.at("curve_family")},
        {"numerical_method", numerical_method},
        {"payoff", product_document.at("product_family")},
        {"implementation", "CUDA"},
        {"device", "gpu"},
    };
    for (const auto& [name, value] : cuda_execution.items()) {
        summary[name] = value;
    }
    nlohmann::ordered_json price_construction = {{"method", "Aligned"}};
    if (construction == PriceConstruction::CartesianProduct) {
        price_construction = {
            {"method", "Cartesian product"},
            {"order", "model, curve, product"},
        };
    }
    const nlohmann::ordered_json catalog = {
        {"title", database_id},
        {"database_id", database_id},
        {"catalog", catalog_path.parent_path().generic_string()},
        {"url", url},
        {"row_count", row_count},
        {"summary", summary},
        {"outputs", {{"price", {{"estimator", "closed-form price"}}}}},
        {"model_dataset", dataset_reference(model_document)},
        {"curve_dataset", dataset_reference(curve_document)},
        {"product_dataset", dataset_reference(product_document)},
        {"price_construction", price_construction},
        {"timing", {
            {"wall_seconds", format_duration(wall_seconds)},
            {"kernel_seconds", format_duration(kernel_seconds)},
        }},
    };
    write_yaml_file(catalog_path, catalog);
}

// Write analytical prices from one model and one product dataset.
void write_analytical_price_dataset_impl(
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
) {
    validate_dataset_url(url);
    const nlohmann::ordered_json model_document =
        read_json_file(model_dataset_path);
    const nlohmann::ordered_json product_document =
        read_json_file(product_dataset_path);
    const auto& model_rows = model_document.at("models");
    const auto& product_rows = product_document.at("products");
    const std::size_t row_count = price_row_count_impl(
        model_rows.size(), product_rows.size(), construction
    );
    if (prices.size() != row_count) {
        throw std::invalid_argument(
            "Prices must match the constructed input row count."
        );
    }
    for (std::size_t index = 0U; index < row_count; ++index) {
        if (!std::isfinite(prices[index])) {
            throw std::runtime_error(
                "Price row id '" + format_row_id(index)
                + "': price must be finite."
            );
        }
    }

    const std::string database_id = dataset_path.stem().string();
    if (database_id.empty()) {
        throw std::invalid_argument(
            "The price dataset must have a non-empty basename."
        );
    }
    const auto dataset_reference = [](
        const nlohmann::ordered_json& document
    ) {
        return nlohmann::ordered_json{
            {"id", document.at("database_id")},
            {"catalog", document.at("catalog")},
            {"url", document.at("url")},
        };
    };

    nlohmann::ordered_json rows = nlohmann::ordered_json::array();
    for (std::size_t index = 0U; index < row_count; ++index) {
        const std::size_t model_index =
            construction == PriceConstruction::CartesianProduct
            ? index / product_rows.size()
            : index;
        const std::size_t product_index =
            construction == PriceConstruction::CartesianProduct
            ? index % product_rows.size()
            : index;
        rows.push_back({
            {"id", format_row_id(index)},
            {"model_id", model_rows.at(model_index).at("id")},
            {"product_id", product_rows.at(product_index).at("id")},
            {"outputs", {{"price", prices[index]}}},
        });
    }

    write_json_file(dataset_path, {
        {"database_id", database_id},
        {"catalog", catalog_path.parent_path().generic_string()},
        {"url", url},
        {"row_count", row_count},
        {"model_dataset", dataset_reference(model_document)},
        {"product_dataset", dataset_reference(product_document)},
        {"timing", {
            {"wall_seconds", wall_seconds},
            {"kernel_seconds", kernel_seconds},
        }},
        {"results", rows},
    });

    if (!cuda_execution.is_object() || cuda_execution.empty()) {
        throw std::invalid_argument(
            "An analytical CUDA result requires execution metadata."
        );
    }
    nlohmann::ordered_json summary = {
        {"pricing_method", "Closed-form"},
        {"model", model_document.at("model_family")},
        {"numerical_method", numerical_method},
        {"payoff", product_document.at("product_family")},
        {"implementation", "CUDA"},
        {"device", "gpu"},
    };
    for (const auto& [name, value] : cuda_execution.items()) {
        summary[name] = value;
    }
    nlohmann::ordered_json price_construction = {{"method", "Aligned"}};
    if (construction == PriceConstruction::CartesianProduct) {
        price_construction = {
            {"method", "Cartesian product"},
            {"order", "model, product"},
        };
    }
    write_yaml_file(catalog_path, {
        {"title", database_id},
        {"database_id", database_id},
        {"catalog", catalog_path.parent_path().generic_string()},
        {"url", url},
        {"row_count", row_count},
        {"summary", summary},
        {"outputs", {{"price", {{"estimator", "closed-form price"}}}}},
        {"model_dataset", dataset_reference(model_document)},
        {"product_dataset", dataset_reference(product_document)},
        {"price_construction", price_construction},
        {"timing", {
            {"wall_seconds", format_duration(wall_seconds)},
            {"kernel_seconds", format_duration(kernel_seconds)},
        }},
    });
}

// Serialize one complete Monte Carlo price dataset and its metadata.
void write_monte_carlo_price_dataset_impl(
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
    const std::string& target_dt,
    const nlohmann::ordered_json& cuda_execution,
    const nlohmann::ordered_json& catalog_sections,
    std::uint64_t first_seed,
    double wall_seconds,
    double kernel_seconds
) {
    validate_dataset_url(url);
    const nlohmann::ordered_json model_document =
        read_json_file(model_dataset_path);
    const nlohmann::ordered_json product_document =
        read_json_file(product_dataset_path);
    const auto& model_rows = model_document.at("models");
    const auto& product_rows = product_document.at("products");
    const std::size_t row_count = price_row_count_impl(
        model_rows.size(), product_rows.size(), construction
    );
    if (prices.size() != row_count || standard_errors.size() != row_count) {
        throw std::invalid_argument(
            "Price vectors must match the constructed input row count."
        );
    }
    for (std::size_t index = 0U; index < row_count; ++index) {
        if (!std::isfinite(prices[index])
            || !std::isfinite(standard_errors[index])
            || standard_errors[index] < 0.0f) {
            throw std::runtime_error(
                "Price row id '" + format_row_id(index)
                + "': price must be finite and standard_error must be "
                "finite and non-negative."
            );
        }
    }

    const std::string model_database_id =
        model_document.at("database_id").get<std::string>();
    const std::string product_database_id =
        product_document.at("database_id").get<std::string>();
    const std::string database_id = dataset_path.stem().string();
    if (database_id.empty()) {
        throw std::invalid_argument(
            "The price dataset must have a non-empty basename."
        );
    }
    nlohmann::ordered_json price_construction = {{"method", "Aligned"}};
    if (construction == PriceConstruction::CartesianProduct) {
        price_construction = {
            {"method", "Cartesian product"},
            {"order", "model, product"},
        };
    }

    // Reference source rows instead of duplicating their parameters.
    nlohmann::ordered_json rows = nlohmann::ordered_json::array();
    for (std::size_t index = 0; index < row_count; ++index) {
        const PriceIndices source = price_indices(
            index, product_rows.size(), construction
        );
        nlohmann::ordered_json row = {
            {"id", format_row_id(index)},
            {"model_id", model_rows.at(source.model).at("id").get<std::string>()},
            {
                "product_id",
                product_rows.at(source.product).at("id").get<std::string>()
            },
            {"seed", first_seed + index},
        };
        row["outputs"] = {
            {"price", prices[index]},
            {"standard_error", standard_errors[index]},
        };
        rows.push_back(std::move(row));
    }

    const nlohmann::ordered_json json_model_dataset = {
        {"id", model_database_id},
        {"catalog", model_document.at("catalog")},
        {"url", model_document.at("url")},
    };
    const nlohmann::ordered_json json_product_dataset = {
        {"id", product_database_id},
        {"catalog", product_document.at("catalog")},
        {"url", product_document.at("url")},
    };
    const nlohmann::ordered_json json_document = {
        {"database_id", database_id},
        {"catalog", catalog_path.parent_path().generic_string()},
        {"url", url},
        {"row_count", row_count},
        {"model_dataset", json_model_dataset},
        {"product_dataset", json_product_dataset},
        {"timing", {
            {"wall_seconds", wall_seconds},
            {"kernel_seconds", kernel_seconds},
        }},
        {"results", rows},
    };
    write_json_file(dataset_path, json_document);

    if (!cuda_execution.is_object() || cuda_execution.empty()) {
        throw std::invalid_argument(
            "A Monte Carlo result requires CUDA execution metadata."
        );
    }
    nlohmann::ordered_json summary = {
        {"pricing_method", "Monte Carlo"},
        {"monte_carlo_paths_per_price", monte_carlo_paths_per_price},
        {"model", model_document.at("model_family")},
        {"numerical_method", numerical_method},
        {"payoff", product_document.at("product_family")},
        {"implementation", "CUDA"},
        {"device", "gpu"},
    };
    for (const auto& [name, value] : cuda_execution.items()) {
        summary[name] = value;
    }
    summary["random_generator"] = random_generator;
    const nlohmann::ordered_json yaml_model_dataset = {
        {"id", model_database_id},
        {"catalog", model_document.at("catalog")},
        {"url", model_document.at("url")},
    };
    const nlohmann::ordered_json yaml_product_dataset = {
        {"id", product_database_id},
        {"catalog", product_document.at("catalog")},
        {"url", product_document.at("url")},
    };
    const nlohmann::ordered_json yaml_timing = {
        {"wall_seconds", format_duration(wall_seconds)},
        {"kernel_seconds", format_duration(kernel_seconds)},
    };
    nlohmann::ordered_json catalog = {
        {"title", database_id},
        {"database_id", database_id},
        {"catalog", catalog_path.parent_path().generic_string()},
        {"url", url},
        {"row_count", row_count},
        {"summary", summary},
        {"time_grid", {
            {"rule", "nearest integer step count to target dt"},
            {"target_dt", target_dt},
            {"step_count", "round(maturity / target_dt)"},
            {"effective_dt", "maturity / step_count"},
        }},
        {"outputs", {
            {"price", {{"estimator", "Monte Carlo discounted payoff mean"}}},
            {
                "standard_error",
                {{"estimator", "Monte Carlo standard error of discounted payoff"}}
            },
        }},
        {"model_dataset", yaml_model_dataset},
        {"product_dataset", yaml_product_dataset},
        {"price_construction", price_construction},
        {"timing", yaml_timing},
    };
    if (!catalog_sections.is_object()) {
        throw std::invalid_argument(
            "Additional catalog sections must form an object."
        );
    }
    for (const auto& [name, value] : catalog_sections.items()) {
        catalog[name] = value;
    }
    write_yaml_file(catalog_path, catalog);
}

}  // namespace

// Return the number of rows implied by the selected price construction.
std::size_t price_row_count(
    std::size_t model_count,
    std::size_t product_count,
    PriceConstruction construction
) {
    return price_row_count_impl(model_count, product_count, construction);
}

// Return the rows implied by one model, curve, and product construction.
std::size_t price_row_count(
    std::size_t model_count,
    std::size_t curve_count,
    std::size_t product_count,
    PriceConstruction construction
) {
    return price_row_count_impl(
        model_count, curve_count, product_count, construction
    );
}

// Write one complete Monte Carlo price dataset and catalog entry.
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
    const std::string& target_dt,
    const nlohmann::ordered_json& cuda_execution,
    const nlohmann::ordered_json& catalog_sections,
    std::uint64_t first_seed,
    double wall_seconds,
    double kernel_seconds
) {
    write_monte_carlo_price_dataset_impl(
        model_dataset_path,
        product_dataset_path,
        construction,
        prices,
        standard_errors,
        random_generator,
        dataset_path,
        catalog_path,
        url,
        numerical_method,
        monte_carlo_paths_per_price,
        target_dt,
        cuda_execution,
        catalog_sections,
        first_seed,
        wall_seconds,
        kernel_seconds
    );
}

// Write one complete closed-form price dataset and catalog entry.
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
) {
    write_analytical_price_dataset_impl(
        model_dataset_path,
        product_dataset_path,
        construction,
        prices,
        dataset_path,
        catalog_path,
        url,
        numerical_method,
        cuda_execution,
        wall_seconds,
        kernel_seconds
    );
}

// Write one complete closed-form price dataset and catalog entry.
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
) {
    write_analytical_price_dataset_impl(
        model_dataset_path,
        curve_dataset_path,
        product_dataset_path,
        construction,
        prices,
        dataset_path,
        catalog_path,
        url,
        numerical_method,
        cuda_execution,
        wall_seconds,
        kernel_seconds
    );
}

}  // namespace ai_factory::workbench::datasets
