// Parameter generation and model/product database writing implementation.
#include "tools/registry/parameters.hpp"

#include "tools/registry/io.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace ai_factory::workbench::registry {
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

void write_parameter_database(
    const std::string& database_id,
    const std::string& family,
    const std::string& family_key,
    const std::string& row_key,
    const std::string& definition_key,
    const std::filesystem::path& json_path,
    const std::filesystem::path& generation_script,
    const nlohmann::ordered_json& parameter_descriptions,
    const nlohmann::ordered_json& definition,
    const GeneratedRows& generated
) {
    if (generated.rows.empty()) {
        throw std::invalid_argument("A parameter database cannot be empty.");
    }
    const std::filesystem::path specification_path =
        json_path.parent_path().parent_path()
        / "specifications" / (database_id + ".yaml");

    write_json_file(json_path, {
        {"database_id", database_id},
        {family_key, family},
        {"specification", specification_path.generic_string()},
        {"generation_script", generation_script.generic_string()},
        {"row_count", generated.rows.size()},
        {row_key, database_rows(generated.rows)},
    });

    write_yaml_file(specification_path, {
        {"title", family + " parameter database " + database_id},
        {"database_id", database_id},
        {family_key, family},
        {"json_path", json_path.generic_string()},
        {"generation_script", generation_script.generic_string()},
        {"parameters", parameter_descriptions},
        {definition_key, definition},
        {"construction", generated.construction},
    });
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
        bounds[parameter.name] = {parameter.minimum, parameter.maximum};
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
            {"row_count", row_count},
            {"method", "independent uniform sample"},
            {"bounds", bounds},
        },
    };
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
            {"row_count", row_count},
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
            {"row_count", row_count},
            {"method", "Cartesian grid"},
            {"grid", grid},
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

void write_model_database(
    const std::string& database_id,
    const std::string& model_family,
    const std::filesystem::path& json_path,
    const std::filesystem::path& generation_script,
    const nlohmann::ordered_json& parameter_descriptions,
    const nlohmann::ordered_json& dynamics,
    const GeneratedRows& generated
) {
    write_parameter_database(
        database_id, model_family, "model_family", "models", "dynamics",
        json_path, generation_script, parameter_descriptions, dynamics, generated
    );
}

void write_product_database(
    const std::string& database_id,
    const std::string& product_family,
    const std::filesystem::path& json_path,
    const std::filesystem::path& generation_script,
    const nlohmann::ordered_json& parameter_descriptions,
    const nlohmann::ordered_json& payoff,
    const GeneratedRows& generated
) {
    write_parameter_database(
        database_id, product_family, "product_family", "products", "payoff",
        json_path, generation_script, parameter_descriptions, payoff, generated
    );
}

}  // namespace ai_factory::workbench::registry
