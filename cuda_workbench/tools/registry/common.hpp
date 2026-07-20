// Small registry utilities shared by parameter and result database writers.
// This file owns construction names and plain JSON/YAML file operations only.
#pragma once

#include <nlohmann/json.hpp>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench {

// ConstructionMethod selects row-wise alignment or a full Cartesian product.
enum class ConstructionMethod : unsigned int {
    Aligned,
    CartesianProduct,
};

namespace registry {

// Format a zero-based index as the registry's six-digit row identifier.
inline std::string row_id(std::size_t index) {
    std::ostringstream stream;
    stream << std::setw(6) << std::setfill('0') << index + 1U;
    return stream.str();
}

// Parse one JSON document and report the source path on failure.
inline nlohmann::ordered_json read_json(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open JSON file: " + path.string());
    }
    nlohmann::ordered_json document;
    input >> document;
    return document;
}

// Create parent directories and write one complete text artifact.
inline void write_text_file(
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

// Return the number of result rows implied by a construction method.
inline std::size_t constructed_row_count(
    std::size_t model_count,
    std::size_t product_count,
    ConstructionMethod construction
) {
    if (construction == ConstructionMethod::Aligned) {
        if (model_count != product_count) {
            throw std::invalid_argument(
                "Aligned construction requires equal model and product counts."
            );
        }
        return model_count;
    }
    if (product_count != 0U
        && model_count > std::numeric_limits<std::size_t>::max() / product_count) {
        throw std::overflow_error("Cartesian result count exceeds size_t.");
    }
    return model_count * product_count;
}

// Serialize a JSON-compatible metadata tree as simple, consistently indented YAML.
inline void write_yaml_value(
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

// Write a metadata object as YAML using the serializer above.
inline void write_yaml_file(
    const std::filesystem::path& path,
    const nlohmann::ordered_json& document
) {
    std::ostringstream output;
    write_yaml_value(output, document, 0U);
    write_text_file(path, output.str());
}

}  // namespace registry
}  // namespace ai_factory::workbench
