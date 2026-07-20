// JSON/YAML input-output implementation shared by registry writers.
#include "tools/registry/io.hpp"

#include <fstream>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace ai_factory::workbench::registry {
namespace {

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

}  // namespace

std::string format_row_id(std::size_t index) {
    std::ostringstream stream;
    stream << std::setw(6) << std::setfill('0') << index + 1U;
    return stream.str();
}

nlohmann::ordered_json read_json_file(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open JSON file: " + path.string());
    }
    nlohmann::ordered_json document;
    input >> document;
    return document;
}

void write_json_file(
    const std::filesystem::path& path,
    const nlohmann::ordered_json& document
) {
    write_text_file(path, document.dump(2) + "\n");
}

void write_yaml_file(
    const std::filesystem::path& path,
    const nlohmann::ordered_json& document
) {
    std::ostringstream output;
    write_yaml_value(output, document, 0U);
    write_text_file(path, output.str());
}

}  // namespace ai_factory::workbench::registry
