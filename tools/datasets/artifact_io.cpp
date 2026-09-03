// JSON and YAML filesystem serialization for offline artifacts.
#include "tools/datasets/artifact_io.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::datasets {
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

void write_yaml_file(
    const std::filesystem::path& path,
    const nlohmann::ordered_json& document
) {
    std::ostringstream output;
    write_yaml_value(output, document, 0U);
    write_text_file(path, output.str());
}

}  // namespace

std::string format_row_id(std::size_t index) {
    std::ostringstream stream;
    stream << std::setw(6) << std::setfill('0') << index + 1U;
    return stream.str();
}

std::string format_duration(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0.0) {
        throw std::invalid_argument(
            "A formatted duration must be finite and non-negative; received "
            + std::to_string(seconds) + "."
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

nlohmann::ordered_json read_json_file(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open JSON file: " + path.string());
    }
    nlohmann::ordered_json document;
    input >> document;
    return document;
}

nlohmann::ordered_json price_validation_metadata(
    const std::filesystem::path& catalog_directory
) {
    return {
        {"status", "pending"},
        {"verified", false},
        {"reference", "none"},
        {
            "notebook",
            (catalog_directory / "validation.ipynb").generic_string()
        },
    };
}

void write_json_file(
    const std::filesystem::path& path,
    const nlohmann::ordered_json& document
) {
    write_text_file(path, document.dump(2) + "\n");
}

void validate_dataset_url(const std::string& url) {
    if (!(url.rfind("https://", 0U) == 0U
          || url.rfind("http://", 0U) == 0U)) {
        throw std::invalid_argument(
            "A dataset URL must start with http:// or https://."
        );
    }
}

void write_catalog_yaml(
    const std::filesystem::path& path,
    const nlohmann::ordered_json& document
) {
    write_yaml_file(path, document);
}

}  // namespace ai_factory::workbench::datasets
