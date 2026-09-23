// Filesystem serialization for offline JSON and catalog-YAML artifacts.
#pragma once

#include <nlohmann/json.hpp>

#include <cstddef>
#include <filesystem>
#include <string>

namespace ai_factory::workbench::datasets {

void write_yaml_document(
    const std::filesystem::path& path,
    const nlohmann::ordered_json& document
);

void write_generation_receipt(
    const std::filesystem::path& path,
    std::size_t row_count,
    const nlohmann::ordered_json& execution,
    double wall_seconds,
    double kernel_seconds
);

// Shared artifact primitives. They contain no pricing or sampling policy and
// are independently testable with temporary files.
std::string format_row_id(std::size_t index);
std::string format_duration(double seconds);
nlohmann::ordered_json read_json_file(const std::filesystem::path& path);
// Generation never certifies prices. Canonical model datasets also name their
// intended independent-reference cache; its existence is not implied.
nlohmann::ordered_json price_validation_metadata(
    const std::filesystem::path& dataset_path
);
void write_json_file(
    const std::filesystem::path& path,
    const nlohmann::ordered_json& document
);
void validate_dataset_url(const std::string& url);

}  // namespace ai_factory::workbench::datasets
