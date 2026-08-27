// Filesystem serialization for offline JSON and catalog-YAML artifacts.
#pragma once

#include <nlohmann/json.hpp>

#include <cstddef>
#include <filesystem>
#include <string>

namespace ai_factory::workbench::datasets {

void write_catalog_yaml(
    const std::filesystem::path& path,
    const nlohmann::ordered_json& document
);

// Shared artifact primitives. They contain no pricing or sampling policy and
// are independently testable with temporary files.
std::string format_row_id(std::size_t index);
std::string format_duration(double seconds);
nlohmann::ordered_json read_json_file(const std::filesystem::path& path);
nlohmann::ordered_json price_validation_metadata(
    const std::filesystem::path& catalog_directory
);
void write_json_file(
    const std::filesystem::path& path,
    const nlohmann::ordered_json& document
);
void validate_dataset_url(const std::string& url);

}  // namespace ai_factory::workbench::datasets
