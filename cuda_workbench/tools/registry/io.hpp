// Shared JSON/YAML input-output API for registry artifacts.
#pragma once

#include <nlohmann/json.hpp>

#include <cstddef>
#include <filesystem>
#include <string>

namespace ai_factory::workbench::registry {

// Format a zero-based index as the registry's six-digit row identifier.
std::string format_row_id(std::size_t index);

// Parse one JSON document and report the source path on failure.
nlohmann::ordered_json read_json_file(const std::filesystem::path& path);

// Serialize one JSON registry artifact with stable indentation.
void write_json_file(
    const std::filesystem::path& path,
    const nlohmann::ordered_json& document
);

// Serialize one metadata tree as the registry's simple YAML representation.
void write_yaml_file(
    const std::filesystem::path& path,
    const nlohmann::ordered_json& document
);

}  // namespace ai_factory::workbench::registry
