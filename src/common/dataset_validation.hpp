// Host-side structural validation for model, curve, product, and price JSON.
#pragma once

#include <nlohmann/json.hpp>

#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ai_factory::workbench::datasets {

// Select the common envelope and row collection of a parameter dataset.
enum class ParameterDatasetFamily {
    Model,
    Curve,
    Product,
};

// Open, parse, and validate one model/curve/product artifact. This is the only
// filesystem/JSON entry point used by runtime parameter loaders.
nlohmann::json read_parameter_dataset(
    const std::filesystem::path& dataset_path,
    ParameterDatasetFamily family,
    const std::string& display_name
);

// Apply one family-specific row parser while the common loader owns document
// I/O, envelope validation, row extraction, capacity planning, and JSON-field
// diagnostics. The callback owns only conversion and semantic row validation.
template<typename Output, typename ParseRow>
std::vector<Output> load_parameter_rows(
    const std::filesystem::path& dataset_path,
    ParameterDatasetFamily family,
    const std::string& display_name,
    ParseRow&& parse_row
) {
    const nlohmann::json document = read_parameter_dataset(
        dataset_path, family, display_name
    );
    const char* collection = family == ParameterDatasetFamily::Model
        ? "models"
        : family == ParameterDatasetFamily::Curve ? "curves" : "products";
    const auto& input_rows = document.at(collection);
    std::vector<Output> output_rows;
    output_rows.reserve(input_rows.size());
    for (const auto& row : input_rows) {
        const std::string row_id = row.at("id").template get<std::string>();
        const std::string prefix =
            display_name + " row id '" + row_id + "': ";
        try {
            output_rows.push_back(std::invoke(
                parse_row, row.at("parameters"), prefix
            ));
        } catch (const nlohmann::json::exception& error) {
            throw std::invalid_argument(
                prefix + "invalid or missing JSON field: " + error.what()
            );
        }
    }
    return output_rows;
}

// Validate the common envelope and rows of one model dataset.
void validate_model_dataset(const nlohmann::json& document);

// Validate the common envelope and rows of one curve dataset.
void validate_curve_dataset(const nlohmann::json& document);

// Validate the common envelope and rows of one product dataset.
void validate_product_dataset(const nlohmann::json& document);

// Validate one price dataset with an optional curve reference.
void validate_price_dataset(const nlohmann::json& document);

// Reload and validate one generated model JSON artifact.
void validate_model_dataset_file(const std::filesystem::path& dataset_path);

// Reload and validate one generated curve JSON artifact.
void validate_curve_dataset_file(const std::filesystem::path& dataset_path);

// Reload and validate one generated product JSON artifact.
void validate_product_dataset_file(const std::filesystem::path& dataset_path);

// Reload and validate one generated price JSON artifact.
void validate_price_dataset_file(const std::filesystem::path& dataset_path);

}  // namespace ai_factory::workbench::datasets
