// Shared structural validation for model, curve, product, and price JSON.
#pragma once

#include <nlohmann/json_fwd.hpp>

#include <filesystem>

namespace ai_factory::workbench::datasets {

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
