// Assemble and publish offline analytical and Monte Carlo price artifacts.
#pragma once

#include "common/price_construction.cuh"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ai_factory::workbench::datasets {

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
    const std::string& delta_t,
    const nlohmann::ordered_json& cuda_execution,
    const nlohmann::ordered_json& catalog_sections,
    std::uint64_t first_seed,
    double wall_seconds,
    double kernel_seconds
);

void write_monte_carlo_price_dataset(
    const std::filesystem::path& model_dataset_path,
    const std::filesystem::path& curve_dataset_path,
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
    const std::string& delta_t,
    const nlohmann::ordered_json& cuda_execution,
    const nlohmann::ordered_json& catalog_sections,
    std::uint64_t first_seed,
    double wall_seconds,
    double kernel_seconds
);

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
);

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
);

}  // namespace ai_factory::workbench::datasets
