// Assemble and publish model, curve, and product parameter artifacts.
#pragma once

#include "tools/datasets/sampling.hpp"

#include <filesystem>
#include <string>

namespace ai_factory::workbench::datasets {

void write_model_dataset(
    const std::string& database_id,
    const std::string& model_family,
    const std::filesystem::path& dataset_path,
    const std::filesystem::path& catalog_path,
    const std::string& url,
    const nlohmann::ordered_json& parameter_descriptions,
    const nlohmann::ordered_json& dynamics,
    const GeneratedRows& generated
);

// Project an existing complete model base onto its endogenous parameters.
// The new artifact contains no risk-free rate or Brownian correlation.
void write_factorized_model_dataset(
    const std::filesystem::path& source_dataset_path,
    const std::string& database_id,
    const std::filesystem::path& dataset_path,
    const std::filesystem::path& catalog_path,
    const std::string& url
);

// Draw a reproducible core-only factor base from the first core rows of a
// complete model dataset, preserving the native model parameter domain.
void write_core_factorized_model_dataset(
    const std::filesystem::path& source_dataset_path,
    const std::string& database_id,
    const std::filesystem::path& dataset_path,
    const std::filesystem::path& catalog_path,
    const std::string& url,
    std::size_t core_prefix_count,
    std::size_t row_count,
    std::uint64_t seed
);

void write_curve_dataset(
    const std::string& database_id,
    const std::string& curve_family,
    const std::filesystem::path& dataset_path,
    const std::filesystem::path& catalog_path,
    const std::string& url,
    const nlohmann::ordered_json& parameter_descriptions,
    const nlohmann::ordered_json& curve_definition,
    const GeneratedRows& generated
);

void write_product_dataset(
    const std::string& database_id,
    const std::string& product_family,
    const std::filesystem::path& dataset_path,
    const std::filesystem::path& catalog_path,
    const std::string& url,
    const nlohmann::ordered_json& parameter_descriptions,
    const nlohmann::ordered_json& payoff,
    const GeneratedRows& generated,
    const nlohmann::ordered_json& quote_convention = {}
);

// Fixed-income quotes are model-independent; pricing resolves them against
// the central initial term structure before any sensitivity bump.
void write_fixed_income_product_dataset(
    const std::string& database_id,
    const std::string& product_family,
    const std::filesystem::path& dataset_path,
    const std::filesystem::path& catalog_path,
    const std::string& url,
    const nlohmann::ordered_json& parameter_descriptions,
    const nlohmann::ordered_json& payoff,
    const GeneratedRows& generated,
    const nlohmann::ordered_json& quote_convention
);

// Equity strike and absolute-barrier fields are quoted as S0 ratios.
void write_spot_relative_equity_product_dataset(
    const std::string& database_id,
    const std::string& product_family,
    const std::filesystem::path& dataset_path,
    const std::filesystem::path& catalog_path,
    const std::string& url,
    const nlohmann::ordered_json& parameter_descriptions,
    const nlohmann::ordered_json& payoff,
    const GeneratedRows& generated
);

}  // namespace ai_factory::workbench::datasets
