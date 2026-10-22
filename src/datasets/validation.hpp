// Host-side structural validation for model, curve, product, and price JSON.
#pragma once

#include <nlohmann/json.hpp>

#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
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

// A declared relative fixed-income quote changes the units read by pricing.
// An absent declaration retains the absolute-input convention.
inline bool fixed_income_relative_quote(
    const nlohmann::json& document,
    std::string_view strike_mode,
    std::string_view strike_unit,
    std::string_view barrier_mode = {}
) {
    if (!document.contains("quote_convention")) return false;
    const auto& convention = document.at("quote_convention");
    const auto matches = [&](std::string_view field, std::string_view mode,
                             std::string_view unit) {
        if (!convention.is_object()
            || !convention.contains(std::string(field))) return false;
        const auto& entry = convention.at(std::string(field));
        return entry.is_object()
            && entry.contains("mode") && entry.at("mode").is_string()
            && entry.at("mode") == std::string(mode)
            && entry.contains("unit") && entry.at("unit").is_string()
            && entry.at("unit") == std::string(unit);
    };
    if (!matches("strike", strike_mode, strike_unit)
        || (!barrier_mode.empty()
            && !matches("barrier", barrier_mode, "dimensionless"))) {
        throw std::invalid_argument("Invalid fixed-income quote_convention.");
    }
    return true;
}

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
    if constexpr (requires(Output& value) { value.quote_is_ratio; }) {
        if (family != ParameterDatasetFamily::Product
            || !document.contains("quote_convention")
            || !document.at("quote_convention").is_object()) {
            throw std::invalid_argument(
                display_name + " requires product quote_convention."
            );
        }
        const auto& convention = document.at("quote_convention");
        const auto require_ratio = [&](const char* field, const char* mode) {
            if (!convention.contains(field)
                || !convention.at(field).is_object()
                || !convention.at(field).contains("mode")
                || !convention.at(field).at("mode").is_string()
                || convention.at(field).at("mode") != mode
                || !convention.at(field).contains("unit")
                || convention.at(field).at("unit") != "dimensionless") {
                throw std::invalid_argument(
                    display_name + " quote_convention." + field
                    + " must declare " + mode + " in dimensionless units."
                );
            }
        };
        constexpr bool intrinsically_relative = [] {
            if constexpr (requires { Output::kAlreadyRelativePriceLevels; })
                return Output::kAlreadyRelativePriceLevels;
            return false;
        }();
        if constexpr (intrinsically_relative) {
            require_ratio("lower_barrier", "ratio_to_initial_spot");
            require_ratio("upper_barrier", "ratio_to_initial_spot");
        } else {
            if constexpr (requires(Output value) { value.strike; })
                require_ratio("strike", "ratio_to_central_spot");
            if constexpr (requires(Output value) { value.trigger_strike; })
                require_ratio("trigger_strike", "ratio_to_central_spot");
            if constexpr (requires(Output value) { value.payoff_strike; })
                require_ratio("payoff_strike", "ratio_to_central_spot");
            if constexpr (requires(Output value) { value.barrier; })
                require_ratio("barrier", "ratio_to_central_spot");
            if constexpr (requires(Output value) { value.lower_barrier; })
                require_ratio("lower_barrier", "ratio_to_central_spot");
            if constexpr (requires(Output value) { value.upper_barrier; })
                require_ratio("upper_barrier", "ratio_to_central_spot");
            if constexpr (requires(Output value) { value.autocall_barrier; })
                require_ratio("autocall_barrier", "ratio_to_central_spot");
            if constexpr (requires(Output value) { value.coupon_barrier; })
                require_ratio("coupon_barrier", "ratio_to_central_spot");
            if constexpr (requires(Output value) { value.protection_barrier; })
                require_ratio("protection_barrier", "ratio_to_central_spot");
        }
    }
    std::vector<Output> output_rows;
    output_rows.reserve(input_rows.size());
    for (const auto& row : input_rows) {
        const std::string row_id = row.at("id").template get<std::string>();
        const std::string prefix =
            display_name + " row id '" + row_id + "': ";
        try {
            auto parsed = std::invoke(
                parse_row, row.at("parameters"), prefix
            );
            if constexpr (requires(Output& value) { value.quote_is_ratio; }) {
                if constexpr (requires { Output::kAlreadyRelativePriceLevels; }) {
                    parsed.quote_is_ratio =
                        family == ParameterDatasetFamily::Product
                        && !Output::kAlreadyRelativePriceLevels;
                } else {
                    parsed.quote_is_ratio =
                        family == ParameterDatasetFamily::Product;
                }
            }
            output_rows.push_back(std::move(parsed));
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
