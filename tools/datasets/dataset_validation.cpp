// Structural checks shared by every JSON dataset family.
#include "tools/datasets/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace ai_factory::workbench::datasets {
namespace {

// Use the database identifier in every structural error when available.
std::string database_id(const nlohmann::json& document) {
    if (document.is_object() && document.contains("database_id")
        && document.at("database_id").is_string()) {
        return document.at("database_id").get<std::string>();
    }
    return "<unknown>";
}

// Stop loading with one uniform dataset-level diagnostic.
[[noreturn]] void reject(
    const char* family,
    const std::string& id,
    const std::string& reason
) {
    throw std::invalid_argument(
        std::string(family) + " dataset id '" + id + "': " + reason
    );
}

// Require one non-empty textual field and return its value.
const std::string& require_string(
    const nlohmann::json& object,
    const char* field,
    const char* family,
    const std::string& id
) {
    if (!object.contains(field) || !object.at(field).is_string()
        || object.at(field).get_ref<const std::string&>().empty()) {
        reject(
            family, id,
            std::string(field) + " must be a non-empty string."
        );
    }
    return object.at(field).get_ref<const std::string&>();
}

// Require a non-empty HTTP(S) dataset location.
void require_url(
    const nlohmann::json& object,
    const char* field,
    const char* family,
    const std::string& id
) {
    const std::string& url = require_string(object, field, family, id);
    if (!(url.rfind("https://", 0U) == 0U
          || url.rfind("http://", 0U) == 0U)) {
        reject(family, id, std::string(field) + " must be an HTTP(S) URL.");
    }
}

// Require a positive row count matching the corresponding JSON array.
void require_rows(
    const nlohmann::json& document,
    const char* row_field,
    const char* family,
    const std::string& id
) {
    if (!document.contains("row_count")
        || !document.at("row_count").is_number_unsigned()) {
        reject(family, id, "row_count must be a positive integer.");
    }
    if (!document.contains(row_field) || !document.at(row_field).is_array()) {
        reject(
            family, id,
            std::string(row_field) + " must be an array."
        );
    }
    const std::size_t row_count =
        document.at("row_count").get<std::size_t>();
    const auto& rows = document.at(row_field);
    if (row_count == 0U || rows.size() != row_count) {
        reject(
            family, id,
            "row_count must be positive and match "
                + std::string(row_field) + ".size()."
        );
    }
    std::unordered_set<std::string> row_ids;
    for (const auto& row : rows) {
        if (!row.is_object())
            reject(family, id, std::string(row_field) + " rows must be objects.");
        const std::string& row_id = require_string(row, "id", family, id);
        if (!row_ids.insert(row_id).second) {
            reject(
                family, id,
                "row id '" + row_id + "' must be unique."
            );
        }
        if (!row.contains("parameters") || !row.at("parameters").is_object()) {
            reject(
                family, id,
                "row id '" + row_id + "': parameters must be an object."
            );
        }
    }
}

// Validate the envelope shared by model, curve, and product datasets.
void validate_parameter_dataset(
    const nlohmann::json& document,
    const char* family,
    const char* family_field,
    const char* row_field
) {
    if (!document.is_object())
        reject(family, "<unknown>", "the JSON root must be an object.");
    const std::string id = database_id(document);
    require_string(document, "database_id", family, id);
    require_string(document, family_field, family, id);
    require_string(document, "catalog", family, id);
    require_url(document, "url", family, id);
    require_rows(document, row_field, family, id);
}

void validate_time_convention(
    const nlohmann::json& document,
    const char* family,
    const std::string& id
) {
    if (!document.contains("time_convention")
        || !document.at("time_convention").is_object()) {
        reject(family, id, "time_convention must be an object.");
    }
    const auto& convention = document.at("time_convention");
    require_string(convention, "unit", family, id);
    if (!convention.contains("days_per_year")
        || !convention.at("days_per_year").is_number_unsigned()
        || convention.at("days_per_year").get<std::uint32_t>() == 0U) {
        reject(
            family, id,
            "time_convention.days_per_year must be a positive integer."
        );
    }
}

bool is_business_day_field(const std::string& name) {
    return name == "maturity"
        || name == "exercise_time"
        || name == "exercise_interval"
        || name == "reset_time"
        || name == "observation_interval"
        || name == "fixing_time"
        || name == "payment_time"
        || name == "accrual_period"
        || name == "option_expiry"
        || name == "bond_maturity";
}

bool is_business_day_array_field(const std::string& name) {
    return name == "payment_times"
        || name == "accrual_periods"
        || name == "fixing_times"
        || name == "option_expiries"
        || name == "bond_tenors";
}

void validate_business_day_fields(
    const nlohmann::json& document,
    const std::string& id
) {
    for (const auto& row : document.at("products")) {
        const std::string& row_id = row.at("id").get_ref<const std::string&>();
        for (const auto& [name, value] : row.at("parameters").items()) {
            if (is_business_day_field(name) && !value.is_number_unsigned()) {
                reject(
                    "Product", id,
                    "row id '" + row_id + "': " + name
                        + " must be an unsigned business-day count."
                );
            }
            if (is_business_day_array_field(name)) {
                if (!value.is_array()) {
                    reject(
                        "Product", id,
                        "row id '" + row_id + "': " + name
                            + " must be an array of unsigned business-day "
                              "counts."
                    );
                }
                for (const auto& day : value) {
                    if (!day.is_number_unsigned()) {
                        reject(
                            "Product", id,
                            "row id '" + row_id + "': " + name
                                + " must contain only unsigned business-day "
                                  "counts."
                        );
                    }
                }
            }
        }
    }
}

// Validate one referenced model, curve, or product dataset.
void validate_dataset_reference(
    const nlohmann::json& document,
    const char* field,
    const std::string& price_id
) {
    if (!document.contains(field) || !document.at(field).is_object()) {
        reject("Price", price_id, std::string(field) + " must be an object.");
    }
    const auto& reference = document.at(field);
    require_string(reference, "id", "Price", price_id);
    require_string(reference, "catalog", "Price", price_id);
    require_url(reference, "url", "Price", price_id);
}

// Require one finite non-negative timing value.
void validate_duration(
    const nlohmann::json& timing,
    const char* field,
    const std::string& price_id
) {
    if (!timing.contains(field) || !timing.at(field).is_number()) {
        reject("Price", price_id, std::string(field) + " must be numeric.");
    }
    const double seconds = timing.at(field).get<double>();
    if (!std::isfinite(seconds) || seconds < 0.0) {
        reject(
            "Price", price_id,
            std::string(field) + " must be finite and non-negative."
        );
    }
}

// Parse a generated artifact before applying its family validator.
nlohmann::json read_generated_dataset(
    const std::filesystem::path& dataset_path
) {
    std::ifstream stream(dataset_path);
    if (!stream) {
        throw std::runtime_error(
            "Could not reopen generated JSON: " + dataset_path.string()
        );
    }
    nlohmann::json document;
    try {
        stream >> document;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Invalid generated JSON '" + dataset_path.string()
            + "': " + error.what()
        );
    }
    return document;
}

}  // namespace

// Validate the common envelope and rows of one model dataset.
void validate_model_dataset(const nlohmann::json& document) {
    validate_parameter_dataset(document, "Model", "model_family", "models");
}

// Validate the common envelope and rows of one curve dataset.
void validate_curve_dataset(const nlohmann::json& document) {
    validate_parameter_dataset(document, "Curve", "curve_family", "curves");
}

// Validate the common envelope and rows of one product dataset.
void validate_product_dataset(const nlohmann::json& document) {
    validate_parameter_dataset(
        document, "Product", "product_family", "products"
    );
    validate_time_convention(document, "Product", database_id(document));
    validate_business_day_fields(document, database_id(document));
}

// Validate one price dataset with an optional curve reference.
void validate_price_dataset(const nlohmann::json& document) {
    if (!document.is_object())
        reject("Price", "<unknown>", "the JSON root must be an object.");
    const std::string id = database_id(document);
    require_string(document, "database_id", "Price", id);
    require_string(document, "catalog", "Price", id);
    require_url(document, "url", "Price", id);
    validate_time_convention(document, "Price", id);
    validate_dataset_reference(document, "model_dataset", id);
    validate_dataset_reference(document, "product_dataset", id);
    const bool has_curve = document.contains("curve_dataset");
    if (has_curve) validate_dataset_reference(document, "curve_dataset", id);

    if (!document.contains("timing") || !document.at("timing").is_object())
        reject("Price", id, "timing must be an object.");
    const auto& timing = document.at("timing");
    validate_duration(timing, "wall_seconds", id);
    validate_duration(timing, "kernel_seconds", id);

    if (!document.contains("row_count")
        || !document.at("row_count").is_number_unsigned()) {
        reject("Price", id, "row_count must be a positive integer.");
    }
    if (!document.contains("results") || !document.at("results").is_array())
        reject("Price", id, "results must be an array.");
    const std::size_t row_count =
        document.at("row_count").get<std::size_t>();
    const auto& results = document.at("results");
    if (row_count == 0U || results.size() != row_count) {
        reject(
            "Price", id,
            "row_count must be positive and match results.size()."
        );
    }
    std::unordered_set<std::string> result_ids;
    for (const auto& row : results) {
        if (!row.is_object()) reject("Price", id, "results rows must be objects.");
        const std::string& row_id = require_string(row, "id", "Price", id);
        if (!result_ids.insert(row_id).second) {
            reject(
                "Price", id,
                "row id '" + row_id + "' must be unique."
            );
        }
        require_string(row, "model_id", "Price", id);
        require_string(row, "product_id", "Price", id);
        if (has_curve) require_string(row, "curve_id", "Price", id);
        if (!row.contains("outputs") || !row.at("outputs").is_object()
            || row.at("outputs").empty()) {
            reject(
                "Price", id,
                "row id '" + row_id + "': outputs must be a non-empty object."
            );
        }
        const auto& outputs = row.at("outputs");
        if (!outputs.contains("price") || !outputs.at("price").is_number()) {
            reject(
                "Price", id,
                "row id '" + row_id + "': outputs.price must be numeric."
            );
        }
        for (const auto& [name, value] : outputs.items()) {
            if (!value.is_number()
                || !std::isfinite(value.get<double>())) {
                reject(
                    "Price", id,
                    "row id '" + row_id + "': outputs." + name
                        + " must be numeric and finite."
                );
            }
        }
    }
}

// Reload and validate one generated model JSON artifact.
void validate_model_dataset_file(const std::filesystem::path& dataset_path) {
    validate_model_dataset(read_generated_dataset(dataset_path));
}

// Reload and validate one generated curve JSON artifact.
void validate_curve_dataset_file(const std::filesystem::path& dataset_path) {
    validate_curve_dataset(read_generated_dataset(dataset_path));
}

// Reload and validate one generated product JSON artifact.
void validate_product_dataset_file(const std::filesystem::path& dataset_path) {
    validate_product_dataset(read_generated_dataset(dataset_path));
}

// Reload and validate one generated price JSON artifact.
void validate_price_dataset_file(const std::filesystem::path& dataset_path) {
    validate_price_dataset(read_generated_dataset(dataset_path));
}

}  // namespace ai_factory::workbench::datasets
