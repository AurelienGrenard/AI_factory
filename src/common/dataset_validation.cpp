// Structural checks shared by every JSON dataset family.
#include "common/dataset_validation.hpp"

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
    if (document.contains("catalog")) require_string(document, "catalog", family, id);
    if (document.contains("url")) require_url(document, "url", family, id);
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
        || name == "payment_interval"
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
    const std::string& price_id,
    bool allow_missing_url = false
) {
    if (!document.contains(field) || !document.at(field).is_object()) {
        reject("Price", price_id, std::string(field) + " must be an object.");
    }
    const auto& reference = document.at(field);
    require_string(reference, "id", "Price", price_id);
    if (reference.contains("catalog"))
        require_string(reference, "catalog", "Price", price_id);
    if (reference.contains("url"))
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

nlohmann::json read_parameter_dataset(
    const std::filesystem::path& dataset_path,
    ParameterDatasetFamily family,
    const std::string& display_name
) {
    std::ifstream stream(dataset_path);
    if (!stream) {
        throw std::runtime_error(
            "Could not open " + display_name + " JSON: "
                + dataset_path.string()
        );
    }
    nlohmann::json document;
    try {
        stream >> document;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Invalid " + display_name + " JSON '" + dataset_path.string()
                + "': " + error.what()
        );
    }
    switch (family) {
        case ParameterDatasetFamily::Model:
            validate_model_dataset(document);
            if (document.value("parameterization", std::string("complete"))
                == "factorized") {
                for (auto& row : document.at("models")) {
                    auto& parameters = row.at("parameters");
                    // Legacy family loaders parse complete rows. These
                    // neutral values exist only in memory; the correlated
                    // scenario supplies the actual Brownian correlation.
                    parameters["risk_free_rate"] = 0.0f;
                    parameters["rho"] = 0.0f;
                    parameters["correlation"] = 0.0f;
                }
            }
            break;
        case ParameterDatasetFamily::Curve:
            validate_curve_dataset(document);
            break;
        case ParameterDatasetFamily::Product:
            validate_product_dataset(document);
            break;
    }
    return document;
}

// Validate the common envelope and rows of one model dataset.
void validate_model_dataset(const nlohmann::json& document) {
    validate_parameter_dataset(document, "Model", "model_family", "models");
    if (!document.contains("parameterization")) return;
    const std::string id = database_id(document);
    if (!document.at("parameterization").is_string())
        reject("Model", id, "parameterization must be a string.");
    const auto mode = document.at("parameterization").get<std::string>();
    if (mode != "complete" && mode != "factorized")
        reject("Model", id, "unsupported parameterization.");
    if (mode != "factorized") return;
    for (const auto& row : document.at("models")) {
        const auto& parameters = row.at("parameters");
        if (parameters.contains("risk_free_rate")
            || parameters.contains("rho")
            || parameters.contains("correlation"))
            reject("Model", id, "factorized row id '"
                + row.at("id").get<std::string>()
                + "' contains a rate or Brownian correlation.");
    }
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

// Validate compact single-asset prices. Technical provenance belongs to the
// generation receipt; row data contains identifiers and numeric estimates.
void validate_compact_price_dataset(const nlohmann::json& document) {
    const std::string id = database_id(document);
    require_string(document, "database_id", "Price", id);
    require_string(document, "model_family", "Price", id);
    require_string(document, "product_family", "Price", id);
    if (document.contains("catalog")) require_string(document, "catalog", "Price", id);
    validate_dataset_reference(document, "model_dataset", id, true);
    validate_dataset_reference(document, "product_dataset", id, true);
    const bool has_curve = document.contains("curve_dataset");
    if (has_curve) {
        require_string(document, "curve_family", "Price", id);
        validate_dataset_reference(document, "curve_dataset", id, true);
    }
    if (!document.contains("row_count")
        || !document.at("row_count").is_number_unsigned()
        || !document.contains("results")
        || !document.at("results").is_array()) {
        reject("Price", id, "row_count and results must describe the rows.");
    }
    const auto& results = document.at("results");
    if (results.empty()
        || results.size() != document.at("row_count").get<std::size_t>()) {
        reject("Price", id, "row_count must match results.size().");
    }
    for (const auto& row : results) {
        if (!row.is_object()) reject("Price", id, "results rows must be objects.");
        require_string(row, "model_id", "Price", id);
        require_string(row, "product_id", "Price", id);
        if (has_curve) require_string(row, "curve_id", "Price", id);
        for (const auto& [name, value] : row.items()) {
            if (name == "model_id" || name == "product_id" || name == "curve_id")
                continue;
            if (!value.is_number() || !std::isfinite(value.get<double>())) {
                reject("Price", id, "result " + name + " must be finite numeric.");
            }
        }
        if (!row.contains("price") || !row.at("price").is_number()) {
            reject("Price", id, "every result must contain a numeric price.");
        }
    }
}

// Validate one price dataset with an optional curve reference.
void validate_price_dataset(const nlohmann::json& document) {
    if (!document.is_object())
        reject("Price", "<unknown>", "the JSON root must be an object.");
    if (!document.contains("time_convention")) {
        validate_compact_price_dataset(document);
        return;
    }
    if (!document.is_object())
        reject("Price", "<unknown>", "the JSON root must be an object.");
    const std::string id = database_id(document);
    require_string(document, "database_id", "Price", id);
    if (document.contains("catalog")) require_string(document, "catalog", "Price", id);
    validate_time_convention(document, "Price", id);
    const bool multi_asset = document.contains("model_datasets");
    if (document.contains("url"))
        require_url(document, "url", "Price", id);
    if (multi_asset) {
        const auto& references = document.at("model_datasets");
        if (!references.is_array() || references.empty()) {
            reject("Price", id, "model_datasets must be a non-empty array.");
        }
        for (const auto& reference : references) {
            const nlohmann::json wrapper = {{"model_dataset", reference}};
            validate_dataset_reference(wrapper, "model_dataset", id, true);
        }
        validate_dataset_reference(document, "correlation_dataset", id, true);
    } else {
        validate_dataset_reference(document, "model_dataset", id);
    }
    validate_dataset_reference(document, "product_dataset", id, multi_asset);
    const bool has_curve = document.contains("curve_dataset");
    if (has_curve)
        validate_dataset_reference(document, "curve_dataset", id, multi_asset);
    const bool has_rate_model = document.contains("rate_model_dataset");
    if (has_rate_model) {
        if (!multi_asset)
            reject("Price", id, "rate_model_dataset requires model_datasets.");
        validate_dataset_reference(document, "rate_model_dataset", id, true);
    }

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
        if (multi_asset) {
            if (!row.contains("model_ids")
                || !row.at("model_ids").is_array()
                || row.at("model_ids").size()
                    != document.at("model_datasets").size()) {
                reject("Price", id, "model_ids must match model_datasets.");
            }
            for (const auto& model_id : row.at("model_ids")) {
                if (!model_id.is_string()
                    || model_id.get<std::string>().empty()) {
                    reject("Price", id, "model_ids must be non-empty strings.");
                }
            }
            require_string(row, "correlation_id", "Price", id);
        } else {
            require_string(row, "model_id", "Price", id);
        }
        require_string(row, "product_id", "Price", id);
        if (has_curve) require_string(row, "curve_id", "Price", id);
        if (has_rate_model) require_string(row, "rate_model_id", "Price", id);
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
