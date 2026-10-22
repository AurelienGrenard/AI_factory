// Price-result assembly and publication; CUDA execution lives elsewhere.
#include "tools/datasets/price_dataset.hpp"
#include "tools/datasets/artifact_io.hpp"

#include "common/result_index.cuh"

#include <cmath>
#include <optional>
#include <stdexcept>

namespace ai_factory::workbench::datasets {
namespace {
std::optional<std::string> option_side_from_path(
    const std::filesystem::path& dataset_path
) {
    const auto family = dataset_path.parent_path().filename().string();
    if (family.find("payer") != std::string::npos) return "payer";
    if (family.find("receiver") != std::string::npos) return "receiver";
    if (family.find("calls") != std::string::npos
        || family == "caplets") return "call";
    if (family.find("puts") != std::string::npos
        || family == "floorlets") return "put";
    return std::nullopt;
}

nlohmann::ordered_json dataset_reference(
    const nlohmann::ordered_json& document
) {
    return {{"id", document.at("database_id")}};
}

void add_option_side(
    nlohmann::ordered_json& document,
    const std::filesystem::path& dataset_path
) {
    if (const auto side = option_side_from_path(dataset_path)) {
        document["option_side"] = *side;
    }
}

void write_analytical_price_dataset_impl(
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
) {
    validate_dataset_url(url);
    const nlohmann::ordered_json model_document =
        read_json_file(model_dataset_path);
    const nlohmann::ordered_json curve_document =
        read_json_file(curve_dataset_path);
    const nlohmann::ordered_json product_document =
        read_json_file(product_dataset_path);
    const auto& model_rows = model_document.at("models");
    const auto& curve_rows = curve_document.at("curves");
    const auto& product_rows = product_document.at("products");
    const std::size_t row_count = price_row_count(
        model_rows.size(),
        curve_rows.size(),
        product_rows.size(),
        construction
    );
    if (prices.size() != row_count) {
        throw std::invalid_argument(
            "Prices must match the constructed input row count."
        );
    }
    for (std::size_t index = 0U; index < row_count; ++index) {
        if (!std::isfinite(prices[index])) {
            throw std::runtime_error(
                "Price row id '" + format_row_id(index)
                + "': price must be finite."
            );
        }
    }

    const std::string database_id = dataset_path.stem().string();
    if (database_id.empty()) {
        throw std::invalid_argument(
            "The price dataset must have a non-empty basename."
        );
    }

    // Reference source identifiers without duplicating their parameters.
    nlohmann::ordered_json rows = nlohmann::ordered_json::array();
    for (std::size_t index = 0U; index < row_count; ++index) {
        const auto indices = decode_model_curve_product_result_index(
            index, curve_rows.size(), product_rows.size(), construction
        );
        const auto model_index = indices.model_index;
        const auto curve_index = indices.curve_index;
        const auto product_index = indices.product_index;
        rows.push_back({
            {"model_id", model_rows.at(model_index).at("id")},
            {"product_id", product_rows.at(product_index).at("id")},
            {"curve_id", curve_rows.at(curve_index).at("id")},
            {"price", prices[index]},
        });
    }

    nlohmann::ordered_json json_document = {
        {"database_id", database_id},
        {"model_family", model_document.at("model_family")},
        {"product_family", product_document.at("product_family")},
        {"curve_family", curve_document.at("curve_family")},
    };
    if (product_document.contains("quote_convention")) {
        json_document["quote_convention"] = product_document.at("quote_convention");
    }
    add_option_side(json_document, dataset_path);
    json_document["row_count"] = row_count;
    json_document["model_dataset"] = dataset_reference(model_document);
    json_document["curve_dataset"] = dataset_reference(curve_document);
    json_document["product_dataset"] = dataset_reference(product_document);
    json_document["results"] = std::move(rows);
    write_json_file(dataset_path, json_document);

    if (!cuda_execution.is_object() || cuda_execution.empty()) {
        throw std::invalid_argument(
            "An analytical CUDA result requires execution metadata."
        );
    }
    nlohmann::ordered_json receipt_execution = cuda_execution;
    receipt_execution["paths_per_price"] = 0U;
    write_generation_receipt(
        catalog_path, row_count, receipt_execution, wall_seconds, kernel_seconds
    );
}

void write_analytical_price_dataset_impl(
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
) {
    validate_dataset_url(url);
    const nlohmann::ordered_json model_document =
        read_json_file(model_dataset_path);
    const nlohmann::ordered_json product_document =
        read_json_file(product_dataset_path);
    const auto& model_rows = model_document.at("models");
    const auto& product_rows = product_document.at("products");
    const std::size_t row_count = price_row_count(
        model_rows.size(), product_rows.size(), construction
    );
    if (prices.size() != row_count) {
        throw std::invalid_argument(
            "Prices must match the constructed input row count."
        );
    }
    for (std::size_t index = 0U; index < row_count; ++index) {
        if (!std::isfinite(prices[index])) {
            throw std::runtime_error(
                "Price row id '" + format_row_id(index)
                + "': price must be finite."
            );
        }
    }

    const std::string database_id = dataset_path.stem().string();
    if (database_id.empty()) {
        throw std::invalid_argument(
            "The price dataset must have a non-empty basename."
        );
    }
    nlohmann::ordered_json rows = nlohmann::ordered_json::array();
    for (std::size_t index = 0U; index < row_count; ++index) {
        const std::size_t model_index =
            is_cartesian(construction)
            ? index / product_rows.size()
            : index;
        const std::size_t product_index =
            is_cartesian(construction)
            ? index % product_rows.size()
            : index;
        rows.push_back({
            {"model_id", model_rows.at(model_index).at("id")},
            {"product_id", product_rows.at(product_index).at("id")},
            {"price", prices[index]},
        });
    }

    nlohmann::ordered_json json_document = {
        {"database_id", database_id},
        {"model_family", model_document.at("model_family")},
        {"product_family", product_document.at("product_family")},
    };
    if (product_document.contains("quote_convention")) {
        json_document["quote_convention"] = product_document.at("quote_convention");
    }
    add_option_side(json_document, dataset_path);
    json_document["row_count"] = row_count;
    json_document["model_dataset"] = dataset_reference(model_document);
    json_document["product_dataset"] = dataset_reference(product_document);
    json_document["results"] = std::move(rows);
    write_json_file(dataset_path, json_document);

    if (!cuda_execution.is_object() || cuda_execution.empty()) {
        throw std::invalid_argument(
            "An analytical CUDA result requires execution metadata."
        );
    }
    nlohmann::ordered_json receipt_execution = cuda_execution;
    receipt_execution["paths_per_price"] = 0U;
    write_generation_receipt(
        catalog_path, row_count, receipt_execution, wall_seconds, kernel_seconds
    );
}

void write_monte_carlo_price_dataset_impl(
    const std::filesystem::path& model_dataset_path,
    const std::filesystem::path* curve_dataset_path,
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
) {
    validate_dataset_url(url);
    if (!catalog_sections.is_object() || catalog_sections.contains("validation")) {
        throw std::invalid_argument(
            "Additional catalog sections must form an object without validation metadata."
        );
    }
    const nlohmann::ordered_json model_document =
        read_json_file(model_dataset_path);
    const nlohmann::ordered_json curve_document = curve_dataset_path == nullptr
        ? nlohmann::ordered_json{}
        : read_json_file(*curve_dataset_path);
    const nlohmann::ordered_json product_document =
        read_json_file(product_dataset_path);
    const auto& model_rows = model_document.at("models");
    const auto& product_rows = product_document.at("products");
    const bool has_curve = curve_dataset_path != nullptr;
    const std::size_t row_count = has_curve
        ? price_row_count(
            model_rows.size(), curve_document.at("curves").size(),
            product_rows.size(), construction
        )
        : price_row_count(
            model_rows.size(), product_rows.size(), construction
        );
    if (prices.size() != row_count || standard_errors.size() != row_count) {
        throw std::invalid_argument(
            "Price vectors must match the constructed input row count."
        );
    }
    for (std::size_t index = 0U; index < row_count; ++index) {
        if (!std::isfinite(prices[index])
            || !std::isfinite(standard_errors[index])
            || standard_errors[index] < 0.0f) {
            throw std::runtime_error(
                "Price row id '" + format_row_id(index)
                + "': price must be finite and standard_error must be "
                "finite and non-negative."
            );
        }
    }

    const std::string database_id = dataset_path.stem().string();
    if (database_id.empty()) {
        throw std::invalid_argument(
            "The price dataset must have a non-empty basename."
        );
    }
    // Keep only input identifiers and pricing results in the JSON. The
    // sampling plan, seed, methodology and measured time live in YAML.
    nlohmann::ordered_json rows = nlohmann::ordered_json::array();
    for (std::size_t index = 0; index < row_count; ++index) {
        std::size_t model_index = index;
        std::size_t curve_index = index;
        std::size_t product_index = index;
        if (has_curve) {
            const auto source = decode_model_curve_product_result_index(
                index, curve_document.at("curves").size(),
                product_rows.size(), construction
            );
            model_index = source.model_index;
            curve_index = source.curve_index;
            product_index = source.product_index;
        } else {
            const auto source = decode_model_product_result_index(
                index, product_rows.size(), construction
            );
            model_index = source.model_index;
            product_index = source.product_index;
        }
        nlohmann::ordered_json row = {
            {"model_id", model_rows.at(model_index).at("id")},
            {"product_id", product_rows.at(product_index).at("id")},
        };
        if (has_curve) {
            row["curve_id"] = curve_document.at("curves")
                .at(curve_index).at("id");
        }
        row["price"] = prices[index];
        row["standard_error"] = standard_errors[index];
        rows.push_back(std::move(row));
    }

    nlohmann::ordered_json json_document = {
        {"database_id", database_id},
        {"model_family", model_document.at("model_family")},
        {"product_family", product_document.at("product_family")},
    };
    if (has_curve) {
        json_document["curve_family"] = curve_document.at("curve_family");
    }
    if (product_document.contains("quote_convention")) {
        json_document["quote_convention"] = product_document.at("quote_convention");
    }
    add_option_side(json_document, dataset_path);
    json_document["row_count"] = row_count;
    json_document["model_dataset"] = dataset_reference(model_document);
    if (has_curve) {
        json_document["curve_dataset"] = dataset_reference(curve_document);
    }
    json_document["product_dataset"] = dataset_reference(product_document);
    json_document["results"] = std::move(rows);
    write_json_file(dataset_path, json_document);

    if (!cuda_execution.is_object() || cuda_execution.empty()) {
        throw std::invalid_argument(
            "A Monte Carlo result requires CUDA execution metadata."
        );
    }
    nlohmann::ordered_json receipt_execution = cuda_execution;
    receipt_execution["paths_per_price"] = monte_carlo_paths_per_price;
    write_generation_receipt(
        catalog_path, row_count, receipt_execution, wall_seconds, kernel_seconds
    );
}
}  // namespace

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
) {
    write_monte_carlo_price_dataset_impl(
        model_dataset_path,
        nullptr,
        product_dataset_path,
        construction,
        prices,
        standard_errors,
        random_generator,
        dataset_path,
        catalog_path,
        url,
        numerical_method,
        monte_carlo_paths_per_price,
        delta_t,
        cuda_execution,
        catalog_sections,
        first_seed,
        wall_seconds,
        kernel_seconds
    );
}

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
) {
    write_monte_carlo_price_dataset_impl(
        model_dataset_path,
        &curve_dataset_path,
        product_dataset_path,
        construction,
        prices,
        standard_errors,
        random_generator,
        dataset_path,
        catalog_path,
        url,
        numerical_method,
        monte_carlo_paths_per_price,
        delta_t,
        cuda_execution,
        catalog_sections,
        first_seed,
        wall_seconds,
        kernel_seconds
    );
}

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
) {
    write_analytical_price_dataset_impl(
        model_dataset_path,
        product_dataset_path,
        construction,
        prices,
        dataset_path,
        catalog_path,
        url,
        numerical_method,
        cuda_execution,
        wall_seconds,
        kernel_seconds
    );
}

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
) {
    write_analytical_price_dataset_impl(
        model_dataset_path,
        curve_dataset_path,
        product_dataset_path,
        construction,
        prices,
        dataset_path,
        catalog_path,
        url,
        numerical_method,
        cuda_execution,
        wall_seconds,
        kernel_seconds
    );
}
}  // namespace ai_factory::workbench::datasets
