// Price-result assembly and publication; CUDA execution lives elsewhere.
#include "tools/datasets/price_dataset.hpp"
#include "tools/datasets/artifact_io.hpp"

#include "common/result_index.cuh"

#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::datasets {
namespace {
void add_fixed_time_grid(
    nlohmann::ordered_json& catalog,
    const nlohmann::ordered_json& cuda_execution,
    const nlohmann::ordered_json& product_document,
    const std::string& delta_t = ""
) {
    if (!cuda_execution.contains("simulation_steps_per_day")) return;
    const std::uint32_t simulation_steps_per_day =
        cuda_execution.at("simulation_steps_per_day").get<std::uint32_t>();
    if (simulation_steps_per_day == 0U) {
        throw std::invalid_argument(
            "simulation_steps_per_day must be positive."
        );
    }
    const std::uint32_t days_per_year = product_document
        .at("time_convention")
        .at("days_per_year")
        .get<std::uint32_t>();
    const std::uint32_t steps_per_year =
        simulation_steps_per_day * days_per_year;
    catalog["time_grid"] = {
        {"simulation_steps_per_day", simulation_steps_per_day},
        {"steps_per_year", steps_per_year},
        {
            "delta_t",
            delta_t.empty()
                ? "1 / " + std::to_string(steps_per_year)
                : delta_t
        },
    };
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
        std::size_t model_index = index;
        std::size_t curve_index = index;
        std::size_t product_index = index;
        if (is_cartesian(construction)) {
            const std::size_t curve_product_count =
                curve_rows.size() * product_rows.size();
            model_index = index / curve_product_count;
            const std::size_t remainder = index % curve_product_count;
            curve_index = remainder / product_rows.size();
            product_index = remainder % product_rows.size();
        }
        rows.push_back({
            {"id", format_row_id(index)},
            {"model_id", model_rows.at(model_index).at("id")},
            {"curve_id", curve_rows.at(curve_index).at("id")},
            {"product_id", product_rows.at(product_index).at("id")},
            {"outputs", {{"price", prices[index]}}},
        });
    }

    const auto dataset_reference = [](
        const nlohmann::ordered_json& document
    ) {
        return nlohmann::ordered_json{
            {"id", document.at("database_id")},
            {"catalog", document.at("catalog")},
            {"url", document.at("url")},
        };
    };
    const nlohmann::ordered_json json_document = {
        {"database_id", database_id},
        {"catalog", catalog_path.parent_path().generic_string()},
        {"url", url},
        {"row_count", row_count},
        {"time_convention", product_document.at("time_convention")},
        {"model_dataset", dataset_reference(model_document)},
        {"curve_dataset", dataset_reference(curve_document)},
        {"product_dataset", dataset_reference(product_document)},
        {"timing", {
            {"wall_seconds", wall_seconds},
            {"kernel_seconds", kernel_seconds},
        }},
        {"results", rows},
    };
    write_json_file(dataset_path, json_document);

    if (!cuda_execution.is_object() || cuda_execution.empty()) {
        throw std::invalid_argument(
            "An analytical CUDA result requires execution metadata."
        );
    }
    nlohmann::ordered_json summary = {
        {"pricing_method", "Closed-form"},
        {"model", model_document.at("model_family")},
        {"curve", curve_document.at("curve_family")},
        {"numerical_method", numerical_method},
        {"payoff", product_document.at("product_family")},
        {"implementation", "CUDA"},
        {"device", "gpu"},
    };
    for (const auto& [name, value] : cuda_execution.items()) {
        if (name == "simulation_steps_per_day") continue;
        summary[name] = value;
    }
    nlohmann::ordered_json price_construction = {{"method", "Aligned"}};
    if (is_cartesian(construction)) {
        price_construction = {
            {"method", "Cartesian product"},
            {"order", "model, curve, product"},
        };
    }
    nlohmann::ordered_json catalog = {
        {"title", database_id},
        {"database_id", database_id},
        {"catalog", catalog_path.parent_path().generic_string()},
        {"url", url},
        {"row_count", row_count},
        {"time_convention", product_document.at("time_convention")},
        {"summary", summary},
        {"validation", price_validation_metadata(catalog_path.parent_path())},
        {"outputs", {{"price", {{"estimator", "closed-form price"}}}}},
        {"model_dataset", dataset_reference(model_document)},
        {"curve_dataset", dataset_reference(curve_document)},
        {"product_dataset", dataset_reference(product_document)},
        {"price_construction", price_construction},
        {"timing", {
            {"wall_seconds", format_duration(wall_seconds)},
            {"kernel_seconds", format_duration(kernel_seconds)},
        }},
    };
    add_fixed_time_grid(catalog, cuda_execution, product_document);
    write_catalog_yaml(catalog_path, catalog);
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
    const auto dataset_reference = [](
        const nlohmann::ordered_json& document
    ) {
        return nlohmann::ordered_json{
            {"id", document.at("database_id")},
            {"catalog", document.at("catalog")},
            {"url", document.at("url")},
        };
    };

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
            {"id", format_row_id(index)},
            {"model_id", model_rows.at(model_index).at("id")},
            {"product_id", product_rows.at(product_index).at("id")},
            {"outputs", {{"price", prices[index]}}},
        });
    }

    write_json_file(dataset_path, {
        {"database_id", database_id},
        {"catalog", catalog_path.parent_path().generic_string()},
        {"url", url},
        {"row_count", row_count},
        {"time_convention", product_document.at("time_convention")},
        {"model_dataset", dataset_reference(model_document)},
        {"product_dataset", dataset_reference(product_document)},
        {"timing", {
            {"wall_seconds", wall_seconds},
            {"kernel_seconds", kernel_seconds},
        }},
        {"results", rows},
    });

    if (!cuda_execution.is_object() || cuda_execution.empty()) {
        throw std::invalid_argument(
            "An analytical CUDA result requires execution metadata."
        );
    }
    nlohmann::ordered_json summary = {
        {"pricing_method", "Closed-form"},
        {"model", model_document.at("model_family")},
        {"numerical_method", numerical_method},
        {"payoff", product_document.at("product_family")},
        {"implementation", "CUDA"},
        {"device", "gpu"},
    };
    for (const auto& [name, value] : cuda_execution.items()) {
        if (name == "simulation_steps_per_day") continue;
        summary[name] = value;
    }
    nlohmann::ordered_json price_construction = {{"method", "Aligned"}};
    if (is_cartesian(construction)) {
        price_construction = {
            {"method", "Cartesian product"},
            {"order", "model, product"},
        };
    }
    nlohmann::ordered_json catalog = {
        {"title", database_id},
        {"database_id", database_id},
        {"catalog", catalog_path.parent_path().generic_string()},
        {"url", url},
        {"row_count", row_count},
        {"time_convention", product_document.at("time_convention")},
        {"summary", summary},
        {"validation", price_validation_metadata(catalog_path.parent_path())},
        {"outputs", {{"price", {{"estimator", "closed-form price"}}}}},
        {"model_dataset", dataset_reference(model_document)},
        {"product_dataset", dataset_reference(product_document)},
        {"price_construction", price_construction},
        {"timing", {
            {"wall_seconds", format_duration(wall_seconds)},
            {"kernel_seconds", format_duration(kernel_seconds)},
        }},
    };
    add_fixed_time_grid(catalog, cuda_execution, product_document);
    write_catalog_yaml(catalog_path, catalog);
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
    nlohmann::ordered_json price_construction = {{"method", "Aligned"}};
    if (is_cartesian(construction)) {
        price_construction = {
            {"method", "Cartesian product"},
            {"order", has_curve ? "model, curve, product" : "model, product"},
        };
    }

    // Reference source rows instead of duplicating their parameters.
    nlohmann::ordered_json rows = nlohmann::ordered_json::array();
    for (std::size_t index = 0; index < row_count; ++index) {
        std::size_t model_index = index;
        std::size_t curve_index = index;
        std::size_t product_index = index;
        if (is_cartesian(construction)) {
            if (has_curve) {
                const std::size_t curve_product_count =
                    curve_document.at("curves").size() * product_rows.size();
                model_index = index / curve_product_count;
                const std::size_t remainder = index % curve_product_count;
                curve_index = remainder / product_rows.size();
                product_index = remainder % product_rows.size();
            } else {
                const ModelProductIndices source =
                    decode_model_product_result_index(
                    index, product_rows.size(), construction
                );
                model_index = source.model_index;
                product_index = source.product_index;
            }
        }
        nlohmann::ordered_json row = {
            {"id", format_row_id(index)},
            {"model_id", model_rows.at(model_index).at("id").get<std::string>()},
            {
                "product_id",
                product_rows.at(product_index).at("id").get<std::string>()
            },
            {"seed", first_seed + index},
        };
        if (has_curve) {
            row["curve_id"] = curve_document.at("curves")
                .at(curve_index).at("id").get<std::string>();
        }
        row["outputs"] = {
            {"price", prices[index]},
            {"standard_error", standard_errors[index]},
        };
        rows.push_back(std::move(row));
    }

    const auto dataset_reference = [](
        const nlohmann::ordered_json& document
    ) {
        return nlohmann::ordered_json{
            {"id", document.at("database_id")},
            {"catalog", document.at("catalog")},
            {"url", document.at("url")},
        };
    };
    nlohmann::ordered_json json_document = {
        {"database_id", database_id},
        {"catalog", catalog_path.parent_path().generic_string()},
        {"url", url},
        {"row_count", row_count},
        {"time_convention", product_document.at("time_convention")},
        {"model_dataset", dataset_reference(model_document)},
        {"product_dataset", dataset_reference(product_document)},
        {"timing", {
            {"wall_seconds", wall_seconds},
            {"kernel_seconds", kernel_seconds},
        }},
        {"results", rows},
    };
    if (has_curve) {
        json_document["curve_dataset"] = dataset_reference(curve_document);
    }
    write_json_file(dataset_path, json_document);

    if (!cuda_execution.is_object() || cuda_execution.empty()) {
        throw std::invalid_argument(
            "A Monte Carlo result requires CUDA execution metadata."
        );
    }
    nlohmann::ordered_json summary = {
        {"pricing_method", "Monte Carlo"},
        {"monte_carlo_paths_per_price", monte_carlo_paths_per_price},
        {"model", model_document.at("model_family")},
        {"numerical_method", numerical_method},
        {"payoff", product_document.at("product_family")},
        {"implementation", "CUDA"},
        {"device", "gpu"},
    };
    if (has_curve) summary["curve"] = curve_document.at("curve_family");
    for (const auto& [name, value] : cuda_execution.items()) {
        if (name == "simulation_steps_per_day") continue;
        summary[name] = value;
    }
    summary["random_generator"] = random_generator;
    const nlohmann::ordered_json yaml_timing = {
        {"wall_seconds", format_duration(wall_seconds)},
        {"kernel_seconds", format_duration(kernel_seconds)},
    };
    const nlohmann::ordered_json time_grid = {
            {"rule", "nearest integer step count to target dt"},
            {"target_dt", delta_t},
            {"step_count", "round(maturity / target_dt)"},
            {"effective_dt", "maturity / step_count"},
        };
    nlohmann::ordered_json catalog = {
        {"title", database_id},
        {"database_id", database_id},
        {"catalog", catalog_path.parent_path().generic_string()},
        {"url", url},
        {"row_count", row_count},
        {"time_convention", product_document.at("time_convention")},
        {"summary", summary},
        {"validation", price_validation_metadata(catalog_path.parent_path())},
        {"outputs", {
            {"price", {{"estimator", "Monte Carlo discounted payoff mean"}}},
            {
                "standard_error",
                {{"estimator", "Monte Carlo standard error of discounted payoff"}}
            },
        }},
        {"model_dataset", dataset_reference(model_document)},
        {"product_dataset", dataset_reference(product_document)},
        {"price_construction", price_construction},
        {"timing", yaml_timing},
    };
    if (has_curve) {
        catalog["curve_dataset"] = dataset_reference(curve_document);
    }
    if (!delta_t.empty()) catalog["time_grid"] = time_grid;
    if (!catalog_sections.is_object()) {
        throw std::invalid_argument(
            "Additional catalog sections must form an object."
        );
    }
    for (const auto& [name, value] : catalog_sections.items()) {
        catalog[name] = value;
    }
    add_fixed_time_grid(
        catalog, cuda_execution, product_document, delta_t
    );
    write_catalog_yaml(catalog_path, catalog);
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
