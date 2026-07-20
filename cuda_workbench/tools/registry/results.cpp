// Result row construction and Monte Carlo database writing implementation.
#include "tools/registry/results.hpp"

#include "tools/registry/io.hpp"

#include <limits>
#include <stdexcept>

namespace ai_factory::workbench::registry {
namespace {

struct ResultIndices {
    std::size_t model;
    std::size_t product;
};

ResultIndices result_indices(
    std::size_t result_index,
    std::size_t product_count,
    ResultConstruction construction
) {
    if (construction == ResultConstruction::Aligned) {
        return {result_index, result_index};
    }
    return {
        result_index / product_count,
        result_index % product_count,
    };
}

}  // namespace

std::size_t result_row_count(
    std::size_t model_count,
    std::size_t product_count,
    ResultConstruction construction
) {
    if (model_count == 0U || product_count == 0U) {
        throw std::invalid_argument(
            "Result construction requires non-empty model and product databases."
        );
    }
    if (construction == ResultConstruction::Aligned) {
        if (model_count != product_count) {
            throw std::invalid_argument(
                "Aligned construction requires equal model and product counts."
            );
        }
        return model_count;
    }
    if (model_count > std::numeric_limits<std::size_t>::max() / product_count) {
        throw std::overflow_error("Cartesian result count exceeds size_t.");
    }
    return model_count * product_count;
}

void write_monte_carlo_result_database(
    const std::filesystem::path& model_json_path,
    const std::filesystem::path& product_json_path,
    ResultConstruction construction,
    const std::vector<float>& prices,
    const std::vector<float>& standard_errors,
    const std::string& engine,
    const std::filesystem::path& output_root,
    const std::filesystem::path& generation_script,
    const std::vector<std::filesystem::path>& source_files,
    const std::string& numerical_method,
    std::size_t monte_carlo_paths_per_price,
    float target_dt,
    unsigned int threads_per_block,
    std::uint64_t first_seed,
    double wall_seconds,
    double kernel_seconds
) {
    const nlohmann::ordered_json model_document = read_json_file(model_json_path);
    const nlohmann::ordered_json product_document = read_json_file(product_json_path);
    const auto& model_rows = model_document.at("models");
    const auto& product_rows = product_document.at("products");
    const std::size_t row_count = result_row_count(
        model_rows.size(), product_rows.size(), construction
    );
    if (prices.size() != row_count || standard_errors.size() != row_count) {
        throw std::invalid_argument(
            "Result vectors must match the constructed input row count."
        );
    }

    const std::string model_database_id =
        model_document.at("database_id").get<std::string>();
    const std::string product_database_id =
        product_document.at("database_id").get<std::string>();
    const std::string database_id = generation_script.stem().string();
    if (database_id.empty()) {
        throw std::invalid_argument(
            "The generation script must have a non-empty basename."
        );
    }
    const std::filesystem::path json_path =
        output_root / "data" / (database_id + ".json");
    const std::filesystem::path yaml_path =
        output_root / "specifications" / (database_id + ".yaml");
    const std::string construction_rule =
        construction == ResultConstruction::Aligned
        ? "aligned row pairing"
        : "Cartesian product in model-major order";

    nlohmann::ordered_json rows = nlohmann::ordered_json::array();
    for (std::size_t index = 0; index < row_count; ++index) {
        const ResultIndices source = result_indices(
            index, product_rows.size(), construction
        );
        rows.push_back({
            {"id", format_row_id(index)},
            {"model_id", model_rows.at(source.model).at("id").get<std::string>()},
            {"product_id", product_rows.at(source.product).at("id").get<std::string>()},
            {"seed", first_seed + index},
            {"outputs", {
                {"price", prices[index]},
                {"standard_error", standard_errors[index]},
            }},
        });
    }

    const nlohmann::ordered_json model_database = {
        {"id", model_database_id},
        {"json_path", model_json_path.generic_string()},
    };
    const nlohmann::ordered_json product_database = {
        {"id", product_database_id},
        {"json_path", product_json_path.generic_string()},
    };
    const nlohmann::ordered_json timing = {
        {"wall_seconds", wall_seconds},
        {"kernel_seconds", kernel_seconds},
    };
    write_json_file(json_path, {
        {"database_id", database_id},
        {"specification", yaml_path.generic_string()},
        {"generation_script", generation_script.generic_string()},
        {"row_count", row_count},
        {"model_database", model_database},
        {"product_database", product_database},
        {"timing", timing},
        {"results", rows},
    });

    if (source_files.empty()) {
        throw std::invalid_argument("A result must reference its source files.");
    }
    nlohmann::ordered_json serialized_source_files =
        nlohmann::ordered_json::array();
    for (const std::filesystem::path& source_file : source_files) {
        serialized_source_files.push_back(source_file.generic_string());
    }

    const std::string device =
        engine.find("gpu") != std::string::npos ? "gpu" : "cpu";
    const std::string implementation = device == "gpu" ? "CUDA" : "C++";
    write_yaml_file(yaml_path, {
        {"title", model_database_id + " x " + product_database_id + " " + engine},
        {"database_id", database_id},
        {"json_path", json_path.generic_string()},
        {"generation_script", generation_script.generic_string()},
        {"summary", {
            {"row_count", row_count},
            {"monte_carlo_paths_per_price", monte_carlo_paths_per_price},
            {"model", model_document.at("model_family")},
            {"numerical_method", numerical_method},
            {"payoff", product_document.at("product_family")},
            {"implementation", implementation},
            {"device", device},
            {"threads_per_block", threads_per_block},
            {"source_files", serialized_source_files},
        }},
        {"time_grid", {
            {"rule", "nearest integer step count to target dt"},
            {"target_dt", target_dt},
            {"step_count", "round(maturity / target_dt)"},
            {"effective_dt", "maturity / step_count"},
        }},
        {"outputs", {
            {"price", {{"estimator", "Monte Carlo discounted payoff mean"}}},
            {"standard_error", {{"estimator", "Monte Carlo standard error of discounted payoff"}}},
        }},
        {"model_database", model_database},
        {"product_database", product_database},
        {"result_construction", {{"rule", construction_rule}}},
        {"timing", timing},
    });
}

}  // namespace ai_factory::workbench::registry
