#include "tools/datasets/multi_asset_price_dataset.hpp"

#include "common/dataset_validation.hpp"
#include "tools/datasets/artifact_io.hpp"

#include <cmath>
#include <map>
#include <stdexcept>

namespace ai_factory::workbench::datasets {

void write_multi_asset_price_dataset(
    const MultiAssetPriceDatasetRecipe& recipe,
    const std::vector<std::uint32_t>& scenario_model_indices,
    const std::vector<monte_carlo::MultiAssetPriceRow>& rows,
    const std::vector<float>& prices,
    const std::vector<float>& standard_errors,
    std::size_t paths_per_price,
    std::uint64_t base_seed,
    const nlohmann::ordered_json& execution,
    double wall_seconds,
    double kernel_seconds,
    const std::vector<std::uint32_t>& scenario_rate_indices
) {
    validate_dataset_url(recipe.url);
    const auto asset_count = recipe.model_dataset_paths.size();
    if (asset_count == 0U || scenario_model_indices.empty()
        || scenario_model_indices.size() % asset_count != 0U
        || rows.empty() || prices.size() != rows.size()
        || standard_errors.size() != rows.size()
        || paths_per_price < 2U) {
        throw std::invalid_argument("Invalid multi-asset price output dimensions.");
    }
    std::map<std::filesystem::path, nlohmann::ordered_json> model_cache;
    std::vector<const nlohmann::ordered_json*> models;
    models.reserve(asset_count);
    for (const auto& path : recipe.model_dataset_paths) {
        auto [entry, inserted] = model_cache.try_emplace(path);
        if (inserted) entry->second = read_json_file(path);
        models.push_back(&entry->second);
    }
    const auto product = read_json_file(recipe.product_dataset_path);
    const auto correlation = read_json_file(recipe.correlation_dataset_path);
    const auto curve = recipe.curve_dataset_path
        ? read_json_file(*recipe.curve_dataset_path)
        : nlohmann::ordered_json{};
    const auto scenario_count =
        scenario_model_indices.size() / asset_count;
    if (recipe.rate_model_dataset_path.has_value()
            != !scenario_rate_indices.empty()
        || (recipe.rate_model_dataset_path
            && scenario_rate_indices.size() != scenario_count)) {
        throw std::invalid_argument("Invalid multi-asset short-rate references.");
    }
    const auto rate_models = recipe.rate_model_dataset_path
        ? read_json_file(*recipe.rate_model_dataset_path)
        : nlohmann::ordered_json{};
    const auto reference = [](const nlohmann::ordered_json& document) {
        return nlohmann::ordered_json{
            {"id", document.at("database_id")},
            {"catalog", document.at("catalog")},
            {"url", document.at("url")}
        };
    };
    nlohmann::ordered_json model_references =
        nlohmann::ordered_json::array();
    for (const auto& model : models) {
        model_references.push_back(reference(*model));
    }
    nlohmann::ordered_json results = nlohmann::ordered_json::array();
    for (std::size_t i = 0U; i < rows.size(); ++i) {
        const auto& row = rows[i];
        if (row.scenario_index >= scenario_count
            || row.product_index >= product.at("products").size()
            || row.correlation_index >= correlation.at("correlations").size()
            || (recipe.rate_model_dataset_path
                && scenario_rate_indices[row.scenario_index]
                    >= rate_models.at("models").size())
            || (recipe.curve_dataset_path
                && row.curve_index >= curve.at("curves").size())
            || !std::isfinite(prices[i])
            || !std::isfinite(standard_errors[i])
            || standard_errors[i] < 0.0f) {
            throw std::invalid_argument("Invalid multi-asset result row.");
        }
        nlohmann::ordered_json ids = nlohmann::ordered_json::array();
        for (std::size_t asset = 0U; asset < asset_count; ++asset) {
            const auto index = scenario_model_indices[
                row.scenario_index * asset_count + asset
            ];
            ids.push_back(models[asset]->at("models")
                .at(index).at("id"));
        }
        nlohmann::ordered_json result{
            {"id", format_row_id(i)},
            {"model_ids", ids},
            {"correlation_id", correlation.at("correlations")
                .at(row.correlation_index).at("id")},
            {"product_id", product.at("products")
                .at(row.product_index).at("id")},
            {"seed", base_seed + i},
            {"outputs", {
                {"price", prices[i]},
                {"standard_error", standard_errors[i]}
            }}
        };
        if (recipe.rate_model_dataset_path) {
            result["rate_model_id"] = rate_models.at("models")
                .at(scenario_rate_indices[row.scenario_index]).at("id");
        }
        if (recipe.curve_dataset_path) {
            result["curve_id"] = curve.at("curves")
                .at(row.curve_index).at("id");
        }
        results.push_back(std::move(result));
    }
    nlohmann::ordered_json document{
        {"database_id", recipe.dataset_path.stem().string()},
        {"catalog", recipe.catalog_path.parent_path().generic_string()},
        {"url", recipe.url},
        {"row_count", rows.size()},
        {"time_convention", product.at("time_convention")},
        {"model_datasets", model_references},
        {"product_dataset", reference(product)},
        {"correlation_dataset", reference(correlation)},
        {"price_construction", {
            {"method", "explicit multi-asset scenario rows"},
            {"model_order", "model_datasets order"}
        }},
        {"timing", {
            {"wall_seconds", wall_seconds},
            {"kernel_seconds", kernel_seconds}
        }},
        {"results", results}
    };
    if (recipe.curve_dataset_path) {
        document["curve_dataset"] = reference(curve);
    }
    if (recipe.rate_model_dataset_path) {
        document["rate_model_dataset"] = reference(rate_models);
    }
    write_json_file(recipe.dataset_path, document);
    validate_price_dataset_file(recipe.dataset_path);
    auto receipt = execution;
    if (!receipt.is_object()) {
        throw std::invalid_argument("Execution metadata must be an object.");
    }
    receipt["paths_per_price"] = paths_per_price;
    write_generation_receipt(
        recipe.catalog_path, rows.size(), receipt,
        wall_seconds, kernel_seconds
    );
}

}  // namespace ai_factory::workbench::datasets
