// Validate paired outputs and write native JSON/YAML for staged publication.
#include "tools/datasets/price_delta_dataset.hpp"
#include "tools/datasets/artifact_io.hpp"
#include "common/result_index.cuh"
#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::datasets {

void write_price_delta_dataset(const PriceDeltaRecipe& recipe, const PriceDeltaResults& result) {
    validate_dataset_url(recipe.url);
    const auto models = read_json_file(recipe.model_input);
    const auto products = read_json_file(recipe.product_input);
    const auto& model_rows = models.at("models");
    const auto& product_rows = products.at("products");
    const auto count = price_row_count(
        model_rows.size(), product_rows.size(), recipe.construction);
    const bool stochastic = result.execution.at("paths_per_price").get<std::size_t>() != 0;
    if (result.prices.size() != count || result.deltas.size() != count || result.lower_spots.size() != count
        || result.upper_spots.size() != count || result.bump_widths.size() != count
        || result.price_errors.size() != (stochastic ? count : 0)
        || result.delta_errors.size() != (stochastic ? count : 0))
        throw std::invalid_argument("Price-delta requires complete constructed output arrays.");
    if (!std::isfinite(recipe.relative_bump_width) || !(recipe.relative_bump_width > 0)
        || !(recipe.relative_bump_width < 2))
        throw std::invalid_argument("Invalid price-delta bump width.");
    if (!std::isfinite(result.wall_seconds) || !std::isfinite(result.kernel_seconds)
        || result.wall_seconds < 0 || result.kernel_seconds < 0)
        throw std::invalid_argument("Invalid price-delta timings.");
    nlohmann::ordered_json rows = nlohmann::ordered_json::array();
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(result.prices[i]) || !std::isfinite(result.deltas[i])
            || !std::isfinite(result.lower_spots[i]) || !(result.lower_spots[i] > 0)
            || !std::isfinite(result.upper_spots[i]) || !(result.bump_widths[i] > 0)
            || result.upper_spots[i] - result.lower_spots[i] != result.bump_widths[i])
            throw std::runtime_error("Invalid price-delta row " + format_row_id(i)
                + ": price=" + std::to_string(result.prices[i]) + ", delta=" + std::to_string(result.deltas[i])
                + ", lower=" + std::to_string(result.lower_spots[i])
                + ", upper=" + std::to_string(result.upper_spots[i])
                + ", width=" + std::to_string(result.bump_widths[i]));
        nlohmann::ordered_json outputs{{"price", result.prices[i]}, {"delta", result.deltas[i]}};
        if (stochastic) {
            if (!std::isfinite(result.price_errors[i]) || !std::isfinite(result.delta_errors[i])
                || result.price_errors[i] < 0 || result.delta_errors[i] < 0)
                throw std::runtime_error("Invalid paired errors at row " + format_row_id(i));
            outputs["standard_error"] = result.price_errors[i];
            outputs["delta_standard_error"] = result.delta_errors[i];
        }
        const auto indices = decode_model_product_result_index(
            i, product_rows.size(), recipe.construction);
        rows.push_back({{"id", format_row_id(i)},
            {"model_id", model_rows.at(indices.model_index).at("id")},
            {"product_id", product_rows.at(indices.product_index).at("id")},
            {"spot_bump", {{"lower", result.lower_spots[i]}, {"upper", result.upper_spots[i]},
                           {"represented_width", result.bump_widths[i]}}}, {"outputs", outputs}});
    }
    const auto reference = [](const auto& input) {
        return nlohmann::ordered_json{{"id", input.at("database_id")},
            {"catalog", input.at("catalog")}, {"url", input.at("url")}};
    };
    const nlohmann::ordered_json sensitivity{
        {"parameter", "spot"}, {"method", recipe.delta_method},
        {"relative_full_width", recipe.relative_bump_width},
        {"denominator", "per-row represented upper minus lower"},
        {"source_price_recipe", recipe.source_price_recipe},
        {"qualification", "implementation tested on bounded cases; no global bias certification"},
        {"standard_error_scope", stochastic ? "paired sampling error; excludes bump and stopping-policy bias" : "not applicable"}};
    nlohmann::ordered_json price_construction{{"method", "Aligned"}};
    if (is_cartesian(recipe.construction)) {
        price_construction = {
            {"method", "Cartesian product"},
            {"order", "model, product"},
        };
    }
    nlohmann::ordered_json catalog{
        {"title", recipe.dataset.stem().string()}, {"database_id", recipe.dataset.stem().string()},
        {"catalog", recipe.catalog.parent_path().generic_string()}, {"url", recipe.url},
        {"row_count", count}, {"time_convention", products.at("time_convention")},
        {"model_dataset", reference(models)}, {"product_dataset", reference(products)},
        {"price_construction", price_construction},
        {"sensitivity", sensitivity}, {"summary", result.execution},
        {"validation", {{"status", "pending"}, {"verified", false}}},
        {"outputs", {{"price", "central price"}, {"delta", "centered S0 finite difference"}}},
        {"timing", {{"wall_seconds", result.wall_seconds}, {"kernel_seconds", result.kernel_seconds}}}};
    if (stochastic) {
        catalog["outputs"]["standard_error"] = "central price sampling error";
        catalog["outputs"]["delta_standard_error"] = "paired delta sampling error";
    }
    if (recipe.simulation_steps_per_day) {
        const auto steps = 252U * recipe.simulation_steps_per_day;
        catalog["time_grid"] = {{"simulation_steps_per_day", recipe.simulation_steps_per_day},
            {"steps_per_year", steps}, {"delta_t", "1 / " + std::to_string(steps)}};
    }
    auto document = catalog;
    document.erase("title");
    document.erase("outputs");
    document["results"] = std::move(rows);
    // All outputs are checked before either artifact is written. Atomic pair
    // publication remains the campaign controller's responsibility.
    write_json_file(recipe.dataset, document);
    write_catalog_yaml(recipe.catalog, catalog);
}

}  // namespace ai_factory::workbench::datasets
