// Check paired native artifacts and fail-before-write behavior without CUDA.
#include "tools/datasets/artifact_io.hpp"
#include "tools/datasets/price_delta_dataset.hpp"
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

int main() {
    using namespace ai_factory::workbench::datasets;
    char pattern[] = "/tmp/ai_factory_price_delta_stage_XXXXXX";
    const char* allocated = mkdtemp(pattern);
    if (!allocated) return 1;
    const std::filesystem::path directory(allocated);
    try {
        const auto require = [](bool condition) {
            if (!condition) throw std::runtime_error("Price-delta artifact contract violated");
        };
        nlohmann::ordered_json model{{"database_id", "models"}, {"catalog", "catalog/test"},
            {"url", "https://datasets.ai-factory.example/models.json"},
            {"models", {{{"id", "000001"}, {"parameters", {{"spot", 1.0f}}}}}}};
        auto product = model;
        product.erase("models");
        product["products"] = {{{"id", "000001"}}};
        product["time_convention"] = {{"unit", "business_day"}, {"days_per_year", 252}};
        write_json_file(directory / "model.json", model);
        write_json_file(directory / "product.json", product);
        PriceDeltaRecipe recipe{directory / "model.json", directory / "product.json",
            directory / "paired.json", directory / "generation.yaml",
            "https://datasets.ai-factory.example/paired.json", "source/generator.cpp", "centered_crn", .01, 2};
        PriceDeltaResults result;
        result.prices = {.1f}; result.price_errors = {.01f};
        result.deltas = {.5f}; result.delta_errors = {.02f};
        result.lower_spots = {.995f}; result.upper_spots = {1.005f};
        result.bump_widths = {result.upper_spots[0] - result.lower_spots[0]};
        result.execution = {{"paths_per_price", 1048576}, {"seed", 719}};
        write_price_delta_dataset(recipe, result);
        const auto published = read_json_file(recipe.dataset);
        require(published["validation"]["verified"] == false);
        require(published["sensitivity"]["relative_full_width"] == .01);
        require(published["time_grid"]["steps_per_year"] == 504);
        require(published["results"][0]["outputs"].contains("delta_standard_error"));
        result.deltas[0] = std::numeric_limits<float>::quiet_NaN();
        bool rejected = false;
        try { write_price_delta_dataset(recipe, result); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected && read_json_file(recipe.dataset) == published);
        result.deltas[0] = .5f;
        result.price_errors.clear(); result.delta_errors.clear();
        result.execution["paths_per_price"] = 0;
        recipe.delta_method = "centered_closed_form";
        recipe.simulation_steps_per_day = 0;
        write_price_delta_dataset(recipe, result);
        const auto closed = read_json_file(recipe.dataset);
        require(!closed["results"][0]["outputs"].contains("delta_standard_error"));
        require(!closed.contains("time_grid"));

        model["models"].push_back({{"id", "000002"}, {"parameters", {{"spot", 2.0f}}}});
        product["products"].push_back({{"id", "000002"}});
        write_json_file(directory / "model.json", model);
        write_json_file(directory / "product.json", product);
        recipe.construction = ai_factory::workbench::PriceConstruction::CartesianProduct;
        result.prices = {.1f, .2f, .3f, .4f};
        result.deltas = {.5f, .6f, .7f, .8f};
        result.lower_spots = {.995f, .995f, 1.99f, 1.99f};
        result.upper_spots = {1.005f, 1.005f, 2.01f, 2.01f};
        result.bump_widths.clear();
        for (std::size_t index = 0; index < result.lower_spots.size(); ++index)
            result.bump_widths.push_back(result.upper_spots[index] - result.lower_spots[index]);
        write_price_delta_dataset(recipe, result);
        const auto cartesian = read_json_file(recipe.dataset);
        require(cartesian["row_count"] == 4);
        require(cartesian["price_construction"]["method"] == "Cartesian product");
        require(cartesian["results"][0]["model_id"] == "000001");
        require(cartesian["results"][1]["product_id"] == "000002");
        require(cartesian["results"][2]["model_id"] == "000002");
        require(cartesian["results"][2]["product_id"] == "000001");
        std::filesystem::remove_all(directory);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        std::filesystem::remove_all(directory);
        return 1;
    }
}
