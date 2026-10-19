// Native gradient artifact checks include one-sided stencils and fail-before-write behavior.
#include "tools/datasets/price_gradients/dataset.hpp"
#include "tools/datasets/artifact_io.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace data = datasets::price_gradients;
    char pattern[] = "/tmp/ai_factory_gradients_artifact_XXXXXX";
    const auto* allocated = mkdtemp(pattern);
    if (!allocated) return 1;
    const std::filesystem::path directory(allocated);
    try {
        const auto require = [](bool value) { if (!value) throw std::runtime_error("Gradient artifact contract failed."); };
        nlohmann::ordered_json models{{"database_id","models"},{"model_family","Test model"},{"catalog","catalog/test"},
            {"url","https://datasets.ai-factory.example/models.json"}, {"models",{{{"id","000001"}}}}};
        auto products = models;
        products.erase("models"); products["products"] = {{{"id","000001"}}};
        products["time_convention"] = {{"unit","business_day"},{"days_per_year",252}};
        products["product_family"] = "Test product";
        datasets::write_json_file(directory/"models.json",models);
        datasets::write_json_file(directory/"products.json",products);
        data::Recipe recipe{directory/"models.json",directory/"products.json",directory/"gradients.json",directory/"generation.yaml",
            "source/generator.cpp",PriceConstruction::Aligned,
            {{{"model.rho",{.125,pg::BumpScale::absolute}}}}, {}};
        data::Results result{{.1f},{.01f},{.2f},{.02f},
            {pg::prepare_stencil(1.f,{.125,pg::BumpScale::absolute},[](float p){return p>=-1 && p<=1;})},
            {{"paths_per_price",32},{"sensitivity_count",1},{"scenario_count",3},{"seed",719}}};
        data::write_dataset(recipe,result);
        const auto valid = datasets::read_json_file(recipe.dataset);
        require(valid.size() == 8U && valid.contains("results"));
        require(valid["database_id"] == "gradients"
            && valid["model_family"] == "Test model"
            && valid["product_family"] == "Test product"
            && valid["catalog"] == directory.generic_string()
            && valid["row_count"] == 1U);
        require(valid["model_dataset"] == nlohmann::ordered_json({
            {"id", "models"}, {"catalog", "catalog/test"}
        }));
        require(valid["product_dataset"] == valid["model_dataset"]);
        require(valid["results"].size() == 1U);
        require(valid["results"][0]["model_id"] == "000001");
        require(valid["results"][0]["product_id"] == "000001");
        require(valid["results"][0]["price"] == .1f);
        require(valid["results"][0]["gradients"]["model.rho"] == .2f);
        require(valid["results"][0]["standard_error"] == .01f);
        require(valid["results"][0]["gradient_standard_errors"]["model.rho"] == .02f);
        require(!valid["results"][0].contains("stencils"));
        auto call_recipe = recipe;
        call_recipe.dataset = directory / "call_gradients.json";
        call_recipe.source_price_recipe =
            "catalog/model/test/prices/european_calls/test/recipe.yaml";
        call_recipe.option_side = "call";
        data::write_dataset(call_recipe, result);
        const auto call_document = datasets::read_json_file(
            call_recipe.dataset
        );
        require(call_document["product_family"] == "Test product"
            && call_document["option_side"] == "call");
        // Exact rate increments still need a monitored path for a bond barrier.
        auto barrier_recipe = recipe;
        barrier_recipe.dataset = directory / "barrier_gradients.json";
        barrier_recipe.catalog = directory / "barrier_generation.yaml";
        barrier_recipe.exact_transition = true;
        barrier_recipe.barrier_grid_monitoring = true;
        data::write_dataset(barrier_recipe, result);
        const auto barrier_document = datasets::read_json_file(
            barrier_recipe.dataset
        );
        require(barrier_document["results"] == valid["results"]);
        // A spot-relative product level executes at an absolute S0-scaled
        // strike, while the published stencil and gradient use ratio units.
        auto quote_recipe = recipe;
        quote_recipe.configuration.sensitivities[0].parameter =
            "product.strike";
        quote_recipe.configuration.sensitivities[0].bump.displacement = .01f;
        quote_recipe.dataset = directory / "quote_gradients.json";
        quote_recipe.catalog = directory / "quote_generation.yaml";
        auto quote_result = result;
        auto& quote_stencil = quote_result.stencils[0];
        quote_stencil.central = 2.0f;
        quote_stencil.first = 1.98f;
        quote_stencil.second = 2.02f;
        quote_stencil.displacement = .01f;
        quote_stencil.represented_width =
            (quote_stencil.second - quote_stencil.first) / 2.0f;
        quote_stencil.kind = pg::StencilKind::centered;
        quote_stencil.first_weight = 0.0f;
        quote_stencil.second_weight = 0.0f;
        data::write_dataset(quote_recipe, quote_result);
        const auto quote_document = datasets::read_json_file(
            quote_recipe.dataset
        );
        require(quote_document["results"][0]["gradients"]["product.strike"] == .2f);
        require(!quote_document["results"][0].contains("stencils"));
        // A small absolute bump at nonunit S0 must survive conversion to
        // ratio units without rounding both endpoints to FP32 again.
        quote_stencil.central = 1.53f;
        quote_stencil.first = 1.53f - 1e-4f;
        quote_stencil.second = 1.53f + 1e-4f;
        quote_stencil.represented_width =
            (quote_stencil.second - quote_stencil.first) / 1.7f;
        data::write_dataset(quote_recipe, quote_result);
        require(datasets::read_json_file(quote_recipe.dataset)["results"][0]["price"] == .1f);
        quote_stencil.central = 2.0f;
        quote_stencil.first = 1.98f;
        quote_stencil.second = 1.96f;
        quote_stencil.represented_width =
            (quote_stencil.second - quote_stencil.first) / 2.0f;
        quote_stencil.kind = pg::StencilKind::backward;
        quote_stencil.first_weight = -200.0f;
        quote_stencil.second_weight = 50.0f;
        data::write_dataset(quote_recipe, quote_result);
        require(datasets::read_json_file(quote_recipe.dataset)["results"][0]["price"] == .1f);
        auto lsm_recipe = recipe;
        lsm_recipe.dataset = directory / "lsm_gradients.json";
        lsm_recipe.catalog = directory / "lsm_generation.yaml";
        lsm_recipe.exercise_replay =
            longstaff_schwartz::price_gradients::ExerciseReplayStrategy::
                frozen_regression_policy;
        data::write_dataset(lsm_recipe, result);
        const auto lsm_document = datasets::read_json_file(lsm_recipe.dataset);
        require(lsm_document["results"] == valid["results"]);
        auto sparse_recipe = recipe;
        sparse_recipe.configuration.sensitivities.push_back(
            {"model.volatility", {.25, pg::BumpScale::absolute}}
        );
        sparse_recipe.configuration.sensitivities.push_back(
            {"product.strike", {.5, pg::BumpScale::absolute}}
        );
        sparse_recipe.sensitivity_request = pg::SensitivityRequest::selected(
            {2U}, {}, {{0U, 1U}}
        );
        const auto sparse_metadata =
            data::sensitivity_metadata(sparse_recipe, true);
        require(
            sparse_metadata["orders"]
                == nlohmann::ordered_json::array(
                    {"first", "mixed_second"}
                )
            && sparse_metadata["first"]
                == nlohmann::ordered_json::array({"product.strike"})
            && sparse_metadata["mixed_second"][0U]["first"]
                == "model.rho"
            && sparse_metadata["mixed_second"][0U]["second"]
                == "model.volatility"
        );
        recipe.orders = pg::SensitivityOrders::first_and_second;
        pg::SensitivityStencil<4U> diagonal{};
        diagonal.kind = pg::StencilKind::backward;
        diagonal.parameter_values[0U] = 1.0f;
        diagonal.parameter_values[1U] = .875f;
        diagonal.parameter_values[2U] = .75f;
        diagonal.parameter_values[3U] = .625f;
        diagonal.displacement = .125f;
        diagonal.represented_width = -.125f;
        diagonal.first_endpoint_weights[0U] = -16.0f;
        diagonal.first_endpoint_weights[1U] = 4.0f;
        diagonal.second_weights[0U] = 128.0f;
        diagonal.second_weights[1U] = -320.0f;
        diagonal.second_weights[2U] = 256.0f;
        diagonal.second_weights[3U] = -64.0f;
        diagonal.node_count = 4U;
        result.diagonal_stencils = {diagonal};
        result.diagonal_hessians = {.3f};
        result.diagonal_hessian_errors = {.03f};
        result.execution["scenario_count"] = nullptr;
        result.execution["materialized_scenario_count"] = 0U;
        data::write_dataset(recipe,result);
        const auto diagonal_document = datasets::read_json_file(recipe.dataset);
        require(diagonal_document["results"][0]["diagonal_hessians"]["model.rho"] == .3f);
        require(!diagonal_document["results"][0].contains("stencils"));
        require(diagonal_document["results"][0].size() == 8U);
        auto full_recipe = recipe;
        full_recipe.dataset = directory / "full_hessian.json";
        full_recipe.catalog = directory / "full_hessian_generation.yaml";
        full_recipe.option_side = "call";
        full_recipe.configuration.sensitivities.push_back(
            {"model.volatility", {.125, pg::BumpScale::absolute}}
        );
        full_recipe.sensitivity_request =
            pg::SensitivityRequest::full_hessian();
        const auto full_metadata = data::sensitivity_metadata(
            full_recipe, true
        );
        require(full_metadata["orders"] == nlohmann::ordered_json::array(
            {"first", "diagonal_second", "mixed_second"}
        ) && full_metadata["mixed_second"] == "all"
          && full_metadata["parameters"].size() == 2U);
        auto full_result = result;
        full_result.gradients = {.2f, .4f};
        full_result.gradient_errors = {.02f, .04f};
        full_result.stencils.push_back(full_result.stencils.front());
        full_result.diagonal_hessians = {.3f, .5f};
        full_result.diagonal_hessian_errors = {.03f, .05f};
        full_result.diagonal_stencils.push_back(diagonal);
        full_result.mixed_hessians = {.6f};
        full_result.mixed_hessian_errors = {.06f};
        full_result.mixed_stencils = {
            pg::make_mixed_sensitivity_stencil(diagonal, diagonal)
        };
        full_result.execution["sensitivity_count"] = 2U;
        data::write_dataset(full_recipe, full_result);
        const auto full_document = datasets::read_json_file(
            full_recipe.dataset
        );
        const auto& full_row = full_document["results"][0];
        require(full_document.size() == 9U
            && full_document["option_side"] == "call"
            && full_row.size() == 10U
            && full_row["gradients"]["model.volatility"] == .4f
            && full_row["diagonal_hessians"]["model.volatility"] == .5f
            && full_row["mixed_hessians"]["model.rho|model.volatility"] == .6f
            && full_row["mixed_hessian_standard_errors"]
                ["model.rho|model.volatility"] == .06f
            && !full_document.contains("sensitivity")
            && !full_document.contains("url")
            && !full_row.contains("stencils"));
        result.diagonal_stencils[0].second_weights[2U] =
            std::numeric_limits<float>::quiet_NaN();
        bool diagonal_rejected = false;
        try { data::write_dataset(recipe,result); }
        catch (const std::invalid_argument&) { diagonal_rejected = true; }
        require(diagonal_rejected && datasets::read_json_file(recipe.dataset) == diagonal_document);
        recipe.orders = pg::SensitivityOrders::first;
        result.diagonal_stencils.clear();
        result.diagonal_hessians.clear();
        result.diagonal_hessian_errors.clear();
        result.execution["scenario_count"] = 3U;
        result.execution.erase("materialized_scenario_count");
        data::write_dataset(recipe,result);
        result.gradients[0] = std::numeric_limits<float>::quiet_NaN();
        bool rejected = false;
        try { data::write_dataset(recipe,result); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected && datasets::read_json_file(recipe.dataset)==valid);
        result.gradients[0] = .2f;
        result.execution["paths_per_price"] = 0;
        result.price_errors.clear(); result.gradient_errors.clear();
        data::write_dataset(recipe,result);
        const auto deterministic = datasets::read_json_file(recipe.dataset);
        require(!deterministic["results"][0].contains("gradient_standard_errors"));
        require(!deterministic["results"][0].contains("standard_error"));

        models["models"].push_back({{"id", "000002"}});
        products["products"].push_back({{"id", "000002"}});
        nlohmann::ordered_json curves{
            {"database_id", "curves"},
            {"catalog", "catalog/test"},
            {"curve_family", "Test curve"},
            {"url", "https://datasets.ai-factory.example/curves.json"},
            {"curves", {{{"id", "000001"}}, {{"id", "000002"}}}},
        };
        datasets::write_json_file(directory/"models.json", models);
        datasets::write_json_file(directory/"products.json", products);
        datasets::write_json_file(directory/"curves.json", curves);
        recipe.curve_input = directory/"curves.json";
        recipe.dataset = directory/"curve_gradients.json";
        recipe.construction = PriceConstruction::CartesianProduct;
        result.prices.assign(8U, .1f);
        result.gradients.assign(8U, .2f);
        result.stencils.assign(8U, result.stencils.front());
        data::write_dataset(recipe, result);
        const auto curve_document = datasets::read_json_file(recipe.dataset);
        require(curve_document.size() == 10U);
        auto family_field = curve_document.begin();
        require(family_field.key() == "database_id");
        ++family_field;
        require(family_field.key() == "model_family");
        ++family_field;
        require(family_field.key() == "product_family");
        ++family_field;
        require(family_field.key() == "curve_family");
        require(curve_document["curve_family"] == "Test curve"
            && curve_document["curve_dataset"]["id"] == "curves"
            && curve_document["curve_dataset"]["catalog"] == "catalog/test");
        require(curve_document["results"].size() == 8U);
        require(curve_document["results"][0]["model_id"] == "000001"
            && curve_document["results"][0]["curve_id"] == "000001"
            && curve_document["results"][0]["product_id"] == "000001");
        require(curve_document["results"][3]["model_id"] == "000001"
            && curve_document["results"][3]["curve_id"] == "000002"
            && curve_document["results"][3]["product_id"] == "000002");
        require(curve_document["results"][4]["model_id"] == "000002"
            && curve_document["results"][4]["curve_id"] == "000001"
            && curve_document["results"][4]["product_id"] == "000001");
        std::filesystem::remove_all(directory);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; std::filesystem::remove_all(directory); return 1;
    }
}
