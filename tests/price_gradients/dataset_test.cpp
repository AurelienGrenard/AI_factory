// Native gradient artifact checks include one-sided stencils and fail-before-write behavior.
#include "tools/datasets/price_gradients/dataset.hpp"
#include "tools/datasets/artifact_io.hpp"
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
        nlohmann::ordered_json models{{"database_id","models"},{"catalog","catalog/test"},
            {"url","https://datasets.ai-factory.example/models.json"}, {"models",{{{"id","000001"}}}}};
        auto products = models;
        products.erase("models"); products["products"] = {{{"id","000001"}}};
        products["time_convention"] = {{"unit","business_day"},{"days_per_year",252}};
        datasets::write_json_file(directory/"models.json",models);
        datasets::write_json_file(directory/"products.json",products);
        data::Recipe recipe{directory/"models.json",directory/"products.json",directory/"gradients.json",directory/"generation.yaml",
            "https://datasets.ai-factory.example/gradients.json","source/generator.cpp",PriceConstruction::Aligned,
            {{{"model.rho",{.125,pg::BumpScale::absolute}}}}, {}};
        data::Results result{{.1f},{.01f},{.2f},{.02f},
            {pg::prepare_stencil(1.f,{.125,pg::BumpScale::absolute},[](float p){return p>=-1 && p<=1;})},
            {{"paths_per_price",32},{"sensitivity_count",1},{"scenario_count",3},{"seed",719}}};
        data::write_dataset(recipe,result);
        const auto valid = datasets::read_json_file(recipe.dataset);
        require(valid["results"][0]["stencils"]["model.rho"]["kind"]=="backward");
        require(valid["validation"]["verified"]==false);
        require(valid["sensitivity"]["parameters"][0]["displacement"]==.125);
        require(valid["sensitivity"]["method"]
            == "finite_difference_shared_innovations");
        auto lsm_recipe = recipe;
        lsm_recipe.dataset = directory / "lsm_gradients.json";
        lsm_recipe.catalog = directory / "lsm_generation.yaml";
        lsm_recipe.exercise_replay =
            longstaff_schwartz::price_gradients::ExerciseReplayStrategy::
                frozen_regression_policy;
        data::write_dataset(lsm_recipe, result);
        const auto lsm_document = datasets::read_json_file(lsm_recipe.dataset);
        require(lsm_document["standard_error_scope"].get<std::string>()
            .find("fit variability") != std::string::npos);
        require(lsm_document["lsm_fit_and_valuation_paths"] == "same paths");
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
        require(diagonal_document["results"][0]["stencils"]["model.rho"]["node_count"] == 4);
        require(diagonal_document["results"][0]["stencils"]["model.rho"]["third"] == .625);
        require(diagonal_document["results"][0]["outputs"]["diagonal_hessians"]["model.rho"] == .3f);
        require(diagonal_document["sensitivity"]["orders"] ==
            nlohmann::ordered_json::array({"first", "diagonal_second"}));
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
        require(!deterministic["results"][0]["outputs"].contains("gradient_standard_errors"));
        require(deterministic["sensitivity"]["method"]
            == "finite_difference");

        models["models"].push_back({{"id", "000002"}});
        products["products"].push_back({{"id", "000002"}});
        nlohmann::ordered_json curves{
            {"database_id", "curves"},
            {"catalog", "catalog/test"},
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
        require(curve_document["row_count"] == 8U);
        require(curve_document["price_construction"]["order"]
            == "model, curve, product");
        require(curve_document["curve_dataset"]["id"] == "curves");
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
