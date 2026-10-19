// A short native campaign verifies diagonal rows and checkpoint replay.
#include "model/equity/markovian/heston/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/heston/product/american_option_price_gradients.cuh"
#include "tools/datasets/artifact_io.hpp"
#include "tools/pricing/price_gradients/generation.cuh"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

#include <cuda_runtime.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    using namespace ai_factory::workbench;
    namespace heston = model::equity::heston;
    namespace pg = price_gradients;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    char pattern[] = "/tmp/ai_factory_diagonal_dataset_XXXXXX";
    if (mkdtemp(pattern) == nullptr) return 1;
    const std::filesystem::path directory(pattern);
    try {
        const auto require = [](bool valid) {
            if (!valid) throw std::runtime_error("Diagonal dataset replay failed.");
        };
        const auto receipt_contains = [](
            const std::filesystem::path& path, const std::string& token
        ) {
            std::ifstream input(path);
            const std::string content{
                std::istreambuf_iterator<char>(input),
                std::istreambuf_iterator<char>()
            };
            return content.find(token) != std::string::npos;
        };
        datasets::write_json_file(directory / "models.json", {
            {"database_id", "model_fixture"}, {"catalog", "catalog/test"},
            {"url", "https://datasets.ai-factory.example/model_fixture.json"},
            {"models", {{{"id", "000001"}}}}
        });
        datasets::write_json_file(directory / "products.json", {
            {"database_id", "product_fixture"}, {"catalog", "catalog/test"},
            {"url", "https://datasets.ai-factory.example/product_fixture.json"},
            {"time_convention", {{"unit", "business_day"}, {"days_per_year", 252}}},
            {"products", {{{"id", "000001"}}}}
        });
        datasets::price_gradients::Recipe recipe{
            directory / "models.json", directory / "products.json",
            directory / "diagonal.json", directory / "generation.yaml",
            "https://datasets.ai-factory.example/diagonal.json",
            "catalog/test/recipe.yaml", PriceConstruction::Aligned,
            {{{"model.spot", {.005, pg::BumpScale::relative}},
              {"model.rho", {.125, pg::BumpScale::absolute}}}},
            {1.0f/504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second
        };
        const std::vector<heston::ModelParameters> models{
            {1.0f, 0.0f, 0.0f, .04f, 1.5f, .04f, .4f, 1.0f}
        };
        const std::vector<product::EuropeanOptionParameters> products{
            {1.0f, 8U}
        };
        const auto prepare = [](const auto& model_rows,
                                const auto& product_rows,
                                PriceConstruction construction,
                                pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return heston::prepare_heston_european_option_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first_and_second}
            );
        };
        const auto execute = [&] {
            return offline::pricing::price_gradients::execute_node_graph_dataset<
                pg::SensitivityOrders::first_and_second>(
                recipe,
                {offline::cuda_tuning::PricingFamily::equity_step_mc,
                 "heston", "european_option", ""},
                1709U, models, products, prepare,
                heston::launch_heston_european_option_diagonal_sensitivities_cuda<
                    OptionSide::call, pg::SensitivityOrders::first_and_second>,
                heston::heston_european_option_node_graph_workspace_bytes<
                    OptionSide::call, pg::SensitivityOrders::first_and_second>,
                heston::launch_heston_european_option_node_graph_sensitivities_cuda<
                    OptionSide::call, pg::SensitivityOrders::first_and_second>,
                2048U,
                heston::prepare_european_option_diagonal_sensitivity_stencils_cuda
            );
        };
        const auto checkpoint = directory / "checkpoint";
        require(setenv("AI_FACTORY_GENERATION_CHECKPOINT_DIR",
                       checkpoint.c_str(), 1) == 0);
        require(setenv("AI_FACTORY_GENERATION_CHECKPOINT_ID",
                       std::string(64U, 'a').c_str(), 1) == 0);
        require(execute() == 0);
        const auto first = datasets::read_json_file(recipe.dataset);
        require(first["results"].size() == 1U);
        require(!first["results"][0U].contains("stencils"));
        require(first["results"][0U]["diagonal_hessians"].size() == 2U);
        require(execute() == 0);
        const auto replay = datasets::read_json_file(recipe.dataset);
        require(replay["results"] == first["results"]);
        require(receipt_contains(recipe.catalog, "resumed_prices: 1"));
        require(receipt_contains(recipe.catalog, "sensitivity_strategy: \"mono\""));
        unsetenv("AI_FACTORY_GENERATION_CHECKPOINT_DIR");
        unsetenv("AI_FACTORY_GENERATION_CHECKPOINT_ID");
        recipe.sensitivity_strategy =
            pg::SensitivityStrategy::node_graph;
        require(execute() == 0);
        const auto graph = datasets::read_json_file(recipe.dataset);
        require(graph["results"] == first["results"]);
        require(receipt_contains(recipe.catalog, "sensitivity_strategy: \"node_graph\""));
        require(receipt_contains(recipe.catalog, "maximum_live_scenarios: 7"));
        require(receipt_contains(recipe.catalog, "sensitivity_graph_node_capacity: 7"));
        require(receipt_contains(recipe.catalog, "represented_nodes_per_sensitivity: 4"));
        require(receipt_contains(recipe.catalog, "sensitivity_workspace_bytes:"));

        auto mixed_recipe = recipe;
        mixed_recipe.dataset = directory / "mixed.json";
        mixed_recipe.catalog = directory / "mixed_generation.yaml";
        mixed_recipe.url =
            "https://datasets.ai-factory.example/mixed.json";
        mixed_recipe.sensitivity_request =
            pg::SensitivityRequest::full_hessian();
        const auto prepare_mixed = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return heston::prepare_heston_european_option_sensitivities(
                model_rows,
                product_rows,
                construction,
                time,
                selected,
                std::move(request)
            );
        };
        const auto mixed_checkpoint = checkpoint / "mixed";
        require(setenv(
            "AI_FACTORY_GENERATION_CHECKPOINT_DIR",
            mixed_checkpoint.c_str(),
            1
        ) == 0);
        require(setenv(
            "AI_FACTORY_GENERATION_CHECKPOINT_ID",
            std::string(64U, 'c').c_str(),
            1
        ) == 0);
        const auto execute_mixed = [&] {
            return offline::pricing::price_gradients::
                execute_mixed_node_graph_dataset<true>(
                    mixed_recipe,
                    {
                        offline::cuda_tuning::PricingFamily::equity_step_mc,
                        "heston",
                        "european_option",
                        "",
                    },
                    1709U,
                    models,
                    products,
                    prepare_mixed,
                    heston::
                        heston_european_option_mixed_node_graph_workspace_bytes<
                            OptionSide::call>,
                    heston::
                        launch_heston_european_option_mixed_node_graph_sensitivities_cuda<
                            OptionSide::call>,
                    2048U,
                    heston::
                        prepare_european_option_diagonal_sensitivity_stencils_cuda
                );
        };
        require(execute_mixed() == 0);
        const auto mixed = datasets::read_json_file(mixed_recipe.dataset);
        require(mixed["results"].size() == 1U);
        require(mixed["results"][0U]["price"]
            == graph["results"][0U]["price"]);
        require(mixed["results"][0U]["gradients"]
            == graph["results"][0U]["gradients"]);
        require(mixed["results"][0U]["diagonal_hessians"]
            == graph["results"][0U]["diagonal_hessians"]);
        require(
            mixed["results"][0U]["mixed_hessians"].size() == 1U
        );
        require(!mixed["results"][0U].contains("mixed_stencils"));
        require(execute_mixed() == 0);
        const auto mixed_replay =
            datasets::read_json_file(mixed_recipe.dataset);
        require(mixed_replay["results"] == mixed["results"]);
        require(
            receipt_contains(mixed_recipe.catalog, "resumed_prices: 1")
        );
        unsetenv("AI_FACTORY_GENERATION_CHECKPOINT_DIR");
        unsetenv("AI_FACTORY_GENERATION_CHECKPOINT_ID");

        datasets::write_json_file(directory / "american_products.json", {
            {"database_id", "american_product_fixture"},
            {"catalog", "catalog/test"},
            {"url", "https://datasets.ai-factory.example/american_product_fixture.json"},
            {"time_convention", {{"unit", "business_day"}, {"days_per_year", 252}}},
            {"products", {{{"id", "000001"}}}}
        });
        const datasets::price_gradients::Recipe american_recipe{
            directory / "models.json", directory / "american_products.json",
            directory / "american_diagonal.json",
            directory / "american_generation.yaml",
            "https://datasets.ai-factory.example/american_diagonal.json",
            "catalog/test/american_recipe.yaml", PriceConstruction::Aligned,
            {{{"model.spot", {.005, pg::BumpScale::relative}},
              {"model.initial_variance", {.001, pg::BumpScale::absolute}},
              {"product.strike", {.005, pg::BumpScale::relative}}}},
            {1.0f/504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second
        };
        const std::vector<product::AmericanOptionParameters> american_products{
            {1.0f, 8U, 2U}
        };
        const auto prepare_american = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected
        ) {
            return heston::prepare_heston_american_option_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first_and_second}
            );
        };
        const auto execute_american = [&] {
            return offline::pricing::price_gradients::execute_dataset<
                true, pg::SensitivityOrders::first_and_second>(
                american_recipe,
                {offline::cuda_tuning::PricingFamily::equity_lsm,
                 "heston", "american_option", ""},
                1709U, models, american_products, prepare_american,
                heston::launch_heston_american_option_diagonal_sensitivities_cuda<
                    OptionSide::put, pg::SensitivityOrders::first_and_second>,
                2048U,
                heston::prepare_american_option_diagonal_sensitivity_stencils_cuda
            );
        };
        const auto american_checkpoint = checkpoint / "american";
        require(setenv("AI_FACTORY_GENERATION_CHECKPOINT_DIR",
                       american_checkpoint.c_str(), 1) == 0);
        require(setenv("AI_FACTORY_GENERATION_CHECKPOINT_ID",
                       std::string(64U, 'b').c_str(), 1) == 0);
        require(execute_american() == 0);
        const auto american_first = datasets::read_json_file(
            american_recipe.dataset
        );
        require(american_first["results"].size() == 1U);
        require(
            american_first["results"][0U]["diagonal_hessians"]
                .size() == 3U
        );
        require(execute_american() == 0);
        const auto american_replay = datasets::read_json_file(
            american_recipe.dataset
        );
        require(american_replay["results"] == american_first["results"]);
        require(
            receipt_contains(american_recipe.catalog, "resumed_prices: 1")
        );
        unsetenv("AI_FACTORY_GENERATION_CHECKPOINT_DIR");
        unsetenv("AI_FACTORY_GENERATION_CHECKPOINT_ID");

        unsetenv("AI_FACTORY_GENERATION_CHECKPOINT_DIR");
        unsetenv("AI_FACTORY_GENERATION_CHECKPOINT_ID");
        std::filesystem::remove_all(directory);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        unsetenv("AI_FACTORY_GENERATION_CHECKPOINT_DIR");
        unsetenv("AI_FACTORY_GENERATION_CHECKPOINT_ID");
        std::filesystem::remove_all(directory);
        return 1;
    }
}
