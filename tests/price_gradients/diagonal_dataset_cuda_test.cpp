// A short native campaign verifies diagonal rows and checkpoint replay.
#include "model/equity/markovian/heston/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/heston/product/american_option_price_gradients.cuh"
#include "model/fixed_income/cir/product/european_swaption_price_gradients.cuh"
#include "tools/datasets/artifact_io.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

#include <cuda_runtime.h>

#include <cstdlib>
#include <filesystem>
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
        const datasets::price_gradients::Recipe recipe{
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
            return offline::pricing::price_gradients::execute_dataset<
                true, pg::SensitivityOrders::first_and_second>(
                recipe,
                {offline::cuda_tuning::PricingFamily::equity_step_mc,
                 "heston", "european_option", ""},
                1709U, models, products, prepare,
                heston::launch_heston_european_option_diagonal_sensitivities_cuda<
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
        require(first["results"][0U]["stencils"]["model.rho"]["node_count"] == 4U);
        require(first["results"][0U]["outputs"]["diagonal_hessians"].size() == 2U);
        require(execute() == 0);
        const auto replay = datasets::read_json_file(recipe.dataset);
        require(replay["results"] == first["results"]);
        require(replay["summary"]["checkpoint"]["resumed_prices"] == 1U);

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
            american_first["results"][0U]["outputs"]["diagonal_hessians"]
                .size() == 3U
        );
        require(execute_american() == 0);
        const auto american_replay = datasets::read_json_file(
            american_recipe.dataset
        );
        require(american_replay["results"] == american_first["results"]);
        require(
            american_replay["summary"]["checkpoint"]["resumed_prices"] == 1U
        );
        unsetenv("AI_FACTORY_GENERATION_CHECKPOINT_DIR");
        unsetenv("AI_FACTORY_GENERATION_CHECKPOINT_ID");

        datasets::write_json_file(directory / "cir_models.json", {
            {"database_id", "cir_model_fixture"}, {"catalog", "catalog/test"},
            {"url", "https://datasets.ai-factory.example/cir_model_fixture.json"},
            {"models", {{{"id", "000001"}}}}
        });
        datasets::write_json_file(directory / "swaptions.json", {
            {"database_id", "swaption_fixture"}, {"catalog", "catalog/test"},
            {"url", "https://datasets.ai-factory.example/swaption_fixture.json"},
            {"time_convention", {{"unit", "business_day"}, {"days_per_year", 252}}},
            {"products", {{{"id", "000001"}}}}
        });
        const datasets::price_gradients::Recipe cir_recipe{
            directory / "cir_models.json", directory / "swaptions.json",
            directory / "cir_diagonal.json",
            directory / "cir_generation.yaml",
            "https://datasets.ai-factory.example/cir_diagonal.json",
            "catalog/test/cir_recipe.yaml", PriceConstruction::Aligned,
            {{{"model.initial_state", {.0005, pg::BumpScale::absolute}},
              {"product.strike", {.0005, pg::BumpScale::absolute}}}},
            {1.0f/504.0f, 2U}, true,
            pg::SensitivityOrders::first_and_second
        };
        namespace cir = model::fixed_income::cir;
        const std::vector<cir::ModelParameters> cir_models{
            {{.6f, .04f, .15f}, 0.0f}
        };
        const std::vector<product::RegularEuropeanSwaptionParameters>
            swaptions{{1.0f, .035f, .5f, 252U, 126U, 4U}};
        const auto prepare_cir = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected
        ) {
            return cir::prepare_cir_european_swaption_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first_and_second}
            );
        };
        const auto launch_cir = [](
            const auto& plan,
            auto inputs,
            auto stencils,
            const auto& configuration,
            auto outputs
        ) {
            cir::launch_cir_european_swaption_diagonal_sensitivities_cuda<
                SwaptionSide::payer,
                pg::SensitivityOrders::first_and_second
            >(
                plan, inputs, stencils, configuration, outputs,
                closed_form::WorkDistribution::cooperative
            );
        };
        const auto execute_cir = [&] {
            return offline::pricing::price_gradients::execute_dataset<
                false, pg::SensitivityOrders::first_and_second>(
                cir_recipe,
                {offline::cuda_tuning::PricingFamily::jamshidian,
                 "cir", "european_swaption", ""},
                0U, cir_models, swaptions, prepare_cir, launch_cir, 0U,
                cir::prepare_european_swaption_diagonal_sensitivity_stencils_cuda
            );
        };
        require(execute_cir() == 0);
        const auto cir_first = datasets::read_json_file(cir_recipe.dataset);
        require(cir_first["results"].size() == 1U);
        require(
            cir_first["results"][0U]["stencils"]["model.initial_state"]
                ["node_count"] == 4U
        );
        require(
            cir_first["results"][0U]["outputs"]["diagonal_hessians"]
                .size() == 2U
        );
        require(execute_cir() == 0);
        const auto cir_regenerated = datasets::read_json_file(cir_recipe.dataset);
        require(cir_regenerated["results"] == cir_first["results"]);
        require(!cir_regenerated["summary"].contains("checkpoint"));
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
