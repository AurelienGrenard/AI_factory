// Generated normal_inverse_gaussian down_and_in_option selected-gradient recipe.
#include "model/equity/markovian/normal_inverse_gaussian/product/down_and_in_option_price_gradients.cuh"
#include "model/equity/markovian/normal_inverse_gaussian/dataset.hpp"
#include "product/down_and_in_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/normal_inverse_gaussian/parameters/normal_inverse_gaussian_01.json", "datasets/product/down_and_in_option/down_and_in_options_01.json", "datasets/model/equity/markovian/normal_inverse_gaussian/price_sensitivities/down_and_in_puts/normal_inverse_gaussian_01__down_and_in_puts_01__01_price_sensitivities_spot.json", "catalog/model/equity/markovian/normal_inverse_gaussian/price_sensitivities/down_and_in_puts/normal_inverse_gaussian_01__down_and_in_puts_01__01_price_sensitivities_spot/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/normal_inverse_gaussian/price_sensitivities/down_and_in_puts/normal_inverse_gaussian_01__down_and_in_puts_01__01_price_sensitivities_spot.json", "catalog/model/equity/markovian/normal_inverse_gaussian/prices/down_and_in_puts/normal_inverse_gaussian_01__down_and_in_puts_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::normal_inverse_gaussian::load_models(recipe.model_input);
        const auto products = product::load_down_and_in_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "normal_inverse_gaussian", "down_and_in_option", ""}, 11668827600884596736ULL,
            models, products, model::equity::normal_inverse_gaussian::prepare_normal_inverse_gaussian_down_and_in_option_price_gradients,
            model::equity::normal_inverse_gaussian::launch_normal_inverse_gaussian_down_and_in_option_price_gradients_cuda<OptionSide::put>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::normal_inverse_gaussian::prepare_down_and_in_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
