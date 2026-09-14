// Generated ${model_display} Bermudan-${swaption_side}-swaption price recipe.
#include "model/fixed_income/${model}/product/bermudan_swaption.cuh"
#include "model/fixed_income/${model}/dataset.hpp"
#include "product/bermudan_swaption/dataset.hpp"
#include "tools/pricing/bermudan_swaption_price_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace rates = model::fixed_income::${model};
    const std::filesystem::path model_path =
        "datasets/model/fixed_income/${model}/parameters/${model}_01.json";
    const std::filesystem::path product_path =
        "datasets/product/bermudan_swaption/bermudan_swaptions_01.json";
    const auto models = rates::load_models(model_path);
    const auto products = product::load_bermudan_swaptions(product_path);
    constexpr std::size_t paths =
        offline::cuda_tuning::kProductionPathsPerPrice;
    constexpr std::uint64_t seed = ${dynamics_seed}ULL;
    auto configuration = datasets::make_bermudan_swaption_generation_configuration(
        "${model}", "${swaption_side}", paths, seed,
        "${bermudan_numerical_method}",
        "${bermudan_regression_basis}", "${bermudan_state_variables}", "",
        {{"time_day_fraction", "1 / 252"}},
        PriceConstruction::${construction}
    );
${bermudan_configuration_overrides}    datasets::generate_exact_bermudan_swaption_prices(
        model_path, product_path, models, products,
        &rates::launch_${model}_bermudan_swaption_cuda<
            SwaptionSide::${swaption_side}>,
        configuration
    );
}
