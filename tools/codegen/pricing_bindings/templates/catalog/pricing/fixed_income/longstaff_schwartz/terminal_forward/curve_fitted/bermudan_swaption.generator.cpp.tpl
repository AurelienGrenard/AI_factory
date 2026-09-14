// Generated fitted ${model_display} ${curve_display} Bermudan swaption recipe.
#include "model/fixed_income/${model}/product/${curve}/bermudan_swaption.cuh"
#include "model/fixed_income/${model}/dataset.hpp"
#include "curve/${curve}/dataset.hpp"
#include "product/bermudan_swaption/dataset.hpp"
#include "tools/pricing/bermudan_swaption_price_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace rates = model::fixed_income::${model};
    namespace fitted = rates::${curve};
    const std::filesystem::path model_path =
        "datasets/model/fixed_income/${model}/parameters/${model}_01.json";
    const std::filesystem::path curve_path = "datasets/curve/${curve}/${curve}_01.json";
    const std::filesystem::path product_path =
        "datasets/product/bermudan_swaption/bermudan_swaptions_01.json";
    const auto models = rates::load_models(model_path);
    const auto curves = curve::${curve}::load_curves(curve_path);
    const auto products = product::load_bermudan_swaptions(product_path);
    constexpr std::size_t paths = ::ai_factory::workbench::offline::cuda_tuning::kProductionPathsPerPrice;
    constexpr std::uint64_t seed = ${dynamics_seed}ULL;
    auto configuration = datasets::make_fitted_bermudan_swaption_generation_configuration(
        "${model}", "${curve}", "${swaption_side}", paths, seed,
        "Exact fitted CIR terminal-forward transitions + Longstaff-Schwartz",
        "Hermite degree 3", "standardized unshifted CIR factor",
        PriceConstruction::${construction}
    );
    configuration.pricing_measure = "last_exercise_bond_forward";
    configuration.regression_target = "next policy cashflow in P(0,T*) / P(t,T*) units";
    datasets::generate_exact_fitted_bermudan_swaption_prices(
        model_path, curve_path, product_path, models, curves, products,
        &fitted::launch_${model}_${curve}_bermudan_swaption_cuda<SwaptionSide::${swaption_side}>,
        configuration
    );
}
