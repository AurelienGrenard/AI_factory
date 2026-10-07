// Generated fitted CIR++ Flat Bermudan swaption recipe.
#include "model/fixed_income/cir_plus_plus/product/flat/bermudan_swaption.cuh"
#include "model/fixed_income/cir_plus_plus/dataset.hpp"
#include "curve/flat/dataset.hpp"
#include "product/bermudan_swaption/dataset.hpp"
#include "tools/pricing/bermudan_swaption_price_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace rates = model::fixed_income::cir_plus_plus;
    namespace fitted = rates::flat;
    const std::filesystem::path model_path =
        "datasets/model/fixed_income/cir_plus_plus/parameters/cir_plus_plus_01.json";
    const std::filesystem::path curve_path = "datasets/curve/flat/flat_01.json";
    const std::filesystem::path product_path =
        "datasets/product/bermudan_swaption/bermudan_swaptions_01.json";
    const auto models = rates::load_models(model_path);
    const auto curves = curve::flat::load_curves(curve_path);
    const auto products = product::load_bermudan_swaptions(product_path);
    constexpr std::size_t paths = ::ai_factory::workbench::offline::cuda_tuning::kProductionPathsPerPrice;
    constexpr std::uint64_t seed = 11668829211497332736ULL;
    auto configuration = datasets::make_fitted_bermudan_swaption_generation_configuration(
        "cir_plus_plus", "flat", "payer", paths, seed,
        "Exact fitted CIR terminal-forward transitions + Longstaff-Schwartz",
        "Hermite degree 3", "standardized unshifted CIR factor",
        PriceConstruction::Aligned
    );
    configuration.pricing_measure = "last_exercise_bond_forward";
    configuration.regression_target = "next policy cashflow in P(0,T*) / P(t,T*) units";
    datasets::generate_exact_fitted_bermudan_swaption_prices(
        model_path, curve_path, product_path, models, curves, products,
        &fitted::launch_cir_plus_plus_flat_bermudan_swaption_cuda<SwaptionSide::payer>,
        configuration
    );
}
