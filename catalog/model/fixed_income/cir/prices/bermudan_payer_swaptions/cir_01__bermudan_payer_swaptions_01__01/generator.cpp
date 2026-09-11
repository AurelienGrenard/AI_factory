// Build CIR Bermudan-payer-swaption prices with Longstaff-Schwartz.
#include "model/fixed_income/cir/product/bermudan_swaption.cuh"
#include "model/fixed_income/cir/dataset.hpp"
#include "product/bermudan_swaption/dataset.hpp"
#include "tools/pricing/bermudan_swaption_price_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace rates = model::fixed_income::cir;
    const std::filesystem::path model_path =
        "datasets/model/fixed_income/cir/parameters/cir_01.json";
    const std::filesystem::path product_path =
        "datasets/product/bermudan_swaption/"
        "bermudan_swaptions_01.json";
    const auto models = rates::load_models(model_path);
    const auto products = product::load_bermudan_swaptions(product_path);
    constexpr std::size_t paths = offline::cuda_tuning::kProductionPathsPerPrice;
    constexpr std::uint64_t seed = 11668829039698640896ULL;
    auto configuration = datasets::make_bermudan_swaption_generation_configuration(
        "cir", "payer", paths, seed,
        "Exact CIR terminal-forward transitions + Longstaff-Schwartz",
        "Hermite degree 3", "standardized short-rate factor", "",
        {{"time_day_fraction", "1 / 252"}}
    );
    configuration.pricing_measure = "last_exercise_bond_forward";
    configuration.regression_target = "next policy cashflow in P(0,T*) / P(t,T*) units";
    datasets::generate_exact_bermudan_swaption_prices(
        model_path, product_path, models, products,
        &rates::launch_cir_bermudan_swaption_cuda<SwaptionSide::payer>,
        configuration
    );
}
