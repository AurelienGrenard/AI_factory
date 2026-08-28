// Build Vasicek Bermudan-payer-swaption prices with Longstaff-Schwartz.
#include "model/fixed_income/vasicek/bermudan_swaption.cuh"
#include "model/fixed_income/vasicek/dataset.hpp"
#include "tools/pricing/bermudan_swaption_price_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace rates = model::fixed_income::vasicek;
    const std::filesystem::path model_path =
        "datasets/model/fixed_income/vasicek/parameters/vasicek_01.json";
    const std::filesystem::path product_path =
        "datasets/product/fixed_income/bermudan_swaptions/"
        "bermudan_swaptions_01.json";
    const auto models = rates::load_models(model_path);
    const auto products = product::load_bermudan_swaptions(product_path);
    constexpr std::size_t paths = 1U << 20U;
    constexpr std::uint64_t seed = 2'120'000'001ULL;
    datasets::generate_exact_bermudan_swaption_prices(
        model_path, product_path, models, products,
        &rates::launch_vasicek_bermudan_swaption_cuda<SwaptionSide::payer>,
        datasets::make_bermudan_swaption_generation_configuration(
            "vasicek", "payer", paths, seed,
            "Exact Gaussian joint transition + Longstaff-Schwartz",
            "Hermite degree 3", "standardized short-rate factor", "",
            {{"time_day_fraction", "1 / 252"}}
        )
    );
}
