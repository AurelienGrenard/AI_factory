// Generated Ornstein-Uhlenbeck Bermudan-receiver-swaption price recipe.
#include "model/fixed_income/ornstein_uhlenbeck/product/bermudan_swaption.cuh"
#include "model/fixed_income/ornstein_uhlenbeck/dataset.hpp"
#include "product/bermudan_swaption/dataset.hpp"
#include "tools/pricing/bermudan_swaption_price_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace rates = model::fixed_income::ornstein_uhlenbeck;
    const std::filesystem::path model_path =
        "datasets/model/fixed_income/ornstein_uhlenbeck/parameters/ornstein_uhlenbeck_01.json";
    const std::filesystem::path product_path =
        "datasets/product/bermudan_swaption/bermudan_swaptions_01.json";
    const auto models = rates::load_models(model_path);
    const auto products = product::load_bermudan_swaptions(product_path);
    constexpr std::size_t paths =
        offline::cuda_tuning::kProductionPathsPerPrice;
    constexpr std::uint64_t seed = 11668829129892954112ULL;
    auto configuration = datasets::make_bermudan_swaption_generation_configuration(
        "ornstein_uhlenbeck", "receiver", paths, seed,
        "Exact Gaussian joint transition + Longstaff-Schwartz",
        "Hermite degree 3", "standardized short-rate factor", "",
        {{"time_day_fraction", "1 / 252"}},
        PriceConstruction::Aligned
    );
    datasets::generate_exact_bermudan_swaption_prices(
        model_path, product_path, models, products,
        &rates::launch_ornstein_uhlenbeck_bermudan_swaption_cuda<
            SwaptionSide::receiver>,
        configuration
    );
}
