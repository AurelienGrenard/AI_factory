// Build Hull-White/Svensson Bermudan-receiver-swaption prices.
#include "model/fixed_income/hull_white/product/svensson/bermudan_swaption.cuh"
#include "model/fixed_income/hull_white/dataset.hpp"
#include "product/bermudan_swaption/dataset.hpp"
#include "product/bermudan_swaption/dataset.hpp"
#include "tools/pricing/bermudan_swaption_price_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace rates = model::fixed_income::hull_white;
    namespace fitted = rates::svensson;
    const std::filesystem::path model_path =
        "datasets/model/fixed_income/hull_white/parameters/hull_white_01.json";
    const std::filesystem::path curve_path =
        "datasets/curve/svensson/svensson_01.json";
    const std::filesystem::path product_path =
        "datasets/product/bermudan_swaption/bermudan_swaptions_01.json";
    const auto models = rates::load_models(model_path);
    const auto curves = curve::svensson::load_curves(curve_path);
    const auto products = product::load_bermudan_swaptions(product_path);
    constexpr std::size_t paths = 1U << 20U;
    constexpr std::uint64_t seed = 11668829112713084928ULL;
    datasets::generate_exact_fitted_bermudan_swaption_prices(
        model_path, curve_path, product_path, models, curves, products,
        &fitted::launch_hull_white_svensson_bermudan_swaption_cuda<
            SwaptionSide::receiver
        >,
        datasets::make_fitted_bermudan_swaption_generation_configuration(
            "hull_white", "svensson", "receiver", paths, seed,
            "Exact fitted Gaussian joint transition + Longstaff-Schwartz",
            "Hermite degree 3", "standardized centered short-rate factor"
        )
    );
}
