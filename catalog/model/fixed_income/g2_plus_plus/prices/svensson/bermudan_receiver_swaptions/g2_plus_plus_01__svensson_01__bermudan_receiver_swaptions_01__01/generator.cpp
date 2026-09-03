// Build G2++/Svensson Bermudan-receiver-swaption prices.
#include "model/fixed_income/g2_plus_plus/svensson/bermudan_swaption.cuh"
#include "model/fixed_income/g2_plus_plus/dataset.hpp"
#include "product/bermudan_swaption/dataset.hpp"
#include "tools/pricing/bermudan_swaption_price_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace rates = model::fixed_income::g2_plus_plus;
    namespace fitted = rates::svensson;
    const std::filesystem::path model_path =
        "datasets/model/fixed_income/g2_plus_plus/parameters/g2_plus_plus_01.json";
    const std::filesystem::path curve_path =
        "datasets/curve/svensson/svensson_01.json";
    const std::filesystem::path product_path =
        "datasets/product/fixed_income/bermudan_swaptions/bermudan_swaptions_01.json";
    const auto models = rates::load_models(model_path);
    const auto curves = curve::svensson::load_curves(curve_path);
    const auto products = product::load_bermudan_swaptions(product_path);
    constexpr std::size_t paths = 1U << 20U;
    constexpr std::uint64_t seed = 2'161'100'001ULL;
    datasets::generate_exact_fitted_bermudan_swaption_prices(
        model_path, curve_path, product_path, models, curves, products,
        &fitted::launch_g2_plus_plus_svensson_bermudan_swaption_cuda<
            SwaptionSide::receiver
        >,
        datasets::make_fitted_bermudan_swaption_generation_configuration(
            "g2_plus_plus", "svensson", "receiver", paths, seed,
            "Exact fitted two-factor Gaussian joint transition + Longstaff-Schwartz",
            "two-factor Hermite degree 2", "two standardized rate factors"
        )
    );
}
