// Generated Generate CIR++/Nelson-Siegel European payer-swaption prices.
#include "common/fixed_income/swaption_side.cuh"
#include "model/fixed_income/cir_plus_plus/product/nelson_siegel/european_swaption.cuh"
#include "curve/nelson_siegel/dataset.hpp"
#include "model/fixed_income/cir_plus_plus/dataset.hpp"
#include "product/european_swaption/dataset.hpp"
#include "tools/pricing/european_swaption_price_generation.cuh"

#include <filesystem>

int main() {
    using namespace ai_factory::workbench;
    namespace cir_plus_plus = model::fixed_income::cir_plus_plus;
    namespace fitted = cir_plus_plus::nelson_siegel;
    namespace ns = curve::nelson_siegel;

    const std::filesystem::path model_path =
        "datasets/model/fixed_income/cir_plus_plus/parameters/cir_plus_plus_01.json";
    const std::filesystem::path curve_path =
        "datasets/curve/nelson_siegel/nelson_siegel_01.json";
    const std::filesystem::path product_path =
        "datasets/product/european_swaption/"
        "european_swaptions_01.json";
    const auto product_dataset = product::load_european_swaptions(product_path);
    datasets::generate_regular_european_swaption_prices(
        model_path,
        curve_path,
        product_path,
        cir_plus_plus::load_models(model_path),
        ns::load_curves(curve_path),
        product_dataset,
        [maximum_payment_count = product_dataset.maximum_payment_count](
            const offline::cuda_tuning::PricingLaunchPlan& plan, auto... arguments) {
            fitted::launch_cir_plus_plus_nelson_siegel_european_swaption_cuda<
                SwaptionSide::payer
            >(arguments..., maximum_payment_count,
                plan.profile.distribution == offline::cuda_tuning::PriceWorkDistribution::block
                    ? closed_form::WorkDistribution::cooperative : closed_form::WorkDistribution::scalar);
        },
        "datasets/model/fixed_income/cir_plus_plus/prices/nelson_siegel/"
        "european_payer_swaptions/"
        "cir_plus_plus_01__nelson_siegel_01__european_payer_swaptions_01__01_cartesian.json",
        "catalog/model/fixed_income/cir_plus_plus/prices/nelson_siegel/"
        "european_payer_swaptions/"
        "cir_plus_plus_01__nelson_siegel_01__european_payer_swaptions_01__01_cartesian/"
        "generation.yaml",
        "https://datasets.ai-factory.example/v1/model/fixed_income/cir_plus_plus/"
        "prices/nelson_siegel/european_payer_swaptions/"
        "cir_plus_plus_01__nelson_siegel_01__european_payer_swaptions_01__01_cartesian.json",
        "Closed-form Jamshidian decomposition into zero-coupon bond puts",
        "CIR++ Nelson-Siegel European payer swaption",
        ::ai_factory::workbench::offline::cuda_tuning::PricingIdentity{::ai_factory::workbench::offline::cuda_tuning::PricingFamily::jamshidian, "cir_plus_plus", "european_swaption", "nelson_siegel"},
        PriceConstruction::CartesianProduct
    );
}
