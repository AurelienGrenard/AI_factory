// Generated Generate Vasicek European payer-swaption prices.
#include "common/fixed_income/swaption_side.cuh"
#include "model/fixed_income/vasicek/product/european_swaption.cuh"
#include "model/fixed_income/vasicek/dataset.hpp"
#include "product/european_swaption/dataset.hpp"
#include "tools/pricing/european_swaption_price_generation.cuh"

#include <filesystem>

int main() {
    using namespace ai_factory::workbench;
    namespace vasicek = model::fixed_income::vasicek;

    const std::filesystem::path model_path =
        "datasets/model/fixed_income/vasicek/parameters/vasicek_01.json";
    const std::filesystem::path product_path =
        "datasets/product/european_swaption/"
        "european_swaptions_01.json";
    const auto product_dataset = product::load_european_swaptions(product_path);
    datasets::generate_regular_european_swaption_prices(
        model_path,
        product_path,
        vasicek::load_models(model_path),
        product_dataset,
        [maximum_payment_count = product_dataset.maximum_payment_count](
            const offline::cuda_tuning::PricingLaunchPlan& plan, auto... arguments) {
            vasicek::launch_vasicek_european_swaption_cuda<SwaptionSide::payer>(
                arguments..., maximum_payment_count,
                plan.profile.distribution == offline::cuda_tuning::PriceWorkDistribution::block
                    ? closed_form::WorkDistribution::cooperative : closed_form::WorkDistribution::scalar
            );
        },
        "datasets/model/fixed_income/vasicek/prices/european_payer_swaptions/"
        "vasicek_01__european_payer_swaptions_01__01.json",
        "catalog/model/fixed_income/vasicek/prices/european_payer_swaptions/"
        "vasicek_01__european_payer_swaptions_01__01/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/fixed_income/vasicek/"
        "prices/european_payer_swaptions/"
        "vasicek_01__european_payer_swaptions_01__01.json",
        "Closed-form Jamshidian decomposition into zero-coupon bond puts",
        "Vasicek European payer swaption",
        ::ai_factory::workbench::offline::cuda_tuning::PricingIdentity{::ai_factory::workbench::offline::cuda_tuning::PricingFamily::jamshidian, "vasicek", "european_swaption", ""},
        PriceConstruction::Aligned
    );
}
