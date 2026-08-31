// Generated Generate CIR European receiver-swaption prices.
#include "common/fixed_income/swaption_side.cuh"
#include "model/fixed_income/cir/product/european_swaption.cuh"
#include "model/fixed_income/cir/dataset.hpp"
#include "product/european_swaption/dataset.hpp"
#include "tools/pricing/european_swaption_price_generation.cuh"

#include <filesystem>

int main() {
    using namespace ai_factory::workbench;
    namespace cir = model::fixed_income::cir;

    const std::filesystem::path model_path =
        "datasets/model/fixed_income/cir/parameters/cir_01.json";
    const std::filesystem::path product_path =
        "datasets/product/european_swaption/"
        "european_swaptions_01.json";
    const auto product_dataset =
        product::load_european_swaptions(product_path);
    datasets::generate_regular_european_swaption_prices(
        model_path,
        product_path,
        cir::load_models(model_path),
        product_dataset,
        [maximum_payment_count = product_dataset.maximum_payment_count](
            auto... arguments
        ) {
            cir::launch_cir_european_swaption_cuda<SwaptionSide::receiver>(
                arguments...,
                maximum_payment_count
            );
        },
        "datasets/model/fixed_income/cir/prices/european_receiver_swaptions/"
        "cir_01__european_receiver_swaptions_01__01.json",
        "catalog/model/fixed_income/cir/prices/european_receiver_swaptions/"
        "cir_01__european_receiver_swaptions_01__01/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/fixed_income/cir/prices/"
        "european_receiver_swaptions/"
        "cir_01__european_receiver_swaptions_01__01.json",
        "Closed-form Jamshidian decomposition into zero-coupon bond calls",
        "CIR European receiver swaption",
        datasets::EuropeanSwaptionGenerationConfiguration{
            128U,
            datasets::EuropeanSwaptionWorkDistribution::one_price_per_block,
        }
    );
}
