// Generate CIR European receiver-swaption prices.
#include "common/fixed_income/swaption_side.cuh"
#include "model/fixed_income/cir/european_swaption.cuh"
#include "tools/datasets/european_swaption_price_generation.hpp"

#include <filesystem>

int main() {
    using namespace ai_factory::workbench;
    namespace cir = model::cir;

    const std::filesystem::path model_path =
        "datasets/model/fixed_income/cir/parameters/cir_01.json";
    const std::filesystem::path product_path =
        "datasets/product/fixed_income/european_swaptions/"
        "european_swaptions_01.json";
    datasets::generate_regular_european_swaption_prices(
        model_path,
        product_path,
        cir::load_models(model_path),
        product::load_european_swaptions(product_path),
        [](auto... arguments) {
            cir::launch_cir_european_swaption_cuda<SwaptionSide::receiver>(
                arguments...
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
        "CIR European receiver swaption"
    );
}
