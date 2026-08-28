// Generate Ornstein-Uhlenbeck European payer-swaption prices.
#include "common/fixed_income/swaption_side.cuh"
#include "model/fixed_income/ornstein_uhlenbeck/product/european_swaption.cuh"
#include "model/fixed_income/ornstein_uhlenbeck/dataset.hpp"
#include "product/european_swaption/dataset.hpp"
#include "tools/pricing/european_swaption_price_generation.cuh"

#include <filesystem>

int main() {
    using namespace ai_factory::workbench;
    namespace ou = model::fixed_income::ornstein_uhlenbeck;

    const std::filesystem::path model_path =
        "datasets/model/fixed_income/ornstein_uhlenbeck/parameters/"
        "ornstein_uhlenbeck_01.json";
    const std::filesystem::path product_path =
        "datasets/product/fixed_income/european_swaptions/"
        "european_swaptions_01.json";
    datasets::generate_regular_european_swaption_prices(
        model_path,
        product_path,
        ou::load_models(model_path),
        product::load_european_swaptions(product_path),
        [](auto... arguments) {
            ou::launch_ornstein_uhlenbeck_european_swaption_cuda<
                SwaptionSide::payer
            >(arguments...);
        },
        "datasets/model/fixed_income/ornstein_uhlenbeck/prices/"
        "european_payer_swaptions/"
        "ornstein_uhlenbeck_01__european_payer_swaptions_01__01.json",
        "catalog/model/fixed_income/ornstein_uhlenbeck/prices/"
        "european_payer_swaptions/"
        "ornstein_uhlenbeck_01__european_payer_swaptions_01__01/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/fixed_income/"
        "ornstein_uhlenbeck/prices/european_payer_swaptions/"
        "ornstein_uhlenbeck_01__european_payer_swaptions_01__01.json",
        "Closed-form Jamshidian decomposition into zero-coupon bond puts",
        "OU European payer swaption"
    );
}
