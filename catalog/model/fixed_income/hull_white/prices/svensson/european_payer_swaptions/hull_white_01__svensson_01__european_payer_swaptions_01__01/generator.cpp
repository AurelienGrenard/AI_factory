// Generate Hull-White/Svensson European payer-swaption prices.
#include "common/fixed_income/swaption_side.cuh"
#include "model/fixed_income/hull_white/svensson/european_swaption.cuh"
#include "tools/datasets/european_swaption_price_generation.hpp"

#include <filesystem>

int main() {
    using namespace ai_factory::workbench;
    namespace hw = model::hull_white;
    namespace fitted = hw::svensson;
    namespace sv = curve::svensson;

    const std::filesystem::path model_path =
        "datasets/model/fixed_income/hull_white/parameters/hull_white_01.json";
    const std::filesystem::path curve_path =
        "datasets/curve/svensson/svensson_01.json";
    const std::filesystem::path product_path =
        "datasets/product/fixed_income/european_swaptions/"
        "european_swaptions_01.json";
    datasets::generate_regular_european_swaption_prices(
        model_path,
        curve_path,
        product_path,
        hw::load_models(model_path),
        sv::load_curves(curve_path),
        product::load_european_swaptions(product_path),
        [](auto... arguments) {
            fitted::launch_hull_white_svensson_european_swaption_cuda<
                SwaptionSide::payer
            >(arguments...);
        },
        "datasets/model/fixed_income/hull_white/prices/svensson/"
        "european_payer_swaptions/"
        "hull_white_01__svensson_01__european_payer_swaptions_01__01.json",
        "catalog/model/fixed_income/hull_white/prices/svensson/"
        "european_payer_swaptions/"
        "hull_white_01__svensson_01__european_payer_swaptions_01__01/"
        "dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/fixed_income/hull_white/"
        "prices/svensson/european_payer_swaptions/"
        "hull_white_01__svensson_01__european_payer_swaptions_01__01.json",
        "Closed-form Jamshidian decomposition into zero-coupon bond puts",
        "Hull-White Svensson European payer swaption"
    );
}
