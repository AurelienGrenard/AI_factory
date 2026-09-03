// Generated Generate ${model_display} European ${swaption_side}-swaption prices.
#include "common/fixed_income/swaption_side.cuh"
#include "model/fixed_income/${model}/product/european_swaption.cuh"
#include "model/fixed_income/${model}/dataset.hpp"
#include "product/european_swaption/dataset.hpp"
#include "tools/pricing/european_swaption_price_generation.cuh"

#include <filesystem>

int main() {
    using namespace ai_factory::workbench;
    namespace ${model_alias} = model::fixed_income::${model};

    const std::filesystem::path model_path =
        "datasets/model/fixed_income/${model}/parameters/${model}_01.json";
    const std::filesystem::path product_path =
        "datasets/product/european_swaption/"
        "european_swaptions_01.json";
    datasets::generate_regular_european_swaption_prices(
        model_path,
        product_path,
        ${model_alias}::load_models(model_path),
        product::load_european_swaptions(product_path),
        [](auto... arguments) {
            ${model_alias}::launch_${model}_european_swaption_cuda<SwaptionSide::${swaption_side}>(
                arguments...
            );
        },
        "datasets/model/fixed_income/${model}/prices/${variant}/"
        "${model}_01__${variant}_01__01.json",
        "catalog/model/fixed_income/${model}/prices/${variant}/"
        "${model}_01__${variant}_01__01/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/fixed_income/${model}/"
        "prices/${variant}/"
        "${model}_01__${variant}_01__01.json",
        "Closed-form Jamshidian decomposition into zero-coupon bond ${bond_option_side_plural}",
        "${model_display} European ${swaption_side} swaption"
    );
}
