#include "product/basket_call/dataset.hpp"
#include "common/dataset_validation.hpp"

#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::product {

std::vector<BasketCallParameters> load_basket_calls(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<BasketCallParameters>(
        dataset_path, datasets::ParameterDatasetFamily::Product,
        "Basket call",
        [](const nlohmann::json& parameters, const std::string& prefix) {
            const BasketCallParameters product{
                parameters.at("strike").get<float>(),
                parameters.at("maturity").get<std::uint32_t>()
            };
            if (!std::isfinite(product.strike)
                || product.strike < 0.0f || product.maturity_days == 0U)
                throw std::invalid_argument(
                    prefix + "strike and maturity are out of domain.");
            return product;
        }
    );
}

}  // namespace ai_factory::workbench::product
