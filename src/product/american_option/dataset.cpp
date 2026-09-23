// Convert American-option JSON rows into compact CUDA parameters.
#include "product/american_option/dataset.hpp"
#include "product/american_option/parameter_domain.hpp"
#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

namespace ai_factory::workbench::product {

// Parse one dataset and preserve its row order in the returned vector.
std::vector<AmericanOptionParameters> load_american_options(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<AmericanOptionParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Product,
        "American option",
        [&](const nlohmann::json& parameters, const std::string& prefix) {
            const AmericanOptionParameters product = {
                parameters.at("strike").get<float>(),
                parameters.at("maturity").get<std::uint32_t>(),
                parameters.at("exercise_interval").get<std::uint32_t>(),
            };
            american_option::validate_parameters(product, prefix);
            return product;
    });
}

}  // namespace ai_factory::workbench::product
