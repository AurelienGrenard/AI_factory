// Convert digital-option JSON rows into compact CUDA parameters.
#include "product/digital_option/dataset.hpp"
#include "common/dataset_validation.hpp"
#include "product/digital_option/parameter_domain.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::product {

// Parse one dataset and preserve its row order in the returned vector.
std::vector<DigitalOptionParameters> load_digital_options(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<DigitalOptionParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Product,
        "Digital option",
        [&](const nlohmann::json& parameters, const std::string& prefix) {
            const DigitalOptionParameters product = {
                parameters.at("strike").get<float>(),
                parameters.at("maturity").get<std::uint32_t>(),
                parameters.at("cash_payoff").get<float>(),
            };
            digital_option::validate_parameters(product, prefix);
            return product;
    });
}

}  // namespace ai_factory::workbench::product
