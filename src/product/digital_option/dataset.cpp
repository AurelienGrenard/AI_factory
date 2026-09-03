// Convert digital-option JSON rows into compact CUDA parameters.
#include "product/digital_option/dataset.hpp"
#include "common/dataset_validation.hpp"

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
            if (!std::isfinite(product.strike) || !(product.strike > 0.0f))
                throw std::invalid_argument(prefix + "strike must be finite and positive.");
            if (product.maturity_days == 0U)
                throw std::invalid_argument(prefix + "maturity must be a positive business-day count.");
            if (!std::isfinite(product.cash_payoff) || !(product.cash_payoff > 0.0f))
                throw std::invalid_argument(prefix + "cash_payoff must be finite and positive.");
            return product;
    });
}

}  // namespace ai_factory::workbench::product
