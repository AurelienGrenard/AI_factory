// Convert American-option JSON rows into compact CUDA parameters.
#include "product/american_option/dataset.hpp"
#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>

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
            if (!std::isfinite(product.strike) || !(product.strike > 0.0f))
                throw std::invalid_argument(prefix + "strike must be finite and positive.");
            if (product.maturity_days == 0U)
                throw std::invalid_argument(prefix + "maturity must be a positive business-day count.");
            if (product.exercise_interval_days == 0U
                || !(product.exercise_interval_days < product.maturity_days)) {
                throw std::invalid_argument(
                    prefix
                    + "exercise_interval must be positive and below maturity."
                );
            }
            return product;
    });
}

}  // namespace ai_factory::workbench::product
