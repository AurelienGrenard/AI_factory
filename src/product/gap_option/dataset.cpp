// Convert gap-option JSON rows into compact CUDA parameters.
#include "product/gap_option/dataset.hpp"
#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::product {

// Parse one dataset and preserve its row order in the returned vector.
std::vector<GapOptionParameters> load_gap_options(
    const std::filesystem::path& dataset_path,
    OptionSide side
) {
    return datasets::load_parameter_rows<GapOptionParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Product,
        "Gap option",
        [&](const nlohmann::json& parameters, const std::string& prefix) {
            const GapOptionParameters product = {
                parameters.at("trigger_strike").get<float>(),
                parameters.at("payoff_strike").get<float>(),
                parameters.at("maturity").get<std::uint32_t>(),
            };
            if (!std::isfinite(product.trigger_strike)
                || !(product.trigger_strike > 0.0f)) {
                throw std::invalid_argument(
                    prefix + "trigger_strike must be finite and positive."
                );
            }
            if (!std::isfinite(product.payoff_strike)
                || !(product.payoff_strike > 0.0f)) {
                throw std::invalid_argument(
                    prefix + "payoff_strike must be finite and positive."
                );
            }
            if (product.maturity_days == 0U)
                throw std::invalid_argument(prefix + "maturity must be a positive business-day count.");
            if (side == OptionSide::call
                && product.payoff_strike > product.trigger_strike) {
                throw std::invalid_argument(
                    prefix + "call payoff_strike must not exceed trigger_strike."
                );
            }
            if (side == OptionSide::put
                && product.payoff_strike < product.trigger_strike) {
                throw std::invalid_argument(
                    prefix + "put payoff_strike must not be below trigger_strike."
                );
            }
            return product;
    });
}

}  // namespace ai_factory::workbench::product
