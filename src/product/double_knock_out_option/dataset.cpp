// Convert Double-knock-out-option JSON rows into compact CUDA parameters.
#include "product/double_knock_out_option/dataset.hpp"
#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::product {

// Parse one dataset and preserve its row order in the returned vector.
std::vector<DoubleKnockOutOptionParameters> load_double_knock_out_options(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<DoubleKnockOutOptionParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Product,
        "Double-knock-out option",
        [&](const nlohmann::json& parameters, const std::string& prefix) {
            const DoubleKnockOutOptionParameters product = {
                parameters.at("strike").get<float>(),
                parameters.at("lower_barrier").get<float>(),
                parameters.at("upper_barrier").get<float>(),
                parameters.at("maturity").get<std::uint32_t>(),
            };
            if (!std::isfinite(product.strike) || !(product.strike > 0.0f))
                throw std::invalid_argument(prefix + "strike must be finite and positive.");
            if (!std::isfinite(product.lower_barrier)
                || !(product.lower_barrier > 0.0f)) {
                throw std::invalid_argument(prefix + "lower_barrier must be finite and positive.");
            }
            if (!std::isfinite(product.upper_barrier)
                || !(product.upper_barrier > product.lower_barrier)) {
                throw std::invalid_argument(prefix + "upper_barrier must exceed lower_barrier.");
            }
            if (!(product.strike > product.lower_barrier
                && product.strike < product.upper_barrier)) {
                throw std::invalid_argument(prefix + "strike must lie between both barriers.");
            }
            if (product.maturity_days == 0U)
                throw std::invalid_argument(prefix + "maturity must be a positive business-day count.");
            return product;
    });
}

}  // namespace ai_factory::workbench::product
