// Convert Forward-start-option JSON rows into compact CUDA parameters.
#include "product/forward_start_option/dataset.hpp"
#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::product {

// Parse one dataset and preserve its row order in the returned vector.
std::vector<ForwardStartOptionParameters> load_forward_start_options(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<ForwardStartOptionParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Product,
        "Forward-start option",
        [&](const nlohmann::json& parameters, const std::string& prefix) {
            const ForwardStartOptionParameters product = {
                parameters.at("moneyness").get<float>(),
                parameters.at("reset_time").get<std::uint32_t>(),
                parameters.at("maturity").get<std::uint32_t>(),
            };
            if (!std::isfinite(product.moneyness) || !(product.moneyness > 0.0f))
                throw std::invalid_argument(prefix + "moneyness must be finite and positive.");
            if (product.reset_time_days == 0U)
                throw std::invalid_argument(prefix + "reset_time must be a positive business-day count.");
            if (product.maturity_days == 0U)
                throw std::invalid_argument(prefix + "maturity must be a positive business-day count.");
            if (!(product.reset_time_days < product.maturity_days))
                throw std::invalid_argument(prefix + "reset_time must precede maturity.");
            return product;
    });
}

}  // namespace ai_factory::workbench::product
