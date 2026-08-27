// Convert Up-one-touch JSON rows into compact CUDA parameters.
#include "product/up_one_touch/dataset.hpp"
#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::product {

// Parse one dataset and preserve its row order in the returned vector.
std::vector<UpOneTouchParameters> load_up_one_touches(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<UpOneTouchParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Product,
        "Up one-touch",
        [&](const nlohmann::json& parameters, const std::string& prefix) {
            const UpOneTouchParameters product = {
                parameters.at("barrier").get<float>(),
                parameters.at("cash_payoff").get<float>(),
                parameters.at("maturity").get<std::uint32_t>(),
            };
            if (!std::isfinite(product.barrier) || !(product.barrier > 0.0f))
                throw std::invalid_argument(prefix + "barrier must be finite and positive.");
            if (!std::isfinite(product.cash_payoff)
                || !(product.cash_payoff > 0.0f)) {
                throw std::invalid_argument(
                    prefix + "cash_payoff must be finite and positive."
                );
            }
            if (product.maturity_days == 0U)
                throw std::invalid_argument(prefix + "maturity must be a positive business-day count.");
            return product;
    });
}

}  // namespace ai_factory::workbench::product
