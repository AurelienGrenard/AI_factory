// Host implementation of the Vasicek dataset loader.
#include "model/fixed_income/vasicek/dataset.hpp"
#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::model::fixed_income::vasicek {

// Parse and validate Vasicek rows while preserving their dataset order.
std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<ModelParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Model,
        "Vasicek model",
        [&](const nlohmann::json& parameters, const std::string& prefix) {
            const ModelParameters model = {
                {
                    parameters.at("mean_reversion").get<float>(),
                    parameters.at("long_term_mean").get<float>(),
                    parameters.at("volatility").get<float>(),
                },
                parameters.at("initial_state").get<float>(),
            };
            if (!std::isfinite(model.process.mean_reversion)
                || !(model.process.mean_reversion > 0.0f)) {
                throw std::invalid_argument(
                    prefix + "mean_reversion must be finite and positive."
                );
            }
            if (!std::isfinite(model.process.long_term_mean)) {
                throw std::invalid_argument(
                    prefix + "long_term_mean must be finite."
                );
            }
            if (!std::isfinite(model.process.volatility)
                || !(model.process.volatility >= 0.0f)) {
                throw std::invalid_argument(
                    prefix + "volatility must be finite and non-negative."
                );
            }
            if (!std::isfinite(model.initial_state))
                throw std::invalid_argument(prefix + "initial_state must be finite.");
            return model;
    });
}

}  // namespace ai_factory::workbench::model::fixed_income::vasicek
