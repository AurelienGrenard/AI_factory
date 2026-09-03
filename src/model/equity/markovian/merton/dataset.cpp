// Host implementation of the Merton dataset loader.
#include "model/equity/markovian/merton/dataset.hpp"
#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::model::equity::merton {

// Parse and validate Merton rows while preserving their dataset order.
std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<ModelParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Model,
        "Merton model",
        [&](const nlohmann::json& parameters, const std::string& prefix) {
            const ModelParameters model = {
                parameters.at("spot").get<float>(),
                parameters.at("risk_free_rate").get<float>(),
                parameters.at("dividend_yield").get<float>(),
                parameters.at("volatility").get<float>(),
                parameters.at("jump_intensity").get<float>(),
                parameters.at("jump_log_mean").get<float>(),
                parameters.at("jump_log_volatility").get<float>(),
            };
            if (!std::isfinite(model.spot) || !(model.spot > 0.0f)) {
                throw std::invalid_argument(
                    prefix + "spot must be finite and positive."
                );
            }
            if (!std::isfinite(model.risk_free_rate)) {
                throw std::invalid_argument(
                    prefix + "risk_free_rate must be finite."
                );
            }
            if (!std::isfinite(model.dividend_yield)) {
                throw std::invalid_argument(
                    prefix + "dividend_yield must be finite."
                );
            }
            if (!std::isfinite(model.volatility)
                || !(model.volatility > 0.0f)) {
                throw std::invalid_argument(
                    prefix + "volatility must be finite and positive."
                );
            }
            if (!std::isfinite(model.jump_intensity)
                || model.jump_intensity < 0.0f) {
                throw std::invalid_argument(
                    prefix + "jump_intensity must be finite and non-negative."
                );
            }
            if (!std::isfinite(model.jump_log_mean)) {
                throw std::invalid_argument(
                    prefix + "jump_log_mean must be finite."
                );
            }
            if (!std::isfinite(model.jump_log_volatility)
                || model.jump_log_volatility < 0.0f) {
                throw std::invalid_argument(
                    prefix
                    + "jump_log_volatility must be finite and non-negative."
                );
            }
            return model;
    });
}

}  // namespace ai_factory::workbench::model::equity::merton
