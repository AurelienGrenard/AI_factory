// Host implementation of the Kou dataset loader.
#include "model/equity/markovian/kou/dataset.hpp"
#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::model::equity::kou {

// Parse and validate Kou rows while preserving their dataset order.
std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<ModelParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Model,
        "Kou model",
        [&](const nlohmann::json& parameters, const std::string& prefix) {
            const ModelParameters model = {
                parameters.at("spot").get<float>(),
                parameters.at("risk_free_rate").get<float>(),
                parameters.at("dividend_yield").get<float>(),
                parameters.at("volatility").get<float>(),
                parameters.at("jump_intensity").get<float>(),
                parameters.at("up_probability").get<float>(),
                parameters.at("positive_jump_rate").get<float>(),
                parameters.at("negative_jump_rate").get<float>(),
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
            if (!std::isfinite(model.up_probability)
                || !(model.up_probability > 0.0f
                     && model.up_probability < 1.0f)) {
                throw std::invalid_argument(
                    prefix
                    + "up_probability must lie strictly between zero and one."
                );
            }
            if (!std::isfinite(model.positive_jump_rate)
                || !(model.positive_jump_rate > 2.0f)) {
                throw std::invalid_argument(
                    prefix
                    + "positive_jump_rate must exceed two for finite Monte Carlo "
                      "payoff variance."
                );
            }
            if (!std::isfinite(model.negative_jump_rate)
                || !(model.negative_jump_rate > 0.0f)) {
                throw std::invalid_argument(
                    prefix + "negative_jump_rate must be finite and positive."
                );
            }
            return model;
    });
}

}  // namespace ai_factory::workbench::model::equity::kou
