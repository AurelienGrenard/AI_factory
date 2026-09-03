// Host implementation of the Bates dataset loader.
#include "model/equity/markovian/bates/dataset.hpp"
#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::model::equity::bates {

// Parse and validate Bates rows while preserving their dataset order.
std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<ModelParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Model,
        "Bates model",
        [&](const nlohmann::json& parameters, const std::string& prefix) {
            const ModelParameters model = {
                parameters.at("spot").get<float>(),
                parameters.at("risk_free_rate").get<float>(),
                parameters.at("dividend_yield").get<float>(),
                parameters.at("initial_variance").get<float>(),
                parameters.at("kappa").get<float>(),
                parameters.at("theta").get<float>(),
                parameters.at("gamma").get<float>(),
                parameters.at("rho").get<float>(),
                parameters.at("jump_intensity").get<float>(),
                parameters.at("jump_log_mean").get<float>(),
                parameters.at("jump_log_volatility").get<float>(),
            };
            if (!std::isfinite(model.spot) || !(model.spot > 0.0f))
                throw std::invalid_argument(prefix + "spot must be finite and positive.");
            if (!std::isfinite(model.risk_free_rate))
                throw std::invalid_argument(prefix + "risk_free_rate must be finite.");
            if (!std::isfinite(model.dividend_yield))
                throw std::invalid_argument(prefix + "dividend_yield must be finite.");
            if (!std::isfinite(model.initial_variance)
                || !(model.initial_variance >= 0.0f)) {
                throw std::invalid_argument(
                    prefix + "initial_variance must be finite and non-negative."
                );
            }
            if (!std::isfinite(model.kappa) || !(model.kappa > 0.0f))
                throw std::invalid_argument(prefix + "kappa must be finite and positive.");
            if (!std::isfinite(model.theta) || !(model.theta > 0.0f))
                throw std::invalid_argument(prefix + "theta must be finite and positive.");
            if (!std::isfinite(model.gamma) || !(model.gamma > 0.0f))
                throw std::invalid_argument(prefix + "gamma must be finite and positive.");
            if (!std::isfinite(model.rho)
                || !(model.rho >= -1.0f && model.rho <= 1.0f)) {
                throw std::invalid_argument(
                    prefix + "rho must be finite and lie in [-1, 1]."
                );
            }
            if (!std::isfinite(model.jump_intensity)
                || !(model.jump_intensity >= 0.0f)) {
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
                || !(model.jump_log_volatility >= 0.0f)) {
                throw std::invalid_argument(
                    prefix
                    + "jump_log_volatility must be finite and non-negative."
                );
            }
            return model;
    });
}

}  // namespace ai_factory::workbench::model::equity::bates
