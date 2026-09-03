// Host implementation of the rough-SABR dataset loader.
#include "model/equity/rough/rough_sabr/dataset.hpp"
#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::model::equity::rough_sabr {

std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<ModelParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Model,
        "rough-SABR model",
        [&](const nlohmann::json& parameters, const std::string& prefix) {
            const ModelParameters model = {
                parameters.at("spot").get<float>(),
                parameters.at("risk_free_rate").get<float>(),
                parameters.at("dividend_yield").get<float>(),
                parameters.at("xi_0").get<float>(),
                parameters.at("eta").get<float>(),
                parameters.at("hurst_exponent").get<float>(),
                parameters.at("rho").get<float>(),
                parameters.at("beta").get<float>(),
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
            if (!std::isfinite(model.xi_0) || !(model.xi_0 > 0.0f)) {
                throw std::invalid_argument(
                    prefix + "xi_0 must be finite and positive."
                );
            }
            if (!std::isfinite(model.eta) || model.eta < 0.0f) {
                throw std::invalid_argument(
                    prefix + "eta must be finite and non-negative."
                );
            }
            if (!std::isfinite(model.hurst_exponent)
                || !(model.hurst_exponent > 0.0f
                     && model.hurst_exponent < 0.5f)) {
                throw std::invalid_argument(
                    prefix
                    + "hurst_exponent must lie strictly between 0 and 0.5."
                );
            }
            if (!std::isfinite(model.rho)
                || !(model.rho >= -1.0f && model.rho <= 1.0f)) {
                throw std::invalid_argument(
                    prefix + "rho must lie in [-1, 1]."
                );
            }
            if (!std::isfinite(model.beta)
                || !(model.beta >= 0.5f && model.beta <= 1.0f)) {
                throw std::invalid_argument(
                    prefix + "beta must lie in [0.5, 1]."
                );
            }
            return model;
    });
}

}  // namespace ai_factory::workbench::model::equity::rough_sabr
