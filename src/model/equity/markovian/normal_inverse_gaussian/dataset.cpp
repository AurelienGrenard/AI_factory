// Host implementation of the Normal-Inverse-Gaussian dataset loader.
#include "model/equity/markovian/normal_inverse_gaussian/dataset.hpp"
#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::model::equity::normal_inverse_gaussian {

std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<ModelParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Model,
        "Normal-Inverse-Gaussian model",
        [&](const nlohmann::json& parameters, const std::string& prefix) {
            const ModelParameters model = {
                parameters.at("spot").get<float>(),
                parameters.at("risk_free_rate").get<float>(),
                parameters.at("dividend_yield").get<float>(),
                parameters.at("alpha").get<float>(),
                parameters.at("beta").get<float>(),
                parameters.at("delta").get<float>(),
            };
            if (!std::isfinite(model.spot) || !(model.spot > 0.0f))
                throw std::invalid_argument(prefix + "spot must be finite and positive.");
            if (!std::isfinite(model.risk_free_rate))
                throw std::invalid_argument(prefix + "risk_free_rate must be finite.");
            if (!std::isfinite(model.dividend_yield))
                throw std::invalid_argument(prefix + "dividend_yield must be finite.");
            if (!std::isfinite(model.alpha) || !(model.alpha > 0.0f))
                throw std::invalid_argument(prefix + "alpha must be finite and positive.");
            if (!std::isfinite(model.beta))
                throw std::invalid_argument(prefix + "beta must be finite.");
            if (!std::isfinite(model.delta) || !(model.delta > 0.0f))
                throw std::invalid_argument(prefix + "delta must be finite and positive.");
            if (!(model.alpha > std::fabs(model.beta + 1.0f))) {
                throw std::invalid_argument(
                    prefix + "alpha must exceed abs(beta + 1) for the martingale moment."
                );
            }
            return model;
    });
}

}  // namespace ai_factory::workbench::model::equity::normal_inverse_gaussian
