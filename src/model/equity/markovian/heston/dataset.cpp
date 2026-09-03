// Host implementation of the Heston dataset loader.
#include "model/equity/markovian/heston/dataset.hpp"
#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::model::equity::heston {

// Parse and validate Heston rows while preserving their dataset order.
std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<ModelParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Model,
        "Heston model",
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
            return model;
    });
}

}  // namespace ai_factory::workbench::model::equity::heston
