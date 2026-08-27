// Host implementation of the rough-Heston dataset loader.
#include "model/equity/rough/rough_heston/dataset.hpp"
#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::model::equity::rough_heston {

std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<ModelParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Model,
        "rough-Heston model",
        [&](const nlohmann::json& parameters, const std::string& prefix) {
            const ModelParameters model = {
                parameters.at("spot").get<float>(),
                parameters.at("risk_free_rate").get<float>(),
                parameters.at("dividend_yield").get<float>(),
                parameters.at("initial_variance").get<float>(),
                parameters.at("mean_reversion").get<float>(),
                parameters.at("variance_drift").get<float>(),
                parameters.at("volatility_of_variance").get<float>(),
                parameters.at("hurst_exponent").get<float>(),
                parameters.at("rho").get<float>(),
            };
            if (!std::isfinite(model.spot) || !(model.spot > 0.0f))
                throw std::invalid_argument(prefix + "spot must be positive.");
            if (!std::isfinite(model.risk_free_rate)
                || !std::isfinite(model.dividend_yield)) {
                throw std::invalid_argument(prefix + "rates must be finite.");
            }
            if (!std::isfinite(model.initial_variance)
                || !(model.initial_variance > 0.0f)) {
                throw std::invalid_argument(
                    prefix + "initial_variance must be positive."
                );
            }
            if (!std::isfinite(model.mean_reversion)
                || !(model.mean_reversion >= 0.0f)
                || !std::isfinite(model.variance_drift)
                || !(model.variance_drift >= 0.0f)) {
                throw std::invalid_argument(
                    prefix + "variance drift coefficients must be non-negative."
                );
            }
            if (!std::isfinite(model.volatility_of_variance)
                || !(model.volatility_of_variance > 0.0f)) {
                throw std::invalid_argument(
                    prefix + "volatility_of_variance must be positive."
                );
            }
            if (!std::isfinite(model.hurst_exponent)
                || !(model.hurst_exponent > 0.0f
                     && model.hurst_exponent < 0.5f)) {
                throw std::invalid_argument(
                    prefix + "hurst_exponent must lie in (0, 0.5)."
                );
            }
            if (!std::isfinite(model.rho)
                || !(model.rho >= -1.0f && model.rho <= 1.0f)) {
                throw std::invalid_argument(prefix + "rho must lie in [-1, 1].");
            }
            return model;
    });
}

}  // namespace ai_factory::workbench::model::equity::rough_heston
