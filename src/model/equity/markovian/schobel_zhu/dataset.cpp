// Host implementation of the Schobel-Zhu dataset loader.
#include "model/equity/markovian/schobel_zhu/dataset.hpp"
#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::model::equity::schobel_zhu {

// Parse and validate Schobel-Zhu rows while preserving their dataset order.
std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<ModelParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Model,
        "Schobel-Zhu model",
        [&](const nlohmann::json& parameters, const std::string& prefix) {
            const ModelParameters model = {
                parameters.at("spot").get<float>(),
                parameters.at("risk_free_rate").get<float>(),
                parameters.at("dividend_yield").get<float>(),
                parameters.at("initial_volatility").get<float>(),
                parameters.at("mean_reversion").get<float>(),
                parameters.at("long_run_volatility").get<float>(),
                parameters.at("volatility_of_volatility").get<float>(),
                parameters.at("correlation").get<float>(),
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
            if (!std::isfinite(model.initial_volatility)) {
                throw std::invalid_argument(
                    prefix + "initial_volatility must be finite."
                );
            }
            if (!std::isfinite(model.mean_reversion)
                || !(model.mean_reversion > 0.0f)) {
                throw std::invalid_argument(
                    prefix + "mean_reversion must be finite and positive."
                );
            }
            if (!std::isfinite(model.long_run_volatility)) {
                throw std::invalid_argument(
                    prefix + "long_run_volatility must be finite."
                );
            }
            if (!std::isfinite(model.volatility_of_volatility)
                || !(model.volatility_of_volatility > 0.0f)) {
                throw std::invalid_argument(
                    prefix
                    + "volatility_of_volatility must be finite and positive."
                );
            }
            if (!std::isfinite(model.correlation)
                || !(model.correlation > -1.0f
                     && model.correlation < 1.0f)) {
                throw std::invalid_argument(
                    prefix
                    + "correlation must lie strictly between -1 and 1."
                );
            }
            return model;
    });
}

}  // namespace ai_factory::workbench::model::equity::schobel_zhu
