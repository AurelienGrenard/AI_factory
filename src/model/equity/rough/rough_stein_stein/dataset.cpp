#include "model/equity/rough/rough_stein_stein/dataset.hpp"

#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::model::equity::rough_stein_stein {

std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<ModelParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Model,
        "rough Stein-Stein model",
        [](const nlohmann::json& row, const std::string& prefix) {
            const ModelParameters model{
                row.at("spot").get<float>(),
                row.at("risk_free_rate").get<float>(),
                row.at("dividend_yield").get<float>(),
                row.at("volatility_level").get<float>(),
                row.at("mean_reversion").get<float>(),
                row.at("volatility_of_volatility").get<float>(),
                row.at("hurst_exponent").get<float>(),
                row.at("rho").get<float>(),
            };
            const auto finite = [](float value) { return std::isfinite(value); };
            if (!finite(model.spot) || model.spot <= 0.0f
                || !finite(model.volatility_level)
                || model.volatility_level < 0.0f
                || !finite(model.mean_reversion)
                || model.mean_reversion < 0.0f
                || !finite(model.volatility_of_volatility)
                || model.volatility_of_volatility <= 0.0f
                || !finite(model.hurst_exponent)
                || model.hurst_exponent <= 0.0f
                || model.hurst_exponent >= 0.5f
                || !finite(model.rho) || model.rho < -1.0f || model.rho > 1.0f
                || !finite(model.risk_free_rate)
                || !finite(model.dividend_yield)) {
                throw std::invalid_argument(prefix + "invalid model parameters.");
            }
            return model;
        }
    );
}

}  // namespace ai_factory::workbench::model::equity::rough_stein_stein
