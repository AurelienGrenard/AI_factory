// JSON loading and validation for SABR parameter datasets.
#include "model/equity/markovian/sabr/dataset.hpp"

#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::model::equity::sabr {

std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<ModelParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Model,
        "SABR equity model",
        [](const nlohmann::json& row, const std::string& prefix) {
            const ModelParameters model{
                row.at("spot").get<float>(),
                row.at("risk_free_rate").get<float>(),
                row.at("dividend_yield").get<float>(),
                row.at("initial_volatility").get<float>(),
                row.at("volatility_of_volatility").get<float>(),
                row.at("rho").get<float>(),
                row.at("beta").get<float>(),
            };
            const auto finite = [](float value) { return std::isfinite(value); };
            if (!finite(model.spot) || model.spot <= 0.0f
                || !finite(model.initial_volatility)
                || model.initial_volatility <= 0.0f
                || !finite(model.volatility_of_volatility)
                || model.volatility_of_volatility < 0.0f
                || !finite(model.rho) || model.rho < -1.0f || model.rho > 1.0f
                || !finite(model.beta) || model.beta < 0.0f || model.beta > 1.0f
                || !finite(model.risk_free_rate)
                || !finite(model.dividend_yield)) {
                throw std::invalid_argument(prefix + "invalid model parameters.");
            }
            return model;
        }
    );
}

}  // namespace ai_factory::workbench::model::equity::sabr
