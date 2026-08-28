// JSON loading and validation for log-modulated rough Bergomi parameter datasets.
#include "model/equity/rough/log_modulated_rough_bergomi/dataset.hpp"

#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::model::equity::log_modulated_rough_bergomi {

std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<ModelParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Model,
        "log-modulated rough-Bergomi model",
        [](const nlohmann::json& row, const std::string& prefix) {
            const ModelParameters model{
                row.at("spot").get<float>(),
                row.at("risk_free_rate").get<float>(),
                row.at("dividend_yield").get<float>(),
                row.at("xi_0").get<float>(),
                row.at("eta").get<float>(),
                row.at("hurst_exponent").get<float>(),
                row.at("rho").get<float>(),
                row.at("log_modulation_scale").get<float>(),
                row.at("log_modulation_power").get<float>(),
            };
            const auto finite = [](float value) { return std::isfinite(value); };
            if (!finite(model.spot) || model.spot <= 0.0f
                || !finite(model.xi_0) || model.xi_0 <= 0.0f
                || !finite(model.eta) || model.eta < 0.0f
                || !finite(model.hurst_exponent)
                || model.hurst_exponent < 0.0f
                || model.hurst_exponent >= 0.5f
                || !finite(model.rho) || model.rho < -1.0f || model.rho > 1.0f
                || !finite(model.log_modulation_scale)
                || model.log_modulation_scale <= 0.0f
                || !finite(model.log_modulation_power)
                || model.log_modulation_power <= 1.0f
                || !finite(model.risk_free_rate)
                || !finite(model.dividend_yield)) {
                throw std::invalid_argument(prefix + "invalid model parameters.");
            }
            return model;
        }
    );
}

}  // namespace ai_factory::workbench::model::equity::log_modulated_rough_bergomi
