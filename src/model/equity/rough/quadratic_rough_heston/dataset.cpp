#include "model/equity/rough/quadratic_rough_heston/dataset.hpp"

#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::model::equity::quadratic_rough_heston {

std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<ModelParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Model,
        "quadratic rough-Heston model",
        [](const nlohmann::json& row, const std::string& prefix) {
            const ModelParameters model{
                row.at("spot").get<float>(),
                row.at("risk_free_rate").get<float>(),
                row.at("dividend_yield").get<float>(),
                row.at("initial_feedback").get<float>(),
                row.at("quadratic_scale").get<float>(),
                row.at("quadratic_shift").get<float>(),
                row.at("variance_floor").get<float>(),
                row.at("feedback_rate").get<float>(),
                row.at("feedback_volatility").get<float>(),
                row.at("hurst_exponent").get<float>(),
            };
            const auto finite = [](float value) { return std::isfinite(value); };
            if (!finite(model.spot) || model.spot <= 0.0f
                || !finite(model.initial_feedback)
                || !finite(model.quadratic_scale) || model.quadratic_scale <= 0.0f
                || !finite(model.quadratic_shift)
                || !finite(model.variance_floor) || model.variance_floor <= 0.0f
                || !finite(model.feedback_rate) || model.feedback_rate <= 0.0f
                || !finite(model.feedback_volatility)
                || model.feedback_volatility <= 0.0f
                || !finite(model.hurst_exponent)
                || model.hurst_exponent <= 0.0f
                || model.hurst_exponent >= 0.5f
                || !finite(model.risk_free_rate)
                || !finite(model.dividend_yield)) {
                throw std::invalid_argument(prefix + "invalid model parameters.");
            }
            return model;
        }
    );
}

}  // namespace ai_factory::workbench::model::equity::quadratic_rough_heston
