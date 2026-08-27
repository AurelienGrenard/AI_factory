// Host implementation of the G2 dataset loader.
#include "model/fixed_income/g2/dataset.hpp"
#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::model::fixed_income::g2 {

// Parse and validate G2 rows while preserving their dataset order.
std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<ModelParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Model,
        "G2 model",
        [&](const nlohmann::json& values, const std::string& prefix) {
            const ModelParameters model = {
                {
                    values.at("mean_reversion_x").get<float>(),
                    values.at("volatility_x").get<float>(),
                    values.at("mean_reversion_y").get<float>(),
                    values.at("volatility_y").get<float>(),
                    values.at("correlation").get<float>(),
                },
                {
                    values.at("initial_state_x").get<float>(),
                    values.at("initial_state_y").get<float>(),
                },
            };
            if (!std::isfinite(model.process.mean_reversion_x)
                || !(model.process.mean_reversion_x > 0.0f)) {
                throw std::invalid_argument(
                    prefix + "mean_reversion_x must be finite and positive."
                );
            }
            if (!std::isfinite(model.process.mean_reversion_y)
                || !(model.process.mean_reversion_y > 0.0f)) {
                throw std::invalid_argument(
                    prefix + "mean_reversion_y must be finite and positive."
                );
            }
            if (!std::isfinite(model.process.volatility_x)
                || !(model.process.volatility_x >= 0.0f)) {
                throw std::invalid_argument(
                    prefix + "volatility_x must be finite and non-negative."
                );
            }
            if (!std::isfinite(model.process.volatility_y)
                || !(model.process.volatility_y >= 0.0f)) {
                throw std::invalid_argument(
                    prefix + "volatility_y must be finite and non-negative."
                );
            }
            if (!std::isfinite(model.process.correlation)
                || model.process.correlation < -1.0f
                || model.process.correlation > 1.0f) {
                throw std::invalid_argument(
                    prefix + "correlation must be finite and lie in [-1, 1]."
                );
            }
            if (!std::isfinite(model.initial_state.state_x)
                || !std::isfinite(model.initial_state.state_y)) {
                throw std::invalid_argument(
                    prefix + "initial states must be finite."
                );
            }
            return model;
    });
}

}  // namespace ai_factory::workbench::model::fixed_income::g2
