// Host implementation of the G2 dataset loader.
#include "model/g2/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace ai_factory::workbench::model::g2 {

// Parse and validate G2 rows while preserving their dataset order.
std::vector<G2ModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    std::ifstream stream(dataset_path);
    if (!stream) {
        throw std::runtime_error(
            "Could not open G2 JSON: " + dataset_path.string()
        );
    }
    nlohmann::json document;
    try {
        stream >> document;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Invalid G2 JSON '" + dataset_path.string() + "': " + error.what()
        );
    }

    datasets::validate_model_dataset(document);
    const auto& rows = document.at("models");
    std::vector<G2ModelParameters> models;
    models.reserve(rows.size());
    for (const auto& row : rows) {
        const std::string row_id = row.at("id").get<std::string>();
        const auto& values = row.at("parameters");
        const G2ModelParameters model = {
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
        const std::string prefix = "G2 model row id '" + row_id + "': ";
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
        models.push_back(model);
    }
    return models;
}

}  // namespace ai_factory::workbench::model::g2
