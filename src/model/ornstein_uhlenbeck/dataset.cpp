// Host implementation of the Ornstein-Uhlenbeck dataset loader.
#include "model/ornstein_uhlenbeck/dataset.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace ai_factory::workbench::model::ornstein_uhlenbeck {

// Parse and validate OU rows while preserving their dataset order.
std::vector<OrnsteinUhlenbeckModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    std::ifstream stream(dataset_path);
    if (!stream) {
        throw std::runtime_error(
            "Could not open Ornstein-Uhlenbeck JSON: "
            + dataset_path.string()
        );
    }

    nlohmann::json document;
    try {
        stream >> document;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Invalid Ornstein-Uhlenbeck JSON '" + dataset_path.string()
            + "': " + error.what()
        );
    }

    const auto& rows = document.at("models");
    std::vector<OrnsteinUhlenbeckModelParameters> models;
    models.reserve(rows.size());
    for (const auto& row : rows) {
        const auto& parameters = row.at("parameters");
        const OrnsteinUhlenbeckModelParameters model = {
            {
                parameters.at("mean_reversion").get<float>(),
                parameters.at("volatility").get<float>(),
            },
            parameters.at("initial_factor").get<float>(),
        };
        if (!std::isfinite(model.dynamics.mean_reversion)
            || !std::isfinite(model.dynamics.volatility)
            || !std::isfinite(model.initial_factor)
            || !(model.dynamics.mean_reversion > 0.0f)
            || !(model.dynamics.volatility >= 0.0f)) {
            throw std::invalid_argument(
                "Invalid Ornstein-Uhlenbeck model parameters."
            );
        }
        models.push_back(model);
    }
    return models;
}

}  // namespace ai_factory::workbench::model::ornstein_uhlenbeck
