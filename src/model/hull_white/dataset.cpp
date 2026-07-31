// Host implementation of the Hull-White one-factor dataset loader.
#include "model/hull_white/dataset.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace ai_factory::workbench::hull_white {

// Parse and validate model rows while preserving their dataset order.
std::vector<HullWhiteModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    std::ifstream stream(dataset_path);
    if (!stream) {
        throw std::runtime_error(
            "Could not open Hull-White JSON: " + dataset_path.string()
        );
    }

    nlohmann::json document;
    try {
        stream >> document;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Invalid Hull-White JSON '" + dataset_path.string()
            + "': " + error.what()
        );
    }

    const auto& rows = document.at("models");
    std::vector<HullWhiteModelParameters> models;
    models.reserve(rows.size());
    for (const auto& row : rows) {
        const auto& values = row.at("parameters");
        const HullWhiteModelParameters parameters = {
            values.at("mean_reversion").get<float>(),
            values.at("volatility").get<float>(),
        };
        if (!std::isfinite(parameters.mean_reversion)
            || !std::isfinite(parameters.volatility)
            || !(parameters.mean_reversion > 0.0f)
            || !(parameters.volatility > 0.0f)) {
            throw std::invalid_argument(
                "Invalid Hull-White one-factor parameters."
            );
        }
        models.push_back(parameters);
    }
    return models;
}

}  // namespace ai_factory::workbench::hull_white
