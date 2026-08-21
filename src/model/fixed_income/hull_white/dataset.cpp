// Host implementation of the Hull-White one-factor dataset loader.
#include "model/fixed_income/hull_white/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace ai_factory::workbench::model::hull_white {

// Parse and validate model rows while preserving their dataset order.
std::vector<ModelParameters> load_models(
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

    datasets::validate_model_dataset(document);
    const auto& rows = document.at("models");
    std::vector<ModelParameters> models;
    models.reserve(rows.size());
    for (const auto& row : rows) {
        const std::string row_id = row.at("id").get<std::string>();
        const auto& values = row.at("parameters");
        const ModelParameters parameters = {
            values.at("mean_reversion").get<float>(),
            values.at("volatility").get<float>(),
        };
        const std::string prefix =
            "Hull-White model row id '" + row_id + "': ";
        if (!std::isfinite(parameters.mean_reversion)
            || !(parameters.mean_reversion > 0.0f)) {
            throw std::invalid_argument(
                prefix + "mean_reversion must be finite and positive."
            );
        }
        if (!std::isfinite(parameters.volatility)
            || !(parameters.volatility >= 0.0f)) {
            throw std::invalid_argument(
                prefix + "volatility must be finite and non-negative."
            );
        }
        models.push_back(parameters);
    }
    return models;
}

}  // namespace ai_factory::workbench::model::hull_white
