// Host implementation of the Ornstein-Uhlenbeck dataset loader.
#include "model/fixed_income/ornstein_uhlenbeck/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace ai_factory::workbench::model::ornstein_uhlenbeck {

// Parse and validate OU rows while preserving their dataset order.
std::vector<ModelParameters> load_models(
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

    datasets::validate_model_dataset(document);
    const auto& rows = document.at("models");
    std::vector<ModelParameters> models;
    models.reserve(rows.size());
    for (const auto& row : rows) {
        const std::string row_id = row.at("id").get<std::string>();
        const auto& parameters = row.at("parameters");
        const ModelParameters model = {
            {
                parameters.at("mean_reversion").get<float>(),
                parameters.at("volatility").get<float>(),
            },
            parameters.at("initial_state").get<float>(),
        };
        const std::string prefix =
            "Ornstein-Uhlenbeck model row id '" + row_id + "': ";
        if (!std::isfinite(model.process.mean_reversion)
            || !(model.process.mean_reversion > 0.0f)) {
            throw std::invalid_argument(
                prefix + "mean_reversion must be finite and positive."
            );
        }
        if (!std::isfinite(model.process.volatility)
            || !(model.process.volatility >= 0.0f)) {
            throw std::invalid_argument(
                prefix + "volatility must be finite and non-negative."
            );
        }
        if (!std::isfinite(model.initial_state))
            throw std::invalid_argument(prefix + "initial_state must be finite.");
        models.push_back(model);
    }
    return models;
}

}  // namespace ai_factory::workbench::model::ornstein_uhlenbeck
