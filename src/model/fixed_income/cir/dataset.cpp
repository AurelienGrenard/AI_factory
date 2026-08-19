// Host implementation of the CIR dataset loader.
#include "model/fixed_income/cir/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace ai_factory::workbench::model::cir {

// Parse and validate CIR rows while preserving their dataset order.
std::vector<CirModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    std::ifstream stream(dataset_path);
    if (!stream) {
        throw std::runtime_error(
            "Could not open CIR JSON: " + dataset_path.string()
        );
    }

    nlohmann::json document;
    try {
        stream >> document;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Invalid CIR JSON '" + dataset_path.string()
            + "': " + error.what()
        );
    }

    datasets::validate_model_dataset(document);
    const auto& rows = document.at("models");
    std::vector<CirModelParameters> models;
    models.reserve(rows.size());
    for (const auto& row : rows) {
        const std::string row_id = row.at("id").get<std::string>();
        const auto& parameters = row.at("parameters");
        const CirModelParameters model = {
            {
                parameters.at("mean_reversion").get<float>(),
                parameters.at("long_term_mean").get<float>(),
                parameters.at("volatility").get<float>(),
            },
            parameters.at("initial_state").get<float>(),
        };
        const std::string prefix = "CIR model row id '" + row_id + "': ";
        if (!std::isfinite(model.process.mean_reversion)
            || !(model.process.mean_reversion > 0.0f)) {
            throw std::invalid_argument(
                prefix + "mean_reversion must be finite and positive."
            );
        }
        if (!std::isfinite(model.process.long_term_mean)
            || !(model.process.long_term_mean > 0.0f)) {
            throw std::invalid_argument(
                prefix + "long_term_mean must be finite and positive."
            );
        }
        if (!std::isfinite(model.process.volatility)
            || !(model.process.volatility > 0.0f)) {
            throw std::invalid_argument(
                prefix + "volatility must be finite and positive."
            );
        }
        if (!std::isfinite(model.initial_state)
            || !(model.initial_state >= 0.0f)) {
            throw std::invalid_argument(
                prefix + "initial_state must be finite and non-negative."
            );
        }
        models.push_back(model);
    }
    return models;
}

}  // namespace ai_factory::workbench::model::cir
