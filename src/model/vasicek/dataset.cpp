// Host implementation of the Vasicek dataset loader.
#include "model/vasicek/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace ai_factory::workbench::model::vasicek {

// Parse and validate Vasicek rows while preserving their dataset order.
std::vector<VasicekModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    std::ifstream stream(dataset_path);
    if (!stream) {
        throw std::runtime_error(
            "Could not open Vasicek JSON: "
            + dataset_path.string()
        );
    }

    nlohmann::json document;
    try {
        stream >> document;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Invalid Vasicek JSON '" + dataset_path.string()
            + "': " + error.what()
        );
    }

    datasets::validate_model_dataset(document);
    const auto& rows = document.at("models");
    std::vector<VasicekModelParameters> models;
    models.reserve(rows.size());
    for (const auto& row : rows) {
        const std::string row_id = row.at("id").get<std::string>();
        const auto& parameters = row.at("parameters");
        const VasicekModelParameters model = {
            {
                parameters.at("mean_reversion").get<float>(),
                parameters.at("long_term_mean").get<float>(),
                parameters.at("volatility").get<float>(),
            },
            parameters.at("initial_state").get<float>(),
        };
        const std::string prefix =
            "Vasicek model row id '" + row_id + "': ";
        if (!std::isfinite(model.process.mean_reversion)
            || !(model.process.mean_reversion > 0.0f)) {
            throw std::invalid_argument(
                prefix + "mean_reversion must be finite and positive."
            );
        }
        if (!std::isfinite(model.process.long_term_mean)) {
            throw std::invalid_argument(
                prefix + "long_term_mean must be finite."
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

}  // namespace ai_factory::workbench::model::vasicek
