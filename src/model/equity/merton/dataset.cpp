// Host implementation of the Merton dataset loader.
#include "model/equity/merton/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace ai_factory::workbench::merton {

// Parse and validate Merton rows while preserving their dataset order.
std::vector<MertonModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    std::ifstream stream(dataset_path);
    if (!stream) {
        throw std::runtime_error(
            "Could not open Merton JSON: " + dataset_path.string()
        );
    }

    nlohmann::json document;
    try {
        stream >> document;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Invalid Merton JSON '" + dataset_path.string()
            + "': " + error.what()
        );
    }

    datasets::validate_model_dataset(document);
    const auto& rows = document.at("models");
    std::vector<MertonModelParameters> models;
    models.reserve(rows.size());
    for (const auto& row : rows) {
        const std::string row_id = row.at("id").get<std::string>();
        const auto& parameters = row.at("parameters");
        const MertonModelParameters model = {
            parameters.at("spot").get<float>(),
            parameters.at("risk_free_rate").get<float>(),
            parameters.at("dividend_yield").get<float>(),
            parameters.at("volatility").get<float>(),
            parameters.at("jump_intensity").get<float>(),
            parameters.at("jump_log_mean").get<float>(),
            parameters.at("jump_log_volatility").get<float>(),
        };
        const std::string prefix =
            "Merton model row id '" + row_id + "': ";
        if (!std::isfinite(model.spot) || !(model.spot > 0.0f)) {
            throw std::invalid_argument(
                prefix + "spot must be finite and positive."
            );
        }
        if (!std::isfinite(model.risk_free_rate)) {
            throw std::invalid_argument(
                prefix + "risk_free_rate must be finite."
            );
        }
        if (!std::isfinite(model.dividend_yield)) {
            throw std::invalid_argument(
                prefix + "dividend_yield must be finite."
            );
        }
        if (!std::isfinite(model.volatility)
            || !(model.volatility > 0.0f)) {
            throw std::invalid_argument(
                prefix + "volatility must be finite and positive."
            );
        }
        if (!std::isfinite(model.jump_intensity)
            || model.jump_intensity < 0.0f) {
            throw std::invalid_argument(
                prefix + "jump_intensity must be finite and non-negative."
            );
        }
        if (!std::isfinite(model.jump_log_mean)) {
            throw std::invalid_argument(
                prefix + "jump_log_mean must be finite."
            );
        }
        if (!std::isfinite(model.jump_log_volatility)
            || model.jump_log_volatility < 0.0f) {
            throw std::invalid_argument(
                prefix
                + "jump_log_volatility must be finite and non-negative."
            );
        }
        models.push_back(model);
    }
    return models;
}

}  // namespace ai_factory::workbench::merton
