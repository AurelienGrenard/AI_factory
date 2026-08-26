// Host implementation of the Normal-Inverse-Gaussian dataset loader.
#include "model/equity/normal_inverse_gaussian/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace ai_factory::workbench::model::equity::normal_inverse_gaussian {

std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    std::ifstream stream(dataset_path);
    if (!stream) {
        throw std::runtime_error(
            "Could not open Normal-Inverse-Gaussian JSON: "
            + dataset_path.string()
        );
    }

    nlohmann::json document;
    try {
        stream >> document;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Invalid Normal-Inverse-Gaussian JSON '"
            + dataset_path.string() + "': " + error.what()
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
            parameters.at("spot").get<float>(),
            parameters.at("risk_free_rate").get<float>(),
            parameters.at("dividend_yield").get<float>(),
            parameters.at("alpha").get<float>(),
            parameters.at("beta").get<float>(),
            parameters.at("delta").get<float>(),
        };
        const std::string prefix =
            "Normal-Inverse-Gaussian model row id '" + row_id + "': ";
        if (!std::isfinite(model.spot) || !(model.spot > 0.0f))
            throw std::invalid_argument(prefix + "spot must be finite and positive.");
        if (!std::isfinite(model.risk_free_rate))
            throw std::invalid_argument(prefix + "risk_free_rate must be finite.");
        if (!std::isfinite(model.dividend_yield))
            throw std::invalid_argument(prefix + "dividend_yield must be finite.");
        if (!std::isfinite(model.alpha) || !(model.alpha > 0.0f))
            throw std::invalid_argument(prefix + "alpha must be finite and positive.");
        if (!std::isfinite(model.beta))
            throw std::invalid_argument(prefix + "beta must be finite.");
        if (!std::isfinite(model.delta) || !(model.delta > 0.0f))
            throw std::invalid_argument(prefix + "delta must be finite and positive.");
        if (!(model.alpha > std::fabs(model.beta + 1.0f))) {
            throw std::invalid_argument(
                prefix + "alpha must exceed abs(beta + 1) for the martingale moment."
            );
        }
        models.push_back(model);
    }
    return models;
}

}  // namespace ai_factory::workbench::model::equity::normal_inverse_gaussian
