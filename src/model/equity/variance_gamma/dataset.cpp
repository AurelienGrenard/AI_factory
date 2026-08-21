// Host implementation of the Variance-Gamma dataset loader.
#include "model/equity/variance_gamma/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace ai_factory::workbench::variance_gamma {

std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    std::ifstream stream(dataset_path);
    if (!stream) {
        throw std::runtime_error(
            "Could not open Variance-Gamma JSON: " + dataset_path.string()
        );
    }

    nlohmann::json document;
    try {
        stream >> document;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Invalid Variance-Gamma JSON '" + dataset_path.string()
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
            parameters.at("spot").get<float>(),
            parameters.at("risk_free_rate").get<float>(),
            parameters.at("dividend_yield").get<float>(),
            parameters.at("sigma").get<float>(),
            parameters.at("nu").get<float>(),
            parameters.at("theta").get<float>(),
        };
        const std::string prefix =
            "Variance-Gamma model row id '" + row_id + "': ";
        if (!std::isfinite(model.spot) || !(model.spot > 0.0f))
            throw std::invalid_argument(prefix + "spot must be finite and positive.");
        if (!std::isfinite(model.risk_free_rate))
            throw std::invalid_argument(prefix + "risk_free_rate must be finite.");
        if (!std::isfinite(model.dividend_yield))
            throw std::invalid_argument(prefix + "dividend_yield must be finite.");
        if (!std::isfinite(model.sigma) || !(model.sigma > 0.0f))
            throw std::invalid_argument(prefix + "sigma must be finite and positive.");
        if (!std::isfinite(model.nu) || !(model.nu > 0.0f))
            throw std::invalid_argument(prefix + "nu must be finite and positive.");
        if (!std::isfinite(model.theta))
            throw std::invalid_argument(prefix + "theta must be finite.");
        const float exponential_moment = 1.0f
            - model.theta * model.nu
            - 0.5f * model.sigma * model.sigma * model.nu;
        if (!std::isfinite(exponential_moment)
            || !(exponential_moment > 0.0f)) {
            throw std::invalid_argument(
                prefix
                + "1 - theta * nu - 0.5 * sigma^2 * nu must be positive."
            );
        }
        models.push_back(model);
    }
    return models;
}

}  // namespace ai_factory::workbench::variance_gamma
