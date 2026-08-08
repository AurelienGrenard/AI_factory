// Host implementation of the Heston dataset loader.
#include "model/equity/heston/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace ai_factory::workbench::heston {

// Parse and validate Heston rows while preserving their dataset order.
std::vector<HestonModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    std::ifstream stream(dataset_path);
    if (!stream) {
        throw std::runtime_error(
            "Could not open Heston JSON: " + dataset_path.string()
        );
    }

    nlohmann::json document;
    try {
        stream >> document;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Invalid Heston JSON '" + dataset_path.string()
            + "': " + error.what()
        );
    }

    datasets::validate_model_dataset(document);
    const auto& rows = document.at("models");
    std::vector<HestonModelParameters> models;
    models.reserve(rows.size());
    // Keep only the compact FP32 parameters needed by CUDA.
    for (const auto& row : rows) {
        const std::string row_id = row.at("id").get<std::string>();
        const auto& parameters = row.at("parameters");
        const HestonModelParameters model = {
            parameters.at("spot").get<float>(),
            parameters.at("risk_free_rate").get<float>(),
            parameters.at("dividend_yield").get<float>(),
            parameters.at("initial_variance").get<float>(),
            parameters.at("kappa").get<float>(),
            parameters.at("theta").get<float>(),
            parameters.at("gamma").get<float>(),
            parameters.at("rho").get<float>(),
        };
        const std::string prefix =
            "Heston model row id '" + row_id + "': ";
        if (!std::isfinite(model.spot) || !(model.spot > 0.0f))
            throw std::invalid_argument(prefix + "spot must be finite and positive.");
        if (!std::isfinite(model.risk_free_rate))
            throw std::invalid_argument(prefix + "risk_free_rate must be finite.");
        if (!std::isfinite(model.dividend_yield))
            throw std::invalid_argument(prefix + "dividend_yield must be finite.");
        if (!std::isfinite(model.initial_variance)
            || !(model.initial_variance >= 0.0f)) {
            throw std::invalid_argument(
                prefix + "initial_variance must be finite and non-negative."
            );
        }
        if (!std::isfinite(model.kappa) || !(model.kappa > 0.0f))
            throw std::invalid_argument(prefix + "kappa must be finite and positive.");
        if (!std::isfinite(model.theta) || !(model.theta > 0.0f))
            throw std::invalid_argument(prefix + "theta must be finite and positive.");
        if (!std::isfinite(model.gamma) || !(model.gamma > 0.0f))
            throw std::invalid_argument(prefix + "gamma must be finite and positive.");
        if (!std::isfinite(model.rho)
            || !(model.rho >= -1.0f && model.rho <= 1.0f)) {
            throw std::invalid_argument(
                prefix + "rho must be finite and lie in [-1, 1]."
            );
        }
        models.push_back(model);
    }
    return models;
}

}  // namespace ai_factory::workbench::heston
