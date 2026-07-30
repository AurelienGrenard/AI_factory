// Host implementation of the Heston dataset loader.
#include "heston/parameters.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace ai_factory::workbench::heston {

// Parse and validate Heston rows while preserving their dataset order.
std::vector<HestonModelParameters> load_heston(
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

    const auto& rows = document.at("models");
    std::vector<HestonModelParameters> models;
    models.reserve(rows.size());
    // Keep only the compact FP32 parameters needed by CUDA.
    for (const auto& row : rows) {
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
        if (!std::isfinite(model.spot)
            || !std::isfinite(model.risk_free_rate)
            || !std::isfinite(model.dividend_yield)
            || !std::isfinite(model.initial_variance)
            || !std::isfinite(model.kappa)
            || !std::isfinite(model.theta)
            || !std::isfinite(model.gamma)
            || !std::isfinite(model.rho)
            || !(model.spot > 0.0f) || !(model.initial_variance >= 0.0f)
            || !(model.kappa > 0.0f) || !(model.theta > 0.0f)
            || !(model.gamma > 0.0f)
            || !(model.rho >= -1.0f && model.rho <= 1.0f)) {
            throw std::invalid_argument("Invalid Heston model parameters.");
        }
        models.push_back(model);
    }
    return models;
}

}  // namespace ai_factory::workbench::heston
