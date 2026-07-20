// Host implementation of the Heston registry loader declared in common.hpp.
// JSON objects are temporary: only the contiguous FP32 model vector survives
// after this function returns.
#include "heston/common.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>

namespace ai_factory::workbench::heston {

// Parse one Heston database and preserve its row order in the returned vector.
std::vector<HestonModelInput> load_heston(
    const std::filesystem::path& json_path
) {
    std::ifstream stream(json_path);
    if (!stream) {
        throw std::runtime_error(
            "Could not open Heston JSON: " + json_path.string()
        );
    }

    nlohmann::json document;
    try {
        stream >> document;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Invalid Heston JSON '" + json_path.string() + "': " + error.what()
        );
    }

    const auto& rows = document.at("models");
    std::vector<HestonModelInput> models;
    models.reserve(rows.size());

    // Aggregate initialization fixes the in-memory field order explicitly.
    for (const auto& row : rows) {
        const auto& parameters = row.at("parameters");
        const HestonModelInput model = {
            parameters.at("spot").get<float>(),
            parameters.at("risk_free_rate").get<float>(),
            parameters.at("dividend_yield").get<float>(),
            parameters.at("initial_variance").get<float>(),
            parameters.at("kappa").get<float>(),
            parameters.at("theta").get<float>(),
            parameters.at("gamma").get<float>(),
            parameters.at("rho").get<float>(),
        };
        if (!(model.spot > 0.0f) || !(model.initial_variance >= 0.0f)
            || !(model.kappa > 0.0f) || !(model.theta > 0.0f)
            || !(model.gamma > 0.0f)
            || !(model.rho >= -1.0f && model.rho <= 1.0f)) {
            throw std::invalid_argument("Invalid Heston model input.");
        }
        models.push_back(model);
    }
    return models;
}

}  // namespace ai_factory::workbench::heston
