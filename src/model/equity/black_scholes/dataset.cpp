// Host implementation of the Black-Scholes dataset loader.
#include "model/equity/black_scholes/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace ai_factory::workbench::black_scholes {

// Parse and validate Black-Scholes rows while preserving their dataset order.
std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    std::ifstream stream(dataset_path);
    if (!stream) {
        throw std::runtime_error(
            "Could not open Black-Scholes JSON: " + dataset_path.string()
        );
    }

    nlohmann::json document;
    try {
        stream >> document;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Invalid Black-Scholes JSON '" + dataset_path.string()
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
            parameters.at("volatility").get<float>(),
        };
        const std::string prefix =
            "Black-Scholes model row id '" + row_id + "': ";
        if (!std::isfinite(model.spot) || !(model.spot > 0.0f))
            throw std::invalid_argument(prefix + "spot must be finite and positive.");
        if (!std::isfinite(model.risk_free_rate))
            throw std::invalid_argument(prefix + "risk_free_rate must be finite.");
        if (!std::isfinite(model.dividend_yield))
            throw std::invalid_argument(prefix + "dividend_yield must be finite.");
        if (!std::isfinite(model.volatility) || !(model.volatility > 0.0f))
            throw std::invalid_argument(prefix + "volatility must be finite and positive.");
        models.push_back(model);
    }
    return models;
}

}  // namespace ai_factory::workbench::black_scholes
