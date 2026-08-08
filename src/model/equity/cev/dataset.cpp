// Host implementation of the CEV dataset loader.
#include "model/equity/cev/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace ai_factory::workbench::cev {

// Parse and validate CEV rows while preserving their dataset order.
std::vector<CevModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    std::ifstream stream(dataset_path);
    if (!stream) {
        throw std::runtime_error(
            "Could not open CEV JSON: " + dataset_path.string()
        );
    }

    nlohmann::json document;
    try {
        stream >> document;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Invalid CEV JSON '" + dataset_path.string()
            + "': " + error.what()
        );
    }

    datasets::validate_model_dataset(document);
    const auto& rows = document.at("models");
    std::vector<CevModelParameters> models;
    models.reserve(rows.size());
    for (const auto& row : rows) {
        const std::string row_id = row.at("id").get<std::string>();
        const auto& parameters = row.at("parameters");
        const CevModelParameters model = {
            parameters.at("spot").get<float>(),
            parameters.at("risk_free_rate").get<float>(),
            parameters.at("dividend_yield").get<float>(),
            parameters.at("sigma").get<float>(),
            parameters.at("beta").get<float>(),
        };
        const std::string prefix =
            "CEV model row id '" + row_id + "': ";
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
        if (!std::isfinite(model.sigma) || !(model.sigma > 0.0f)) {
            throw std::invalid_argument(
                prefix + "sigma must be finite and positive."
            );
        }
        if (!std::isfinite(model.beta)
            || !(model.beta >= 0.5f && model.beta < 1.0f)) {
            throw std::invalid_argument(
                prefix + "beta must lie in [0.5, 1)."
            );
        }
        models.push_back(model);
    }
    return models;
}

}  // namespace ai_factory::workbench::cev
