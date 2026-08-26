// Host implementation of the rough-Bergomi dataset loader.
#include "model/equity/rough_bergomi/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::model::equity::rough_bergomi {

// Parse and validate rough-Bergomi rows while preserving dataset order.
std::vector<RoughBergomiModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    std::ifstream stream(dataset_path);
    if (!stream) {
        throw std::runtime_error(
            "Could not open rough-Bergomi JSON: " + dataset_path.string()
        );
    }

    nlohmann::json document;
    try {
        stream >> document;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Invalid rough-Bergomi JSON '" + dataset_path.string()
            + "': " + error.what()
        );
    }

    datasets::validate_model_dataset(document);
    const auto& rows = document.at("models");
    std::vector<RoughBergomiModelParameters> models;
    models.reserve(rows.size());
    for (const auto& row : rows) {
        const std::string row_id = row.at("id").get<std::string>();
        const auto& parameters = row.at("parameters");
        const RoughBergomiModelParameters model = {
            parameters.at("spot").get<float>(),
            parameters.at("risk_free_rate").get<float>(),
            parameters.at("dividend_yield").get<float>(),
            parameters.at("xi_0").get<float>(),
            parameters.at("eta").get<float>(),
            parameters.at("hurst_exponent").get<float>(),
            parameters.at("rho").get<float>(),
        };
        const std::string prefix =
            "rough-Bergomi model row id '" + row_id + "': ";
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
        if (!std::isfinite(model.xi_0) || !(model.xi_0 > 0.0f)) {
            throw std::invalid_argument(
                prefix + "xi_0 must be finite and positive."
            );
        }
        if (!std::isfinite(model.eta) || !(model.eta > 0.0f)) {
            throw std::invalid_argument(
                prefix + "eta must be finite and positive."
            );
        }
        if (!std::isfinite(model.hurst_exponent)
            || !(model.hurst_exponent > 0.0f
                 && model.hurst_exponent < 0.5f)) {
            throw std::invalid_argument(
                prefix
                + "hurst_exponent must lie strictly between 0 and 0.5."
            );
        }
        if (!std::isfinite(model.rho)
            || !(model.rho > -1.0f && model.rho < 1.0f)) {
            throw std::invalid_argument(
                prefix + "rho must lie strictly between -1 and 1."
            );
        }
        models.push_back(model);
    }
    return models;
}

}  // namespace ai_factory::workbench::model::equity::rough_bergomi
