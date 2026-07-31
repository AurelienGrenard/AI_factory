// Host implementation of the Nelson-Siegel dataset loader.
#include "curve/nelson_siegel/dataset.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace ai_factory::workbench::curve {
namespace {

// Reject invalid parameters before evaluating or transferring one curve.
void validate_parameters(const NelsonSiegelParameters& parameters) {
    if (!std::isfinite(parameters.beta0)
        || !std::isfinite(parameters.beta1)
        || !std::isfinite(parameters.beta2)
        || !std::isfinite(parameters.tau)
        || !(parameters.tau > 0.0f)) {
        throw std::invalid_argument("Invalid Nelson-Siegel parameters.");
    }
}

}  // namespace

// Parse and validate curve rows while preserving their dataset order.
std::vector<NelsonSiegelParameters> load_nelson_siegel(
    const std::filesystem::path& dataset_path
) {
    std::ifstream stream(dataset_path);
    if (!stream) {
        throw std::runtime_error(
            "Could not open Nelson-Siegel JSON: " + dataset_path.string()
        );
    }

    nlohmann::json document;
    try {
        stream >> document;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Invalid Nelson-Siegel JSON '" + dataset_path.string()
            + "': " + error.what()
        );
    }

    const auto& rows = document.at("curves");
    std::vector<NelsonSiegelParameters> curves;
    curves.reserve(rows.size());
    for (const auto& row : rows) {
        const auto& values = row.at("parameters");
        const NelsonSiegelParameters parameters = {
            values.at("beta0").get<float>(),
            values.at("beta1").get<float>(),
            values.at("beta2").get<float>(),
            values.at("tau").get<float>(),
        };
        validate_parameters(parameters);
        curves.push_back(parameters);
    }
    return curves;
}

}  // namespace ai_factory::workbench::curve
