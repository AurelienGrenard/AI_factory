// Host implementation of the Svensson dataset loader.
#include "curve/svensson/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace ai_factory::workbench::curve::svensson {
// Parse and validate curve rows while preserving their dataset order.
std::vector<SvenssonParameters> load_curves(
    const std::filesystem::path& dataset_path
) {
    std::ifstream stream(dataset_path);
    if (!stream) {
        throw std::runtime_error(
            "Could not open Svensson JSON: " + dataset_path.string()
        );
    }

    nlohmann::json document;
    try {
        stream >> document;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Invalid Svensson JSON '" + dataset_path.string()
            + "': " + error.what()
        );
    }

    datasets::validate_curve_dataset(document);
    const auto& rows = document.at("curves");
    std::vector<SvenssonParameters> curves;
    curves.reserve(rows.size());
    for (const auto& row : rows) {
        const std::string row_id = row.at("id").get<std::string>();
        const auto& values = row.at("parameters");
        const SvenssonParameters parameters = {
            values.at("beta0").get<float>(),
            values.at("beta1").get<float>(),
            values.at("beta2").get<float>(),
            values.at("beta3").get<float>(),
            values.at("tau1").get<float>(),
            values.at("tau2").get<float>(),
        };
        const std::string prefix =
            "Svensson curve row id '" + row_id + "': ";
        if (!std::isfinite(parameters.beta0))
            throw std::invalid_argument(prefix + "beta0 must be finite.");
        if (!std::isfinite(parameters.beta1))
            throw std::invalid_argument(prefix + "beta1 must be finite.");
        if (!std::isfinite(parameters.beta2))
            throw std::invalid_argument(prefix + "beta2 must be finite.");
        if (!std::isfinite(parameters.beta3))
            throw std::invalid_argument(prefix + "beta3 must be finite.");
        if (!std::isfinite(parameters.tau1) || !(parameters.tau1 > 0.0f))
            throw std::invalid_argument(prefix + "tau1 must be finite and positive.");
        if (!std::isfinite(parameters.tau2) || !(parameters.tau2 > 0.0f))
            throw std::invalid_argument(prefix + "tau2 must be finite and positive.");
        if (!(parameters.tau2 > parameters.tau1)) {
            throw std::invalid_argument(prefix + "tau2 must exceed tau1.");
        }
        curves.push_back(parameters);
    }
    return curves;
}

}  // namespace ai_factory::workbench::curve::svensson
