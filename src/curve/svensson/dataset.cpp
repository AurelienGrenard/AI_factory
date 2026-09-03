// Host implementation of the Svensson dataset loader.
#include "curve/svensson/dataset.hpp"
#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::curve::svensson {
// Parse and validate curve rows while preserving their dataset order.
std::vector<SvenssonParameters> load_curves(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<SvenssonParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Curve,
        "Svensson curve",
        [&](const nlohmann::json& values, const std::string& prefix) {
            const SvenssonParameters parameters = {
                values.at("beta0").get<float>(),
                values.at("beta1").get<float>(),
                values.at("beta2").get<float>(),
                values.at("beta3").get<float>(),
                values.at("tau1").get<float>(),
                values.at("tau2").get<float>(),
            };
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
            return parameters;
    });
}

}  // namespace ai_factory::workbench::curve::svensson
