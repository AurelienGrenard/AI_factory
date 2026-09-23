// Host implementation of the Svensson dataset loader.
#include "curve/svensson/dataset.hpp"
#include "curve/svensson/parameter_domain.hpp"
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
            validate_parameters(parameters, prefix);
            return parameters;
    });
}

}  // namespace ai_factory::workbench::curve::svensson
