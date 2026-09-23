// Host implementation of the Nelson-Siegel dataset loader.
#include "curve/nelson_siegel/dataset.hpp"
#include "curve/nelson_siegel/parameter_domain.hpp"
#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::curve::nelson_siegel {
// Parse and validate curve rows while preserving their dataset order.
std::vector<NelsonSiegelParameters> load_curves(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<NelsonSiegelParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Curve,
        "Nelson-Siegel curve",
        [&](const nlohmann::json& values, const std::string& prefix) {
            const NelsonSiegelParameters parameters = {
                values.at("beta0").get<float>(),
                values.at("beta1").get<float>(),
                values.at("beta2").get<float>(),
                values.at("tau").get<float>(),
            };
            validate_parameters(parameters, prefix);
            return parameters;
    });
}

}  // namespace ai_factory::workbench::curve::nelson_siegel
