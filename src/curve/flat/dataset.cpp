// Host implementation of the flat-curve dataset loader.
#include "curve/flat/dataset.hpp"
#include "curve/flat/parameter_domain.hpp"
#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

namespace ai_factory::workbench::curve::flat {

// Parse and validate curve rows while preserving their dataset order.
std::vector<FlatCurveParameters> load_curves(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<FlatCurveParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Curve,
        "Flat curve",
        [&](const nlohmann::json& values, const std::string& prefix) {
            const FlatCurveParameters parameters = {
                values.at("rate").get<float>(),
            };
            validate_parameters(parameters, prefix);
            return parameters;
        }
    );
}

}  // namespace ai_factory::workbench::curve::flat
