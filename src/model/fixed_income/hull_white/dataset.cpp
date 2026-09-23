// Host implementation of the Hull-White one-factor dataset loader.
#include "model/fixed_income/hull_white/dataset.hpp"
#include "model/fixed_income/hull_white/parameter_domain.hpp"
#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::model::fixed_income::hull_white {

// Parse and validate model rows while preserving their dataset order.
std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<ModelParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Model,
        "Hull-White model",
        [&](const nlohmann::json& values, const std::string& prefix) {
            const ModelParameters parameters = {
                values.at("mean_reversion").get<float>(),
                values.at("volatility").get<float>(),
            };
            validate_parameters(parameters, prefix);
            return parameters;
    });
}

}  // namespace ai_factory::workbench::model::fixed_income::hull_white
