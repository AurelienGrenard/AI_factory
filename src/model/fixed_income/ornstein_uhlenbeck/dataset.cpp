// Host implementation of the Ornstein-Uhlenbeck dataset loader.
#include "model/fixed_income/ornstein_uhlenbeck/dataset.hpp"
#include "model/fixed_income/ornstein_uhlenbeck/parameter_domain.hpp"
#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::model::fixed_income::ornstein_uhlenbeck {

// Parse and validate OU rows while preserving their dataset order.
std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<ModelParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Model,
        "Ornstein-Uhlenbeck model",
        [&](const nlohmann::json& parameters, const std::string& prefix) {
            const ModelParameters model = {
                {
                    parameters.at("mean_reversion").get<float>(),
                    parameters.at("volatility").get<float>(),
                },
                parameters.at("initial_state").get<float>(),
            };
            validate_parameters(model, prefix);
            return model;
    });
}

}  // namespace ai_factory::workbench::model::fixed_income::ornstein_uhlenbeck
