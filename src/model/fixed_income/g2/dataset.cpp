// Host implementation of the G2 dataset loader.
#include "model/fixed_income/g2/dataset.hpp"
#include "model/fixed_income/g2/parameter_domain.hpp"
#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::model::fixed_income::g2 {

// Parse and validate G2 rows while preserving their dataset order.
std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<ModelParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Model,
        "G2 model",
        [&](const nlohmann::json& values, const std::string& prefix) {
            const ModelParameters model = {
                {
                    values.at("mean_reversion_x").get<float>(),
                    values.at("volatility_x").get<float>(),
                    values.at("mean_reversion_y").get<float>(),
                    values.at("volatility_y").get<float>(),
                    values.at("correlation").get<float>(),
                },
                {
                    values.at("initial_state_x").get<float>(),
                    values.at("initial_state_y").get<float>(),
                },
            };
            validate_parameters(model, prefix);
            return model;
    });
}

}  // namespace ai_factory::workbench::model::fixed_income::g2
