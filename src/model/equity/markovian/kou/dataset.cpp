// Host implementation of the Kou dataset loader.
#include "model/equity/markovian/kou/dataset.hpp"
#include "common/dataset_validation.hpp"
#include "model/equity/markovian/kou/parameter_domain.hpp"

#include <nlohmann/json.hpp>

namespace ai_factory::workbench::model::equity::kou {

// Parse and validate Kou rows while preserving their dataset order.
std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<ModelParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Model,
        "Kou model",
        [&](const nlohmann::json& parameters, const std::string& prefix) {
            const ModelParameters model = {
                parameters.at("spot").get<float>(),
                parameters.at("risk_free_rate").get<float>(),
                parameters.at("dividend_yield").get<float>(),
                parameters.at("volatility").get<float>(),
                parameters.at("jump_intensity").get<float>(),
                parameters.at("up_probability").get<float>(),
                parameters.at("positive_jump_rate").get<float>(),
                parameters.at("negative_jump_rate").get<float>(),
            };
            validate_parameters(model, prefix);
            return model;
    });
}

}  // namespace ai_factory::workbench::model::equity::kou
