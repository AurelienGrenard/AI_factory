// Host implementation of the Merton dataset loader.
#include "model/equity/markovian/merton/dataset.hpp"
#include "model/equity/markovian/merton/parameter_domain.hpp"
#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

namespace ai_factory::workbench::model::equity::merton {

// Parse and validate Merton rows while preserving their dataset order.
std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<ModelParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Model,
        "Merton model",
        [&](const nlohmann::json& parameters, const std::string& prefix) {
            const ModelParameters model = {
                parameters.at("spot").get<float>(),
                parameters.at("risk_free_rate").get<float>(),
                parameters.at("dividend_yield").get<float>(),
                parameters.at("volatility").get<float>(),
                parameters.at("jump_intensity").get<float>(),
                parameters.at("jump_log_mean").get<float>(),
                parameters.at("jump_log_volatility").get<float>(),
            };
            validate_parameters(model, prefix);
            return model;
    });
}

}  // namespace ai_factory::workbench::model::equity::merton
