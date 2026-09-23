// Host implementation of the Bates dataset loader.
#include "model/equity/markovian/bates/dataset.hpp"
#include "common/dataset_validation.hpp"
#include "model/equity/markovian/bates/parameter_domain.hpp"

#include <nlohmann/json.hpp>

namespace ai_factory::workbench::model::equity::bates {

// Parse and validate Bates rows while preserving their dataset order.
std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<ModelParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Model,
        "Bates model",
        [&](const nlohmann::json& parameters, const std::string& prefix) {
            const ModelParameters model = {
                parameters.at("spot").get<float>(),
                parameters.at("risk_free_rate").get<float>(),
                parameters.at("dividend_yield").get<float>(),
                parameters.at("initial_variance").get<float>(),
                parameters.at("kappa").get<float>(),
                parameters.at("theta").get<float>(),
                parameters.at("gamma").get<float>(),
                parameters.at("rho").get<float>(),
                parameters.at("jump_intensity").get<float>(),
                parameters.at("jump_log_mean").get<float>(),
                parameters.at("jump_log_volatility").get<float>(),
            };
            validate_parameters(model, prefix);
            return model;
    });
}

}  // namespace ai_factory::workbench::model::equity::bates
