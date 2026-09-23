// Host implementation of the Heston dataset loader.
#include "model/equity/markovian/heston/dataset.hpp"
#include "model/equity/markovian/heston/parameter_domain.hpp"
#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

namespace ai_factory::workbench::model::equity::heston {

// Parse and validate Heston rows while preserving their dataset order.
std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<ModelParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Model,
        "Heston model",
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
            };
            validate_parameters(model, prefix);
            return model;
    });
}

}  // namespace ai_factory::workbench::model::equity::heston
