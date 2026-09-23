// Host implementation of the Variance-Gamma dataset loader.
#include "model/equity/markovian/variance_gamma/dataset.hpp"
#include "common/dataset_validation.hpp"
#include "model/equity/markovian/variance_gamma/parameter_domain.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::model::equity::variance_gamma {

std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<ModelParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Model,
        "Variance-Gamma model",
        [&](const nlohmann::json& parameters, const std::string& prefix) {
            const ModelParameters model = {
                parameters.at("spot").get<float>(),
                parameters.at("risk_free_rate").get<float>(),
                parameters.at("dividend_yield").get<float>(),
                parameters.at("sigma").get<float>(),
                parameters.at("nu").get<float>(),
                parameters.at("theta").get<float>(),
            };
            validate_parameters(model, prefix);
            return model;
    });
}

}  // namespace ai_factory::workbench::model::equity::variance_gamma
