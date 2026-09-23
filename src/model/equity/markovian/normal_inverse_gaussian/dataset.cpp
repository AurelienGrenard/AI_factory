// Host implementation of the Normal-Inverse-Gaussian dataset loader.
#include "model/equity/markovian/normal_inverse_gaussian/dataset.hpp"
#include "common/dataset_validation.hpp"
#include "model/equity/markovian/normal_inverse_gaussian/parameter_domain.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::model::equity::normal_inverse_gaussian {

std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<ModelParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Model,
        "Normal-Inverse-Gaussian model",
        [&](const nlohmann::json& parameters, const std::string& prefix) {
            const ModelParameters model = {
                parameters.at("spot").get<float>(),
                parameters.at("risk_free_rate").get<float>(),
                parameters.at("dividend_yield").get<float>(),
                parameters.at("alpha").get<float>(),
                parameters.at("beta").get<float>(),
                parameters.at("delta").get<float>(),
            };
            validate_parameters(model, prefix);
            return model;
    });
}

}  // namespace ai_factory::workbench::model::equity::normal_inverse_gaussian
