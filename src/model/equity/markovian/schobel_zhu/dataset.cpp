// Host implementation of the Schobel-Zhu dataset loader.
#include "model/equity/markovian/schobel_zhu/dataset.hpp"
#include "common/dataset_validation.hpp"
#include "model/equity/markovian/schobel_zhu/parameter_domain.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::model::equity::schobel_zhu {

// Parse and validate Schobel-Zhu rows while preserving their dataset order.
std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<ModelParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Model,
        "Schobel-Zhu model",
        [&](const nlohmann::json& parameters, const std::string& prefix) {
            const ModelParameters model = {
                parameters.at("spot").get<float>(),
                parameters.at("risk_free_rate").get<float>(),
                parameters.at("dividend_yield").get<float>(),
                parameters.at("initial_volatility").get<float>(),
                parameters.at("mean_reversion").get<float>(),
                parameters.at("long_run_volatility").get<float>(),
                parameters.at("volatility_of_volatility").get<float>(),
                parameters.at("correlation").get<float>(),
            };
            validate_parameters(model, prefix);
            return model;
    });
}

}  // namespace ai_factory::workbench::model::equity::schobel_zhu
