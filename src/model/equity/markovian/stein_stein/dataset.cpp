// JSON loading and validation for Stein--Stein parameter datasets.
#include "model/equity/markovian/stein_stein/dataset.hpp"

#include "common/dataset_validation.hpp"
#include "model/equity/markovian/stein_stein/parameter_domain.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::model::equity::stein_stein {

std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<ModelParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Model,
        "Stein-Stein model",
        [](const nlohmann::json& row, const std::string& prefix) {
            const ModelParameters model{
                row.at("spot").get<float>(),
                row.at("risk_free_rate").get<float>(),
                row.at("dividend_yield").get<float>(),
                row.at("initial_volatility").get<float>(),
                row.at("mean_reversion").get<float>(),
                row.at("volatility_of_volatility").get<float>(),
                row.at("rho").get<float>(),
            };
            validate_parameters(model, prefix);
            return model;
        }
    );
}

}  // namespace ai_factory::workbench::model::equity::stein_stein
