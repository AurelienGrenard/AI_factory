// JSON loading and validation for Heston 3/2 parameter datasets.
#include "model/equity/markovian/heston_3_2/dataset.hpp"

#include "common/dataset_validation.hpp"
#include "model/equity/markovian/heston_3_2/parameter_domain.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::model::equity::heston_3_2 {

std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<ModelParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Model,
        "Heston 3/2 model",
        [](const nlohmann::json& row, const std::string& prefix) {
            const ModelParameters model{
                row.at("spot").get<float>(),
                row.at("risk_free_rate").get<float>(),
                row.at("dividend_yield").get<float>(),
                row.at("initial_variance").get<float>(),
                row.at("mean_reversion").get<float>(),
                row.at("long_run_variance").get<float>(),
                row.at("volatility_of_variance").get<float>(),
                row.at("rho").get<float>(),
            };
            validate_parameters(model, prefix);
            return model;
        }
    );
}

}  // namespace ai_factory::workbench::model::equity::heston_3_2
