// Host implementation of the Black-Scholes dataset loader.
#include "model/equity/markovian/black_scholes/dataset.hpp"
#include "model/equity/markovian/black_scholes/parameter_domain.hpp"
#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

namespace ai_factory::workbench::model::equity::black_scholes {

// Parse and validate Black-Scholes rows while preserving their dataset order.
std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<ModelParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Model,
        "Black-Scholes model",
        [&](const nlohmann::json& parameters, const std::string& prefix) {
            const ModelParameters model = {
                parameters.at("spot").get<float>(),
                parameters.at("risk_free_rate").get<float>(),
                parameters.at("dividend_yield").get<float>(),
                parameters.at("volatility").get<float>(),
            };
            validate_parameters(model, prefix);
            return model;
    });
}

}  // namespace ai_factory::workbench::model::equity::black_scholes
