// Host implementation of the Variance-Gamma dataset loader.
#include "model/equity/markovian/variance_gamma/dataset.hpp"
#include "common/dataset_validation.hpp"

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
            if (!std::isfinite(model.spot) || !(model.spot > 0.0f))
                throw std::invalid_argument(prefix + "spot must be finite and positive.");
            if (!std::isfinite(model.risk_free_rate))
                throw std::invalid_argument(prefix + "risk_free_rate must be finite.");
            if (!std::isfinite(model.dividend_yield))
                throw std::invalid_argument(prefix + "dividend_yield must be finite.");
            if (!std::isfinite(model.sigma) || !(model.sigma > 0.0f))
                throw std::invalid_argument(prefix + "sigma must be finite and positive.");
            if (!std::isfinite(model.nu) || !(model.nu > 0.0f))
                throw std::invalid_argument(prefix + "nu must be finite and positive.");
            if (!std::isfinite(model.theta))
                throw std::invalid_argument(prefix + "theta must be finite.");
            const float exponential_moment = 1.0f
                - model.theta * model.nu
                - 0.5f * model.sigma * model.sigma * model.nu;
            if (!std::isfinite(exponential_moment)
                || !(exponential_moment > 0.0f)) {
                throw std::invalid_argument(
                    prefix
                    + "1 - theta * nu - 0.5 * sigma^2 * nu must be positive."
                );
            }
            return model;
    });
}

}  // namespace ai_factory::workbench::model::equity::variance_gamma
