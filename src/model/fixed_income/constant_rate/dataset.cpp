#include "model/fixed_income/constant_rate/dataset.hpp"

#include "common/dataset_validation.hpp"

#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::model::fixed_income::constant_rate {

std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<ModelParameters>(
        dataset_path, datasets::ParameterDatasetFamily::Model,
        "Constant short rate",
        [](const nlohmann::json& parameters, const std::string& prefix) {
            const ModelParameters model{
                parameters.at("rate").get<float>()};
            if (!std::isfinite(model.rate))
                throw std::invalid_argument(prefix + "rate must be finite");
            return model;
        });
}

}  // namespace ai_factory::workbench::model::fixed_income::constant_rate
