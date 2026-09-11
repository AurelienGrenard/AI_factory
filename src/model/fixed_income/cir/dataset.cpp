// Host JSON loader for CIR short-rate parameters; retain the ordered row identity.
#include "model/fixed_income/cir/dataset.hpp"
#include "model/fixed_income/cir/parameter_row.hpp"
#include "common/dataset_validation.hpp"

namespace ai_factory::workbench::model::fixed_income::cir {
std::vector<ModelParameters> load_models(const std::filesystem::path& path) {
    return datasets::load_parameter_rows<ModelParameters>(
        path, datasets::ParameterDatasetFamily::Model, "CIR model",
        cir::parse_parameter_row
    );
}
}  // namespace ai_factory::workbench::model::fixed_income::cir
