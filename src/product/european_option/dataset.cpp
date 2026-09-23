// Convert European-option JSON rows into compact CUDA parameters.
#include "product/european_option/dataset.hpp"
#include "product/european_option/parameter_domain.hpp"
#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

namespace ai_factory::workbench::product {

// Parse one dataset and preserve its row order in the returned vector.
std::vector<EuropeanOptionParameters> load_european_options(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<EuropeanOptionParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Product,
        "European option",
        [&](const nlohmann::json& parameters, const std::string& prefix) {
            const EuropeanOptionParameters product = {
                parameters.at("strike").get<float>(),
                parameters.at("maturity").get<std::uint32_t>(),
            };
            european_option::validate_parameters(product, prefix);
            return product;
    });
}

}  // namespace ai_factory::workbench::product
