// Convert Geometric-Asian-option JSON rows into compact CUDA parameters.
#include "product/geometric_asian_option/dataset.hpp"
#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::product {

// Parse one dataset and preserve its row order in the returned vector.
std::vector<GeometricAsianOptionParameters> load_geometric_asian_options(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<GeometricAsianOptionParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Product,
        "Geometric Asian call",
        [&](const nlohmann::json& parameters, const std::string& prefix) {
            const GeometricAsianOptionParameters product = {
                parameters.at("strike").get<float>(),
                parameters.at("maturity").get<std::uint32_t>(),
            };
            if (!std::isfinite(product.strike) || !(product.strike > 0.0f))
                throw std::invalid_argument(prefix + "strike must be finite and positive.");
            if (product.maturity_days == 0U)
                throw std::invalid_argument(prefix + "maturity must be a positive business-day count.");
            return product;
    });
}

}  // namespace ai_factory::workbench::product
