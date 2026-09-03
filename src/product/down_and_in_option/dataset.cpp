// Convert Down-and-in-option JSON rows into compact CUDA parameters.
#include "product/down_and_in_option/dataset.hpp"
#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::product {

// Parse one dataset and preserve its row order in the returned vector.
std::vector<DownAndInOptionParameters> load_down_and_in_options(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<DownAndInOptionParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Product,
        "Down-and-in option",
        [&](const nlohmann::json& parameters, const std::string& prefix) {
            const DownAndInOptionParameters product = {
                parameters.at("strike").get<float>(),
                parameters.at("barrier").get<float>(),
                parameters.at("maturity").get<std::uint32_t>(),
            };
            if (!std::isfinite(product.strike) || !(product.strike > 0.0f))
                throw std::invalid_argument(prefix + "strike must be finite and positive.");
            if (!std::isfinite(product.barrier) || !(product.barrier > 0.0f))
                throw std::invalid_argument(prefix + "barrier must be finite and positive.");
            if (!(product.barrier < product.strike))
                throw std::invalid_argument(prefix + "barrier must be below strike.");
            if (product.maturity_days == 0U)
                throw std::invalid_argument(prefix + "maturity must be a positive business-day count.");
            return product;
    });
}

}  // namespace ai_factory::workbench::product
