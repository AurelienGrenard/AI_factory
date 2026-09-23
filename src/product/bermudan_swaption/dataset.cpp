// Convert co-terminal Bermudan-swaption JSON rows into compact parameters.
#include "product/bermudan_swaption/dataset.hpp"
#include "product/bermudan_swaption/parameter_domain.hpp"
#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::product {

std::vector<BermudanSwaptionParameters> load_bermudan_swaptions(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<BermudanSwaptionParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Product,
        "Bermudan swaption",
        [](const nlohmann::json& parameters, const std::string& prefix) {
            const BermudanSwaptionParameters product{
                parameters.at("notional").get<float>(),
                parameters.at("strike").get<float>(),
                parameters.at("accrual_fraction").get<float>(),
                parameters.at("first_exercise_time").get<std::uint32_t>(),
                parameters.at("payment_interval").get<std::uint32_t>(),
                parameters.at("payment_count").get<std::uint32_t>(),
                parameters.at("exercise_count").get<std::uint32_t>(),
            };

            bermudan_swaption::validate_parameters(product, prefix);
            return product;
    });
}

}  // namespace ai_factory::workbench::product
