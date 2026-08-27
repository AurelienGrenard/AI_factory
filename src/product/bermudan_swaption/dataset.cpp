// Convert co-terminal Bermudan-swaption JSON rows into compact parameters.
#include "product/bermudan_swaption/dataset.hpp"
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

            if (!std::isfinite(product.notional) || !(product.notional > 0.0f)) {
                throw std::invalid_argument(
                    prefix + "notional must be finite and positive."
                );
            }
            if (!std::isfinite(product.strike) || product.strike < 0.0f) {
                throw std::invalid_argument(
                    prefix + "strike must be finite and non-negative."
                );
            }
            if (!std::isfinite(product.accrual_fraction)
                || !(product.accrual_fraction > 0.0f)) {
                throw std::invalid_argument(
                    prefix + "accrual_fraction must be finite and positive."
                );
            }
            if (product.first_exercise_time_days == 0U
                || product.payment_interval_days == 0U) {
                throw std::invalid_argument(
                    prefix + "exercise and payment day counts must be positive."
                );
            }
            if (product.exercise_count < 2U) {
                throw std::invalid_argument(
                    prefix + "exercise_count must be at least two."
                );
            }
            if (product.payment_count < product.exercise_count) {
                throw std::invalid_argument(
                    prefix
                    + "payment_count must be at least exercise_count for a "
                      "co-terminal swaption."
                );
            }
            const std::uint64_t final_payment_time =
                static_cast<std::uint64_t>(product.first_exercise_time_days)
                + static_cast<std::uint64_t>(product.payment_count)
                    * product.payment_interval_days;
            if (final_payment_time
                > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument(
                    prefix + "final payment time exceeds uint32_t."
                );
            }
            return product;
    });
}

}  // namespace ai_factory::workbench::product
