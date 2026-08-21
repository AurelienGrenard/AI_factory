// Convert European-swaption JSON rows into compact CUDA parameters.
#include "product/european_swaption/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::product {

// Parse one dataset and preserve its row order in the returned vector.
std::vector<EuropeanSwaptionParameters> load_european_swaptions(
    const std::filesystem::path& dataset_path
) {
    std::ifstream stream(dataset_path);
    if (!stream) {
        throw std::runtime_error(
            "Could not open European swaption JSON: "
            + dataset_path.string()
        );
    }

    nlohmann::json document;
    try {
        stream >> document;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Invalid European swaption JSON '" + dataset_path.string()
            + "': " + error.what()
        );
    }

    datasets::validate_product_dataset(document);
    const auto& rows = document.at("products");
    std::vector<EuropeanSwaptionParameters> products;
    products.reserve(rows.size());
    for (const auto& row : rows) {
        const std::string row_id = row.at("id").get<std::string>();
        const auto& parameters = row.at("parameters");
        const std::vector<std::uint32_t> payment_times =
            parameters.at("payment_times").get<std::vector<std::uint32_t>>();
        const std::vector<std::uint32_t> accrual_periods =
            parameters.at("accrual_periods")
                .get<std::vector<std::uint32_t>>();
        const std::string prefix =
            "European swaption row id '" + row_id + "': ";
        if (payment_times.empty()
            || payment_times.size() > kMaximumEuropeanSwaptionPayments) {
            throw std::invalid_argument(
                prefix + "payment_times must contain between 1 and 64 dates."
            );
        }
        if (accrual_periods.size() != payment_times.size()) {
            throw std::invalid_argument(
                prefix
                + "accrual_periods must have the same size as payment_times."
            );
        }

        EuropeanSwaptionParameters product{};
        product.notional = parameters.at("notional").get<float>();
        product.strike = parameters.at("strike").get<float>();
        product.exercise_time =
            parameters.at("exercise_time").get<std::uint32_t>();
        product.payment_count =
            static_cast<std::uint32_t>(payment_times.size());
        if (!std::isfinite(product.notional) || !(product.notional > 0.0f)) {
            throw std::invalid_argument(
                prefix + "notional must be finite and positive."
            );
        }
        if (!std::isfinite(product.strike) || product.strike < 0.0f) {
            throw std::invalid_argument(
                prefix
                + "strike must be finite and non-negative for Jamshidian."
            );
        }
        if (product.exercise_time == 0U) {
            throw std::invalid_argument(
                prefix + "exercise_time must be a positive day count."
            );
        }
        std::uint32_t previous_time = product.exercise_time;
        for (std::size_t payment = 0U;
             payment < payment_times.size();
             ++payment) {
            const std::uint32_t payment_time = payment_times[payment];
            const std::uint32_t accrual_period = accrual_periods[payment];
            if (payment_time <= previous_time) {
                throw std::invalid_argument(
                    prefix
                    + "payment_times must be strictly increasing day counts "
                      "above exercise_time."
                );
            }
            if (accrual_period == 0U) {
                throw std::invalid_argument(
                    prefix + "accrual_periods must be positive day counts."
                );
            }
            product.payment_times[payment] = payment_time;
            product.accrual_periods[payment] = accrual_period;
            previous_time = payment_time;
        }
        products.push_back(product);
    }
    return products;
}

}  // namespace ai_factory::workbench::product
