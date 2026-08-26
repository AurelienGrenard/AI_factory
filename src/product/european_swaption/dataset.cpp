// Convert European-swaption JSON rows into compact CUDA parameters.
#include "product/european_swaption/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::product {
namespace {

// Parse and validate the common European-swaption dataset envelope once.
nlohmann::json load_document(const std::filesystem::path& dataset_path) {
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
    return document;
}

// Validate the contract scalars shared by regular and explicit rows.
void validate_contract(
    float notional,
    float strike,
    std::uint32_t exercise_time,
    const std::string& prefix
) {
    if (!std::isfinite(notional) || !(notional > 0.0f)) {
        throw std::invalid_argument(
            prefix + "notional must be finite and positive."
        );
    }
    if (!std::isfinite(strike) || strike < 0.0f) {
        throw std::invalid_argument(
            prefix + "strike must be finite and non-negative for Jamshidian."
        );
    }
    if (exercise_time == 0U) {
        throw std::invalid_argument(
            prefix + "exercise_time must be a positive day count."
        );
    }
}

}  // namespace

// Parse one dataset whose fixed-leg dates and accruals are both regular.
RegularEuropeanSwaptionDataset load_european_swaptions(
    const std::filesystem::path& dataset_path
) {
    const nlohmann::json document = load_document(dataset_path);
    const auto& rows = document.at("products");
    RegularEuropeanSwaptionDataset dataset{};
    dataset.products.reserve(rows.size());
    for (const auto& row : rows) {
        const std::string row_id = row.at("id").get<std::string>();
        const auto& parameters = row.at("parameters");
        const std::string prefix =
            "Regular European swaption row id '" + row_id + "': ";

        RegularEuropeanSwaptionParameters product{};
        product.notional = parameters.at("notional").get<float>();
        product.strike = parameters.at("strike").get<float>();
        product.accrual_fraction =
            parameters.at("accrual_fraction").get<float>();
        product.exercise_time =
            parameters.at("exercise_time").get<std::uint32_t>();
        product.payment_interval =
            parameters.at("payment_interval").get<std::uint32_t>();
        product.payment_count =
            parameters.at("payment_count").get<std::uint32_t>();

        validate_contract(
            product.notional,
            product.strike,
            product.exercise_time,
            prefix
        );
        if (!std::isfinite(product.accrual_fraction)
            || !(product.accrual_fraction > 0.0f)) {
            throw std::invalid_argument(
                prefix + "accrual_fraction must be finite and positive."
            );
        }
        if (product.payment_interval == 0U) {
            throw std::invalid_argument(
                prefix + "payment_interval must be a positive day count."
            );
        }
        if (product.payment_count == 0U) {
            throw std::invalid_argument(
                prefix + "payment_count must be positive."
            );
        }
        const std::uint64_t final_payment_time =
            static_cast<std::uint64_t>(product.exercise_time)
            + static_cast<std::uint64_t>(product.payment_count)
                * product.payment_interval;
        if (final_payment_time
            > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument(
                prefix + "final payment time exceeds uint32_t."
            );
        }
        dataset.products.push_back(product);
        dataset.maximum_payment_count = std::max(
            dataset.maximum_payment_count,
            product.payment_count
        );
    }
    return dataset;
}

// Parse arbitrary fixed-leg schedules and flatten them in row order.
ExplicitEuropeanSwaptionDataset load_explicit_european_swaptions(
    const std::filesystem::path& dataset_path
) {
    const nlohmann::json document = load_document(dataset_path);
    const auto& rows = document.at("products");
    ExplicitEuropeanSwaptionDataset dataset{};
    dataset.products.reserve(rows.size());
    for (const auto& row : rows) {
        const std::string row_id = row.at("id").get<std::string>();
        const auto& parameters = row.at("parameters");
        const std::vector<std::uint32_t> payment_times =
            parameters.at("payment_times").get<std::vector<std::uint32_t>>();
        const std::vector<float> accrual_fractions =
            parameters.at("accrual_fractions").get<std::vector<float>>();
        const std::string prefix =
            "Explicit European swaption row id '" + row_id + "': ";
        if (payment_times.empty()) {
            throw std::invalid_argument(
                prefix + "payment_times must contain at least one date."
            );
        }
        if (payment_times.size()
            > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument(
                prefix + "payment_count exceeds uint32_t."
            );
        }
        if (accrual_fractions.size() != payment_times.size()) {
            throw std::invalid_argument(
                prefix
                + "accrual_fractions must have the same size as "
                  "payment_times."
            );
        }

        ExplicitEuropeanSwaptionParameters product{};
        product.notional = parameters.at("notional").get<float>();
        product.strike = parameters.at("strike").get<float>();
        product.exercise_time =
            parameters.at("exercise_time").get<std::uint32_t>();
        product.payment_count =
            static_cast<std::uint32_t>(payment_times.size());
        product.schedule_offset = dataset.payment_times.size();
        validate_contract(
            product.notional,
            product.strike,
            product.exercise_time,
            prefix
        );

        std::uint32_t previous_time = product.exercise_time;
        for (std::size_t payment = 0U;
             payment < payment_times.size();
             ++payment) {
            const std::uint32_t payment_time = payment_times[payment];
            const float accrual_fraction = accrual_fractions[payment];
            if (payment_time <= previous_time) {
                throw std::invalid_argument(
                    prefix
                    + "payment_times must be strictly increasing day counts "
                      "above exercise_time."
                );
            }
            if (!std::isfinite(accrual_fraction)
                || !(accrual_fraction > 0.0f)) {
                throw std::invalid_argument(
                    prefix
                    + "accrual_fractions must be finite and positive."
                );
            }
            previous_time = payment_time;
        }
        dataset.payment_times.insert(
            dataset.payment_times.end(),
            payment_times.begin(),
            payment_times.end()
        );
        dataset.accrual_fractions.insert(
            dataset.accrual_fractions.end(),
            accrual_fractions.begin(),
            accrual_fractions.end()
        );
        dataset.products.push_back(product);
        dataset.maximum_payment_count = std::max(
            dataset.maximum_payment_count,
            product.payment_count
        );
    }
    return dataset;
}

}  // namespace ai_factory::workbench::product
