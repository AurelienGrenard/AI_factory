// Convert Range Accrual JSON rows into compact CUDA parameters.
#include "product/range_accrual/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace ai_factory::workbench::product {

// Parse one dataset and enforce its schedule and observation band.
std::vector<RangeAccrualParameters> load_range_accruals(
    const std::filesystem::path& dataset_path
) {
    std::ifstream stream(dataset_path);
    if (!stream) {
        throw std::runtime_error(
            "Could not open Range Accrual JSON: " + dataset_path.string()
        );
    }

    nlohmann::json document;
    try {
        stream >> document;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Invalid Range Accrual JSON '" + dataset_path.string()
            + "': " + error.what()
        );
    }

    datasets::validate_product_dataset(document);
    const auto& rows = document.at("products");
    std::vector<RangeAccrualParameters> products;
    products.reserve(rows.size());
    for (const auto& row : rows) {
        const std::string row_id = row.at("id").get<std::string>();
        const auto& parameters = row.at("parameters");
        const RangeAccrualParameters product = {
            parameters.at("maturity").get<float>(),
            parameters.at("observation_interval").get<float>(),
            parameters.at("lower_barrier").get<float>(),
            parameters.at("upper_barrier").get<float>(),
            parameters.at("coupon_rate").get<float>(),
        };
        const std::string prefix = "Range Accrual row id '" + row_id + "': ";
        if (!std::isfinite(product.maturity) || !(product.maturity > 0.0f))
            throw std::invalid_argument(
                prefix + "maturity must be finite and positive."
            );
        if (!std::isfinite(product.observation_interval)
            || !(product.observation_interval > 0.0f)
            || product.observation_interval > product.maturity) {
            throw std::invalid_argument(
                prefix + "observation_interval must be finite, positive, "
                "and at most maturity."
            );
        }
        const float raw_count =
            product.maturity / product.observation_interval;
        const float nearest_count = roundf(raw_count);
        const float tolerance =
            32.0f * std::numeric_limits<float>::epsilon()
            * std::max(raw_count, 1.0f);
        if (fabsf(raw_count - nearest_count) > tolerance) {
            throw std::invalid_argument(
                prefix + "maturity must be an integer multiple of "
                "observation_interval."
            );
        }
        if (!std::isfinite(product.lower_barrier)
            || !std::isfinite(product.upper_barrier)
            || !(product.lower_barrier > 0.0f)
            || !(product.lower_barrier < 1.0f)
            || !(product.upper_barrier > 1.0f)) {
            throw std::invalid_argument(
                prefix + "barriers must be finite and satisfy "
                "0 < lower_barrier < 1 < upper_barrier."
            );
        }
        if (!std::isfinite(product.coupon_rate)
            || !(product.coupon_rate > 0.0f)) {
            throw std::invalid_argument(
                prefix + "coupon_rate must be finite and positive."
            );
        }
        products.push_back(product);
    }
    return products;
}

}  // namespace ai_factory::workbench::product
