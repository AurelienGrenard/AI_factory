// Convert Up-no-touch JSON rows into compact CUDA parameters.
#include "product/up_no_touch/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace ai_factory::workbench::product {

// Parse one dataset and preserve its row order in the returned vector.
std::vector<UpNoTouchParameters> load_up_no_touches(
    const std::filesystem::path& dataset_path
) {
    std::ifstream stream(dataset_path);
    if (!stream) {
        throw std::runtime_error(
            "Could not open Up no-touch JSON: " + dataset_path.string()
        );
    }

    nlohmann::json document;
    try {
        stream >> document;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Invalid Up no-touch JSON '" + dataset_path.string()
            + "': " + error.what()
        );
    }

    datasets::validate_product_dataset(document);
    const auto& rows = document.at("products");
    std::vector<UpNoTouchParameters> products;
    products.reserve(rows.size());
    // Keep only the three values consumed by the CUDA payoff.
    for (const auto& row : rows) {
        const std::string row_id = row.at("id").get<std::string>();
        const auto& parameters = row.at("parameters");
        const UpNoTouchParameters product = {
            parameters.at("barrier").get<float>(),
            parameters.at("cash_payoff").get<float>(),
            parameters.at("maturity").get<std::uint32_t>(),
        };
        const std::string prefix =
            "Up no-touch row id '" + row_id + "': ";
        if (!std::isfinite(product.barrier) || !(product.barrier > 0.0f))
            throw std::invalid_argument(prefix + "barrier must be finite and positive.");
        if (!std::isfinite(product.cash_payoff)
            || !(product.cash_payoff > 0.0f)) {
            throw std::invalid_argument(
                prefix + "cash_payoff must be finite and positive."
            );
        }
        if (product.maturity == 0U)
            throw std::invalid_argument(prefix + "maturity must be a positive business-day count.");
        products.push_back(product);
    }
    return products;
}

}  // namespace ai_factory::workbench::product
