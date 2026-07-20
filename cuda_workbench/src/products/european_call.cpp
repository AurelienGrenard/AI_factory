// Host implementation of the European-call registry loader.
// The JSON document is discarded after its rows have been converted to the
// compact product representation used by CUDA.
#include "products/european_call.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>

namespace ai_factory::workbench::products {

// Parse one product database and preserve its row order in the returned vector.
std::vector<EuropeanCallInput> load_european_calls(
    const std::filesystem::path& json_path
) {
    std::ifstream stream(json_path);
    if (!stream) {
        throw std::runtime_error(
            "Could not open European call JSON: " + json_path.string()
        );
    }

    nlohmann::json document;
    try {
        stream >> document;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Invalid European call JSON '" + json_path.string()
            + "': " + error.what()
        );
    }

    const auto& rows = document.at("products");
    std::vector<EuropeanCallInput> products;
    products.reserve(rows.size());
    // Only strike and maturity are retained; JSON metadata stays out of CUDA.
    for (const auto& row : rows) {
        const auto& parameters = row.at("parameters");
        const EuropeanCallInput product = {
            parameters.at("strike").get<float>(),
            parameters.at("maturity").get<float>(),
        };
        if (!(product.strike > 0.0f) || !(product.maturity > 0.0f)) {
            throw std::invalid_argument("Invalid European call input.");
        }
        products.push_back(product);
    }
    return products;
}

}  // namespace ai_factory::workbench::products
