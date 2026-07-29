// Validate the workbench input databases and their construction invariants.
#include "heston/parameters.hpp"
#include "products/american_put.hpp"
#include "tools/registry/dataset.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const std::filesystem::path model_json_path =
    "registry/production/model_parameters/heston/data/heston_01.json";
const std::filesystem::path product_json_path =
    "registry/production/product_parameters/american_puts/data/american_puts_01.json";
const std::filesystem::path result_json_path =
    "registry/production/prices/heston/american_puts/data/heston_01__american_puts_01__01.json";
const std::filesystem::path result_yaml_path =
    "registry/production/prices/heston/american_puts/specifications/heston_01__american_puts_01__01.yaml";

// Stop immediately with a readable invariant name.
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// Parse one registry JSON document with an explicit path in any error.
nlohmann::json read_json(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot open registry JSON: " + path.string());
    }
    return nlohmann::json::parse(stream);
}

// Read one compact specification for a few exact recipe assertions.
std::string read_text(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot open registry YAML: " + path.string());
    }
    return {
        std::istreambuf_iterator<char>(stream),
        std::istreambuf_iterator<char>()
    };
}

// Check one Heston row before it reaches CUDA.
void validate_model(
    const ai_factory::workbench::heston::HestonModelParameters& model
) {
    require(std::isfinite(model.spot) && model.spot > 0.0f, "invalid spot");
    require(std::isfinite(model.risk_free_rate), "invalid risk-free rate");
    require(std::isfinite(model.dividend_yield), "invalid dividend yield");
    require(
        std::isfinite(model.initial_variance) && model.initial_variance > 0.0f,
        "invalid initial variance"
    );
    require(std::isfinite(model.kappa) && model.kappa > 0.0f, "invalid kappa");
    require(std::isfinite(model.theta) && model.theta > 0.0f, "invalid theta");
    require(std::isfinite(model.gamma) && model.gamma > 0.0f, "invalid gamma");
    require(
        std::isfinite(model.rho) && model.rho >= -1.0f && model.rho <= 1.0f,
        "invalid rho"
    );
}

// Check the discrete-American schedule contract for one put row.
void validate_product(
    const ai_factory::workbench::products::AmericanPutInput& product
) {
    require(
        std::isfinite(product.strike) && product.strike > 0.0f,
        "invalid strike"
    );
    require(
        std::isfinite(product.maturity) && product.maturity > 0.0f,
        "invalid maturity"
    );
    require(
        std::isfinite(product.exercise_interval)
            && product.exercise_interval > 0.0f,
        "invalid exercise interval"
    );
    require(
        product.exercise_interval < product.maturity,
        "exercise schedule has no pre-maturity date"
    );
    require(
        product.exercise_interval == 1.0f / 12.0f
            || product.exercise_interval == 1.0f / 24.0f,
        "exercise interval is neither monthly nor semi-monthly"
    );
}

}  // namespace

// Load the real inputs and verify both supported result constructions.
int main() {
    using namespace ai_factory::workbench;

    const std::vector<heston::HestonModelParameters> models =
        heston::load_heston(model_json_path);
    const std::vector<products::AmericanPutInput> products =
        products::load_american_puts(product_json_path);

    require(models.size() == 1'000U, "Heston row count is not 1000");
    require(products.size() == 1'000U, "American-put row count is not 1000");
    for (const auto& model : models) validate_model(model);
    for (const auto& product : products) validate_product(product);

    require(
        registry::result_row_count(
            models.size(),
            products.size(),
            registry::ResultConstruction::Aligned
        ) == 1'000U,
        "aligned result count is not 1000"
    );
    require(
        registry::result_row_count(
            models.size(),
            products.size(),
            registry::ResultConstruction::CartesianProduct
        ) == 1'000'000U,
        "Cartesian result count is not 1000000"
    );

    const nlohmann::json model_document = read_json(model_json_path);
    const nlohmann::json product_document = read_json(product_json_path);
    const nlohmann::json result_document = read_json(result_json_path);
    require(
        result_document.at("database_id") == "heston_01__american_puts_01__01",
        "unexpected American-put result database id"
    );
    require(
        result_document.at("row_count") == 1'000U
            && result_document.at("results").size() == 1'000U,
        "American-put result row count is not 1000"
    );
    require(
        result_document.at("model_database").at("id") == "heston_01"
            && result_document.at("product_database").at("id")
                == "american_puts_01",
        "American-put result references unexpected input databases"
    );
    require(
        result_document.at("timing").at("wall_seconds").get<double>() > 0.0
            && result_document.at("timing")
                   .at("kernel_seconds")
                   .get<double>() > 0.0,
        "American-put production timings are not positive"
    );

    for (std::size_t row = 0U; row < 1'000U; ++row) {
        const nlohmann::json& result =
            result_document.at("results").at(row);
        const float price =
            result.at("outputs").at("price").get<float>();
        const float standard_error =
            result.at("outputs").at("standard_error").get<float>();
        const float intrinsic = std::max(
            products[row].strike - models[row].spot, 0.0f
        );
        require(
            result.at("model_id")
                    == model_document.at("models").at(row).at("id")
                && result.at("product_id")
                    == product_document.at("products").at(row).at("id"),
            "American-put result row references unexpected inputs"
        );
        require(
            std::isfinite(price) && std::isfinite(standard_error),
            "American-put production output is not finite"
        );
        require(standard_error >= 0.0f, "negative production standard error");
        require(
            price + 2.0e-6f >= intrinsic,
            "American-put production price is below immediate exercise"
        );
        require(
            price <= products[row].strike + 2.0e-6f,
            "American-put production price exceeds its strike"
        );
    }

    const std::string specification = read_text(result_yaml_path);
    require(
        specification.find(
            "monte_carlo_paths_per_price: 1048576"
        ) != std::string::npos,
        "American-put YAML does not record 1048576 paths per price"
    );
    require(
        specification.find("blocks_per_price: 128") != std::string::npos,
        "American-put YAML does not record 128 blocks per price"
    );
    require(
        specification.find(
            "regression_basis: \"Two-factor Laguerre degree 2\""
        ) != std::string::npos,
        "American-put YAML does not identify its regression basis"
    );
}
