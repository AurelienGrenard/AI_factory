// Validate versioned previews and their catalog contracts.
#include "heston/parameters.hpp"
#include "products/american_put.hpp"
#include "tools/datasets/dataset.hpp"

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

const std::filesystem::path model_preview_path =
    "previews/model/heston/heston_01.json";
const std::filesystem::path product_preview_path =
    "previews/product/american_puts/american_puts_01.json";
const std::filesystem::path price_preview_path =
    "previews/price/heston/american_puts/"
    "heston_01__american_puts_01__01.json";
const std::filesystem::path price_catalog_path =
    "catalog/price/heston/american_puts/"
    "heston_01__american_puts_01__01/dataset.yaml";
const std::vector<std::filesystem::path> catalog_paths = {
    "catalog/model/heston/heston_01/dataset.yaml",
    "catalog/product/european_calls/european_calls_01/dataset.yaml",
    "catalog/product/american_puts/american_puts_01/dataset.yaml",
    "catalog/price/heston/european_calls/"
    "heston_01__european_calls_01__01/dataset.yaml",
    "catalog/price/heston/european_calls/"
    "heston_01__european_calls_01__02/dataset.yaml",
    price_catalog_path,
};

// Stop immediately with a readable invariant name.
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// Parse one dataset JSON document with an explicit path in any error.
nlohmann::json read_json(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot open dataset JSON: " + path.string());
    }
    return nlohmann::json::parse(stream);
}

// Read one compact catalog entry for a few exact recipe assertions.
std::string read_text(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot open catalog YAML: " + path.string());
    }
    return {
        std::istreambuf_iterator<char>(stream),
        std::istreambuf_iterator<char>()
    };
}

// Require YAML locations to identify catalog directories, never local files.
void validate_catalog_locations(const std::filesystem::path& path) {
    const std::string catalog = read_text(path);
    require(
        catalog.find("preview:") == std::string::npos
            && catalog.find("generation_script:") == std::string::npos
            && catalog.find("source_files:") == std::string::npos
            && catalog.find("dataset: \"datasets/") == std::string::npos
            && catalog.find("catalog: \"catalog/") != std::string::npos,
        "catalog YAML exposes a local file instead of a catalog directory"
    );
    require(
        catalog.find(".yaml\"") == std::string::npos
            && catalog.find(".cpp\"") == std::string::npos,
        "catalog YAML contains a file path"
    );
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

// Load versioned previews and verify both supported price constructions.
int main() {
    using namespace ai_factory::workbench;

    const std::vector<heston::HestonModelParameters> models =
        heston::load_heston(model_preview_path);
    const std::vector<products::AmericanPutInput> products =
        products::load_american_puts(product_preview_path);

    require(models.size() == 100U, "Heston preview row count is not 100");
    require(
        products.size() == 100U,
        "American-put preview row count is not 100"
    );
    for (const auto& model : models) validate_model(model);
    for (const auto& product : products) validate_product(product);

    require(
        datasets::price_row_count(
            models.size(),
            products.size(),
            datasets::PriceConstruction::Aligned
        ) == 100U,
        "aligned preview count is not 100"
    );
    require(
        datasets::price_row_count(
            models.size(),
            products.size(),
            datasets::PriceConstruction::CartesianProduct
        ) == 10'000U,
        "Cartesian preview count is not 10000"
    );

    const nlohmann::json model_document = read_json(model_preview_path);
    const nlohmann::json product_document = read_json(product_preview_path);
    const nlohmann::json result_document = read_json(price_preview_path);
    require(
        result_document.at("database_id") == "heston_01__american_puts_01__01",
        "unexpected American-put price dataset id"
    );
    require(
        result_document.at("row_count") == 100U
            && result_document.at("source_row_count") == 1'000U
            && result_document.at("results").size() == 100U,
        "American-put preview does not contain 100 of 1000 rows"
    );
    require(
        result_document.at("model_dataset").at("id") == "heston_01"
            && result_document.at("product_dataset").at("id")
                == "american_puts_01",
        "American-put preview references unexpected input datasets"
    );
    require(
        result_document.at("timing").at("wall_seconds").get<double>() > 0.0
            && result_document.at("timing")
                   .at("kernel_seconds")
                   .get<double>() > 0.0,
        "American-put production timings are not positive"
    );

    for (std::size_t row = 0U; row < 100U; ++row) {
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
            "American-put price row references unexpected inputs"
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

    const std::string catalog = read_text(price_catalog_path);
    require(
        catalog.find(
            "monte_carlo_paths_per_price: 1048576"
        ) != std::string::npos,
        "American-put YAML does not record 1048576 paths per price"
    );
    require(
        catalog.find("blocks_per_price: 128") != std::string::npos,
        "American-put YAML does not record 128 blocks per price"
    );
    require(
        catalog.find(
            "regression_basis: \"Two-factor Laguerre degree 2\""
        ) != std::string::npos,
        "American-put YAML does not identify its regression basis"
    );
    require(
        catalog.find(
            "catalog: \"catalog/price/heston/american_puts/"
            "heston_01__american_puts_01__01\""
        ) != std::string::npos,
        "American-put YAML does not expose its catalog directory"
    );
    require(
        catalog.find(
            "url: \"https://datasets.ai-factory.example/"
        ) != std::string::npos,
        "American-put catalog does not expose a dataset URL"
    );
    for (const auto& path : catalog_paths) validate_catalog_locations(path);
    require(
        catalog.find("batch_count:") == std::string::npos
            && catalog.find("kernel_launch_count:") == std::string::npos
            && catalog.find("maximum_prices_per_batch:") == std::string::npos
            && catalog.find("workspace_bytes:") == std::string::npos,
        "American-put YAML exposes internal batching metadata"
    );
}
