// Validate versioned YAML catalogs without requiring complete local datasets.
#include "tools/datasets/dataset.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const std::filesystem::path price_catalog_path =
    "catalog/price/heston/american_puts/"
    "heston_01__american_puts_01__01/dataset.yaml";
const std::vector<std::filesystem::path> catalog_paths = {
    "catalog/curve/nelson_siegel/nelson_siegel_01/dataset.yaml",
    "catalog/model/heston/heston_01/dataset.yaml",
    "catalog/model/hull_white/hull_white_01/dataset.yaml",
    "catalog/model/ornstein_uhlenbeck/ornstein_uhlenbeck_01/dataset.yaml",
    "catalog/product/equity/european_calls/european_calls_01/dataset.yaml",
    "catalog/product/equity/asian_calls/asian_calls_01/dataset.yaml",
    "catalog/product/equity/lookback_options/lookback_options_01/dataset.yaml",
    "catalog/product/equity/american_puts/american_puts_01/dataset.yaml",
    "catalog/product/fixed_income/caplets/caplets_01/dataset.yaml",
    "catalog/price/heston/european_calls/"
    "heston_01__european_calls_01__01/dataset.yaml",
    "catalog/price/heston/european_calls/"
    "heston_01__european_calls_01__02/dataset.yaml",
    "catalog/price/heston/asian_calls/"
    "heston_01__asian_calls_01__01/dataset.yaml",
    "catalog/price/heston/lookback_options/"
    "heston_01__lookback_options_01__01/dataset.yaml",
    "catalog/price/hull_white/nelson_siegel/caplets/"
    "hull_white_01__nelson_siegel_01__caplets_01__01/dataset.yaml",
    "catalog/price/ornstein_uhlenbeck/caplets/"
    "ornstein_uhlenbeck_01__caplets_01__01/dataset.yaml",
    price_catalog_path,
};

// Stop immediately with a readable invariant name.
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// Read one compact catalog entry for exact recipe assertions.
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

// Count one YAML key to reject nested or duplicated dataset metadata.
std::size_t occurrence_count(
    const std::string& text,
    const std::string& value
) {
    std::size_t count = 0U;
    std::size_t offset = 0U;
    while ((offset = text.find(value, offset)) != std::string::npos) {
        ++count;
        offset += value.size();
    }
    return count;
}

// Require catalog URLs and directories without exposing local data files.
void validate_catalog_locations(const std::filesystem::path& path) {
    const std::string catalog = read_text(path);
    require(
        catalog.find("generation_script:") == std::string::npos
            && catalog.find("source_files:") == std::string::npos
            && catalog.find("dataset: \"datasets/") == std::string::npos
            && catalog.find("catalog: \"catalog/") != std::string::npos
            && catalog.find("url: \"http") != std::string::npos,
        "catalog YAML exposes invalid dataset locations"
    );
    require(
        catalog.find(".yaml\"") == std::string::npos
            && catalog.find(".cpp\"") == std::string::npos,
        "catalog YAML contains a file path"
    );
    const std::size_t url_position = catalog.find("\nurl: ");
    const std::size_t row_count_position =
        catalog.find("\nrow_count: ", url_position);
    require(
        url_position != std::string::npos
            && row_count_position == catalog.find('\n', url_position + 1U)
            && occurrence_count(catalog, "\nrow_count: ") == 1U,
        "catalog YAML must place one root row_count immediately after url"
    );
}

}  // namespace

// Verify construction counts and the stable public catalog contract.
int main() {
    using namespace ai_factory::workbench;

    require(
        datasets::price_row_count(
            100U, 100U, datasets::PriceConstruction::Aligned
        ) == 100U,
        "aligned construction count is not 100"
    );
    require(
        datasets::price_row_count(
            100U, 100U, datasets::PriceConstruction::CartesianProduct
        ) == 10'000U,
        "Cartesian construction count is not 10000"
    );
    require(
        datasets::price_row_count(
            100U, 100U, 100U, datasets::PriceConstruction::Aligned
        ) == 100U,
        "three-input aligned construction count is not 100"
    );
    require(
        datasets::price_row_count(
            100U,
            100U,
            100U,
            datasets::PriceConstruction::CartesianProduct
        ) == 1'000'000U,
        "three-input Cartesian construction count is not 1000000"
    );

    const std::string catalog = read_text(price_catalog_path);
    require(
        catalog.find("monte_carlo_paths_per_price: 1048576")
            != std::string::npos,
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
    for (const auto& path : catalog_paths) validate_catalog_locations(path);
    require(
        catalog.find("batch_count:") == std::string::npos
            && catalog.find("kernel_launch_count:") == std::string::npos
            && catalog.find("maximum_prices_per_batch:") == std::string::npos
            && catalog.find("workspace_bytes:") == std::string::npos,
        "American-put YAML exposes internal batching metadata"
    );
}
