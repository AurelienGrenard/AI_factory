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
    "catalog/curve/svensson/svensson_01/dataset.yaml",
    "catalog/model/heston/heston_01/dataset.yaml",
    "catalog/model/g2/g2_01/dataset.yaml",
    "catalog/model/g2_plus_plus/g2_plus_plus_01/dataset.yaml",
    "catalog/model/hull_white/hull_white_01/dataset.yaml",
    "catalog/model/ornstein_uhlenbeck/ornstein_uhlenbeck_01/dataset.yaml",
    "catalog/model/vasicek/vasicek_01/dataset.yaml",
    "catalog/product/equity/european_calls/european_calls_01/dataset.yaml",
    "catalog/product/equity/european_puts/european_puts_01/dataset.yaml",
    "catalog/product/equity/asian_calls/asian_calls_01/dataset.yaml",
    "catalog/product/equity/asian_puts/asian_puts_01/dataset.yaml",
    "catalog/product/equity/digital_calls/digital_calls_01/dataset.yaml",
    "catalog/product/equity/digital_puts/digital_puts_01/dataset.yaml",
    "catalog/product/equity/asset_or_nothing_calls/"
    "asset_or_nothing_calls_01/dataset.yaml",
    "catalog/product/equity/asset_or_nothing_puts/"
    "asset_or_nothing_puts_01/dataset.yaml",
    "catalog/product/equity/gap_calls/gap_calls_01/dataset.yaml",
    "catalog/product/equity/gap_puts/gap_puts_01/dataset.yaml",
    "catalog/product/equity/straddles/straddles_01/dataset.yaml",
    "catalog/product/equity/lookback_options/lookback_options_01/dataset.yaml",
    "catalog/product/equity/american_puts/american_puts_01/dataset.yaml",
    "catalog/product/equity/american_calls/american_calls_01/dataset.yaml",
    "catalog/product/equity/phoenix_autocalls/"
    "phoenix_autocalls_01/dataset.yaml",
    "catalog/product/equity/phoenix_memory_autocalls/"
    "phoenix_memory_autocalls_01/dataset.yaml",
    "catalog/product/equity/athena_autocalls/"
    "athena_autocalls_01/dataset.yaml",
    "catalog/product/equity/cliquets/cliquets_01/dataset.yaml",
    "catalog/product/equity/range_accruals/"
    "range_accruals_01/dataset.yaml",
    "catalog/product/equity/geometric_asian_calls/"
    "geometric_asian_calls_01/dataset.yaml",
    "catalog/product/equity/geometric_asian_puts/"
    "geometric_asian_puts_01/dataset.yaml",
    "catalog/product/equity/forward_start_calls/"
    "forward_start_calls_01/dataset.yaml",
    "catalog/product/equity/forward_start_puts/"
    "forward_start_puts_01/dataset.yaml",
    "catalog/product/equity/up_and_out_calls/"
    "up_and_out_calls_01/dataset.yaml",
    "catalog/product/equity/down_and_out_puts/"
    "down_and_out_puts_01/dataset.yaml",
    "catalog/product/equity/up_and_in_calls/"
    "up_and_in_calls_01/dataset.yaml",
    "catalog/product/equity/down_and_in_puts/"
    "down_and_in_puts_01/dataset.yaml",
    "catalog/product/equity/up_one_touches/"
    "up_one_touches_01/dataset.yaml",
    "catalog/product/equity/up_no_touches/"
    "up_no_touches_01/dataset.yaml",
    "catalog/product/equity/double_knock_out_calls/"
    "double_knock_out_calls_01/dataset.yaml",
    "catalog/product/equity/double_knock_out_puts/"
    "double_knock_out_puts_01/dataset.yaml",
    "catalog/product/fixed_income/caplets/caplets_01/dataset.yaml",
    "catalog/product/fixed_income/floorlets/floorlets_01/dataset.yaml",
    "catalog/product/fixed_income/zero_coupon_bond_calls/"
    "zero_coupon_bond_calls_01/dataset.yaml",
    "catalog/product/fixed_income/zero_coupon_bond_puts/"
    "zero_coupon_bond_puts_01/dataset.yaml",
    "catalog/price/heston/european_calls/"
    "heston_01__european_calls_01__01/dataset.yaml",
    "catalog/price/heston/european_calls/"
    "heston_01__european_calls_01__02/dataset.yaml",
    "catalog/price/heston/european_puts/"
    "heston_01__european_puts_01__01/dataset.yaml",
    "catalog/price/heston/asian_calls/"
    "heston_01__asian_calls_01__01/dataset.yaml",
    "catalog/price/heston/athena_autocalls/"
    "heston_01__athena_autocalls_01__01/dataset.yaml",
    "catalog/price/heston/phoenix_autocalls/"
    "heston_01__phoenix_autocalls_01__01/dataset.yaml",
    "catalog/price/heston/phoenix_memory_autocalls/"
    "heston_01__phoenix_memory_autocalls_01__01/dataset.yaml",
    "catalog/price/heston/cliquets/"
    "heston_01__cliquets_01__01/dataset.yaml",
    "catalog/price/heston/range_accruals/"
    "heston_01__range_accruals_01__01/dataset.yaml",
    "catalog/price/heston/asian_puts/"
    "heston_01__asian_puts_01__01/dataset.yaml",
    "catalog/price/heston/digital_calls/"
    "heston_01__digital_calls_01__01/dataset.yaml",
    "catalog/price/heston/digital_puts/"
    "heston_01__digital_puts_01__01/dataset.yaml",
    "catalog/price/heston/asset_or_nothing_calls/"
    "heston_01__asset_or_nothing_calls_01__01/dataset.yaml",
    "catalog/price/heston/asset_or_nothing_puts/"
    "heston_01__asset_or_nothing_puts_01__01/dataset.yaml",
    "catalog/price/heston/gap_calls/"
    "heston_01__gap_calls_01__01/dataset.yaml",
    "catalog/price/heston/gap_puts/"
    "heston_01__gap_puts_01__01/dataset.yaml",
    "catalog/price/heston/straddles/"
    "heston_01__straddles_01__01/dataset.yaml",
    "catalog/price/heston/lookback_options/"
    "heston_01__lookback_options_01__01/dataset.yaml",
    "catalog/price/heston/geometric_asian_calls/"
    "heston_01__geometric_asian_calls_01__01/dataset.yaml",
    "catalog/price/heston/geometric_asian_puts/"
    "heston_01__geometric_asian_puts_01__01/dataset.yaml",
    "catalog/price/heston/forward_start_calls/"
    "heston_01__forward_start_calls_01__01/dataset.yaml",
    "catalog/price/heston/forward_start_puts/"
    "heston_01__forward_start_puts_01__01/dataset.yaml",
    "catalog/price/heston/up_and_out_calls/"
    "heston_01__up_and_out_calls_01__01/dataset.yaml",
    "catalog/price/heston/down_and_out_puts/"
    "heston_01__down_and_out_puts_01__01/dataset.yaml",
    "catalog/price/heston/up_and_in_calls/"
    "heston_01__up_and_in_calls_01__01/dataset.yaml",
    "catalog/price/heston/down_and_in_puts/"
    "heston_01__down_and_in_puts_01__01/dataset.yaml",
    "catalog/price/heston/up_one_touches/"
    "heston_01__up_one_touches_01__01/dataset.yaml",
    "catalog/price/heston/up_no_touches/"
    "heston_01__up_no_touches_01__01/dataset.yaml",
    "catalog/price/heston/double_knock_out_calls/"
    "heston_01__double_knock_out_calls_01__01/dataset.yaml",
    "catalog/price/heston/double_knock_out_puts/"
    "heston_01__double_knock_out_puts_01__01/dataset.yaml",
    "catalog/price/heston/american_calls/"
    "heston_01__american_calls_01__01/dataset.yaml",
    "catalog/price/g2/caplets/"
    "g2_01__caplets_01__01/dataset.yaml",
    "catalog/price/g2/floorlets/"
    "g2_01__floorlets_01__01/dataset.yaml",
    "catalog/price/g2/zero_coupon_bond_calls/"
    "g2_01__zero_coupon_bond_calls_01__01/dataset.yaml",
    "catalog/price/g2/zero_coupon_bond_puts/"
    "g2_01__zero_coupon_bond_puts_01__01/dataset.yaml",
    "catalog/price/g2_plus_plus/nelson_siegel/caplets/"
    "g2_plus_plus_01__nelson_siegel_01__caplets_01__01/dataset.yaml",
    "catalog/price/g2_plus_plus/nelson_siegel/floorlets/"
    "g2_plus_plus_01__nelson_siegel_01__floorlets_01__01/dataset.yaml",
    "catalog/price/g2_plus_plus/nelson_siegel/zero_coupon_bond_calls/"
    "g2_plus_plus_01__nelson_siegel_01__zero_coupon_bond_calls_01__01/"
    "dataset.yaml",
    "catalog/price/g2_plus_plus/nelson_siegel/zero_coupon_bond_puts/"
    "g2_plus_plus_01__nelson_siegel_01__zero_coupon_bond_puts_01__01/"
    "dataset.yaml",
    "catalog/price/g2_plus_plus/svensson/caplets/"
    "g2_plus_plus_01__svensson_01__caplets_01__01/dataset.yaml",
    "catalog/price/g2_plus_plus/svensson/floorlets/"
    "g2_plus_plus_01__svensson_01__floorlets_01__01/dataset.yaml",
    "catalog/price/g2_plus_plus/svensson/zero_coupon_bond_calls/"
    "g2_plus_plus_01__svensson_01__zero_coupon_bond_calls_01__01/"
    "dataset.yaml",
    "catalog/price/g2_plus_plus/svensson/zero_coupon_bond_puts/"
    "g2_plus_plus_01__svensson_01__zero_coupon_bond_puts_01__01/"
    "dataset.yaml",
    "catalog/price/hull_white/nelson_siegel/caplets/"
    "hull_white_01__nelson_siegel_01__caplets_01__01/dataset.yaml",
    "catalog/price/hull_white/nelson_siegel/floorlets/"
    "hull_white_01__nelson_siegel_01__floorlets_01__01/dataset.yaml",
    "catalog/price/hull_white/nelson_siegel/zero_coupon_bond_calls/"
    "hull_white_01__nelson_siegel_01__zero_coupon_bond_calls_01__01/dataset.yaml",
    "catalog/price/hull_white/nelson_siegel/zero_coupon_bond_puts/"
    "hull_white_01__nelson_siegel_01__zero_coupon_bond_puts_01__01/dataset.yaml",
    "catalog/price/hull_white/svensson/caplets/"
    "hull_white_01__svensson_01__caplets_01__01/dataset.yaml",
    "catalog/price/hull_white/svensson/floorlets/"
    "hull_white_01__svensson_01__floorlets_01__01/dataset.yaml",
    "catalog/price/hull_white/svensson/zero_coupon_bond_calls/"
    "hull_white_01__svensson_01__zero_coupon_bond_calls_01__01/dataset.yaml",
    "catalog/price/hull_white/svensson/zero_coupon_bond_puts/"
    "hull_white_01__svensson_01__zero_coupon_bond_puts_01__01/dataset.yaml",
    "catalog/price/ornstein_uhlenbeck/caplets/"
    "ornstein_uhlenbeck_01__caplets_01__01/dataset.yaml",
    "catalog/price/ornstein_uhlenbeck/floorlets/"
    "ornstein_uhlenbeck_01__floorlets_01__01/dataset.yaml",
    "catalog/price/ornstein_uhlenbeck/zero_coupon_bond_calls/"
    "ornstein_uhlenbeck_01__zero_coupon_bond_calls_01__01/dataset.yaml",
    "catalog/price/ornstein_uhlenbeck/zero_coupon_bond_puts/"
    "ornstein_uhlenbeck_01__zero_coupon_bond_puts_01__01/dataset.yaml",
    "catalog/price/vasicek/caplets/"
    "vasicek_01__caplets_01__01/dataset.yaml",
    "catalog/price/vasicek/floorlets/"
    "vasicek_01__floorlets_01__01/dataset.yaml",
    "catalog/price/vasicek/zero_coupon_bond_calls/"
    "vasicek_01__zero_coupon_bond_calls_01__01/dataset.yaml",
    "catalog/price/vasicek/zero_coupon_bond_puts/"
    "vasicek_01__zero_coupon_bond_puts_01__01/dataset.yaml",
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
