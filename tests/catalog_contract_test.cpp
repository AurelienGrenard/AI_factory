// Validate versioned YAML catalogs without requiring complete local datasets.
#include "common/check_cuda.cuh"
#include "common/result_index.cuh"
#include "common/time_configuration.cuh"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const std::filesystem::path price_catalog_path =
    "catalog/model/equity/heston/prices/american_puts/"
    "heston_01__american_puts_01__01/dataset.yaml";
const std::filesystem::path cir_catalog_path =
    "catalog/model/fixed_income/cir/parameters/cir_01/dataset.yaml";
const std::vector<std::filesystem::path> catalog_paths = {
    "catalog/curve/nelson_siegel/nelson_siegel_01/dataset.yaml",
    "catalog/curve/svensson/svensson_01/dataset.yaml",
    "catalog/model/equity/heston/parameters/heston_01/dataset.yaml",
    "catalog/model/equity/bates/parameters/bates_01/dataset.yaml",
    "catalog/model/equity/variance_gamma/parameters/variance_gamma_01/dataset.yaml",
    "catalog/model/equity/normal_inverse_gaussian/"
    "parameters/normal_inverse_gaussian_01/dataset.yaml",
    "catalog/model/equity/rough_bergomi/parameters/rough_bergomi_01/dataset.yaml",
    "catalog/model/equity/rough_heston/parameters/rough_heston_01/dataset.yaml",
    "catalog/model/equity/rough_sabr/parameters/rough_sabr_01/dataset.yaml",
    "catalog/model/fixed_income/g2/parameters/g2_01/dataset.yaml",
    "catalog/model/fixed_income/g2_plus_plus/parameters/g2_plus_plus_01/dataset.yaml",
    "catalog/model/fixed_income/hull_white/parameters/hull_white_01/dataset.yaml",
    "catalog/model/fixed_income/ornstein_uhlenbeck/parameters/ornstein_uhlenbeck_01/dataset.yaml",
    "catalog/model/fixed_income/vasicek/parameters/vasicek_01/dataset.yaml",
    cir_catalog_path,
    "catalog/product/equity/european_options/"
    "european_options_01/dataset.yaml",
    "catalog/product/equity/asian_options/asian_options_01/dataset.yaml",
    "catalog/product/equity/digital_options/digital_options_01/dataset.yaml",
    "catalog/product/equity/asset_or_nothing_options/"
    "asset_or_nothing_options_01/dataset.yaml",
    "catalog/product/equity/gap_options/gap_call_options_01/dataset.yaml",
    "catalog/product/equity/gap_options/gap_put_options_01/dataset.yaml",
    "catalog/product/equity/straddles/straddles_01/dataset.yaml",
    "catalog/product/equity/lookback_options/lookback_options_01/dataset.yaml",
    "catalog/product/equity/american_options/american_options_01/dataset.yaml",
    "catalog/product/equity/phoenix_autocalls/"
    "phoenix_autocalls_01/dataset.yaml",
    "catalog/product/equity/phoenix_memory_autocalls/"
    "phoenix_memory_autocalls_01/dataset.yaml",
    "catalog/product/equity/athena_autocalls/"
    "athena_autocalls_01/dataset.yaml",
    "catalog/product/equity/cliquets/cliquets_01/dataset.yaml",
    "catalog/product/equity/range_accruals/"
    "range_accruals_01/dataset.yaml",
    "catalog/product/equity/geometric_asian_options/"
    "geometric_asian_options_01/dataset.yaml",
    "catalog/product/equity/forward_start_options/"
    "forward_start_options_01/dataset.yaml",
    "catalog/product/equity/up_and_out_options/"
    "up_and_out_options_01/dataset.yaml",
    "catalog/product/equity/down_and_out_options/"
    "down_and_out_options_01/dataset.yaml",
    "catalog/product/equity/up_and_in_options/"
    "up_and_in_options_01/dataset.yaml",
    "catalog/product/equity/down_and_in_options/"
    "down_and_in_options_01/dataset.yaml",
    "catalog/product/equity/up_one_touches/"
    "up_one_touches_01/dataset.yaml",
    "catalog/product/equity/up_no_touches/"
    "up_no_touches_01/dataset.yaml",
    "catalog/product/equity/double_knock_out_options/"
    "double_knock_out_options_01/dataset.yaml",
    "catalog/product/fixed_income/rate_options/rate_options_01/dataset.yaml",
    "catalog/product/fixed_income/zero_coupon_bond_options/"
    "zero_coupon_bond_options_01/dataset.yaml",
    "catalog/product/fixed_income/european_swaptions/"
    "european_swaptions_01/dataset.yaml",
    "catalog/product/fixed_income/bermudan_swaptions/"
    "bermudan_swaptions_01/dataset.yaml",
    "catalog/model/equity/heston/prices/european_calls/"
    "heston_01__european_calls_01__01/dataset.yaml",
    "catalog/model/equity/heston/prices/european_puts/"
    "heston_01__european_puts_01__01/dataset.yaml",
    "catalog/model/equity/rough_bergomi/prices/european_calls/"
    "rough_bergomi_01__european_calls_01__01/dataset.yaml",
    "catalog/model/equity/rough_bergomi/prices/european_puts/"
    "rough_bergomi_01__european_puts_01__01/dataset.yaml",
    "catalog/model/equity/heston/prices/asian_calls/"
    "heston_01__asian_calls_01__01/dataset.yaml",
    "catalog/model/equity/heston/prices/athena_autocalls/"
    "heston_01__athena_autocalls_01__01/dataset.yaml",
    "catalog/model/equity/heston/prices/phoenix_autocalls/"
    "heston_01__phoenix_autocalls_01__01/dataset.yaml",
    "catalog/model/equity/heston/prices/phoenix_memory_autocalls/"
    "heston_01__phoenix_memory_autocalls_01__01/dataset.yaml",
    "catalog/model/equity/heston/prices/cliquets/"
    "heston_01__cliquets_01__01/dataset.yaml",
    "catalog/model/equity/heston/prices/range_accruals/"
    "heston_01__range_accruals_01__01/dataset.yaml",
    "catalog/model/equity/heston/prices/asian_puts/"
    "heston_01__asian_puts_01__01/dataset.yaml",
    "catalog/model/equity/heston/prices/digital_calls/"
    "heston_01__digital_calls_01__01/dataset.yaml",
    "catalog/model/equity/heston/prices/digital_puts/"
    "heston_01__digital_puts_01__01/dataset.yaml",
    "catalog/model/equity/heston/prices/asset_or_nothing_calls/"
    "heston_01__asset_or_nothing_calls_01__01/dataset.yaml",
    "catalog/model/equity/heston/prices/asset_or_nothing_puts/"
    "heston_01__asset_or_nothing_puts_01__01/dataset.yaml",
    "catalog/model/equity/heston/prices/gap_calls/"
    "heston_01__gap_calls_01__01/dataset.yaml",
    "catalog/model/equity/heston/prices/gap_puts/"
    "heston_01__gap_puts_01__01/dataset.yaml",
    "catalog/model/equity/heston/prices/straddles/"
    "heston_01__straddles_01__01/dataset.yaml",
    "catalog/model/equity/heston/prices/lookback_options/"
    "heston_01__lookback_options_01__01/dataset.yaml",
    "catalog/model/equity/heston/prices/geometric_asian_calls/"
    "heston_01__geometric_asian_calls_01__01/dataset.yaml",
    "catalog/model/equity/heston/prices/geometric_asian_puts/"
    "heston_01__geometric_asian_puts_01__01/dataset.yaml",
    "catalog/model/equity/heston/prices/forward_start_calls/"
    "heston_01__forward_start_calls_01__01/dataset.yaml",
    "catalog/model/equity/heston/prices/forward_start_puts/"
    "heston_01__forward_start_puts_01__01/dataset.yaml",
    "catalog/model/equity/heston/prices/up_and_out_calls/"
    "heston_01__up_and_out_calls_01__01/dataset.yaml",
    "catalog/model/equity/heston/prices/down_and_out_puts/"
    "heston_01__down_and_out_puts_01__01/dataset.yaml",
    "catalog/model/equity/heston/prices/up_and_in_calls/"
    "heston_01__up_and_in_calls_01__01/dataset.yaml",
    "catalog/model/equity/heston/prices/down_and_in_puts/"
    "heston_01__down_and_in_puts_01__01/dataset.yaml",
    "catalog/model/equity/heston/prices/up_one_touches/"
    "heston_01__up_one_touches_01__01/dataset.yaml",
    "catalog/model/equity/heston/prices/up_no_touches/"
    "heston_01__up_no_touches_01__01/dataset.yaml",
    "catalog/model/equity/heston/prices/double_knock_out_calls/"
    "heston_01__double_knock_out_calls_01__01/dataset.yaml",
    "catalog/model/equity/heston/prices/double_knock_out_puts/"
    "heston_01__double_knock_out_puts_01__01/dataset.yaml",
    "catalog/model/equity/heston/prices/american_calls/"
    "heston_01__american_calls_01__01/dataset.yaml",
    "catalog/model/fixed_income/g2/prices/caplets/"
    "g2_01__caplets_01__01/dataset.yaml",
    "catalog/model/fixed_income/g2/prices/floorlets/"
    "g2_01__floorlets_01__01/dataset.yaml",
    "catalog/model/fixed_income/g2/prices/zero_coupon_bond_calls/"
    "g2_01__zero_coupon_bond_calls_01__01/dataset.yaml",
    "catalog/model/fixed_income/g2/prices/zero_coupon_bond_puts/"
    "g2_01__zero_coupon_bond_puts_01__01/dataset.yaml",
    "catalog/model/fixed_income/g2_plus_plus/prices/nelson_siegel/caplets/"
    "g2_plus_plus_01__nelson_siegel_01__caplets_01__01/dataset.yaml",
    "catalog/model/fixed_income/g2_plus_plus/prices/nelson_siegel/floorlets/"
    "g2_plus_plus_01__nelson_siegel_01__floorlets_01__01/dataset.yaml",
    "catalog/model/fixed_income/g2_plus_plus/prices/nelson_siegel/zero_coupon_bond_calls/"
    "g2_plus_plus_01__nelson_siegel_01__zero_coupon_bond_calls_01__01/"
    "dataset.yaml",
    "catalog/model/fixed_income/g2_plus_plus/prices/nelson_siegel/zero_coupon_bond_puts/"
    "g2_plus_plus_01__nelson_siegel_01__zero_coupon_bond_puts_01__01/"
    "dataset.yaml",
    "catalog/model/fixed_income/g2_plus_plus/prices/svensson/caplets/"
    "g2_plus_plus_01__svensson_01__caplets_01__01/dataset.yaml",
    "catalog/model/fixed_income/g2_plus_plus/prices/svensson/floorlets/"
    "g2_plus_plus_01__svensson_01__floorlets_01__01/dataset.yaml",
    "catalog/model/fixed_income/g2_plus_plus/prices/svensson/zero_coupon_bond_calls/"
    "g2_plus_plus_01__svensson_01__zero_coupon_bond_calls_01__01/"
    "dataset.yaml",
    "catalog/model/fixed_income/g2_plus_plus/prices/svensson/zero_coupon_bond_puts/"
    "g2_plus_plus_01__svensson_01__zero_coupon_bond_puts_01__01/"
    "dataset.yaml",
    "catalog/model/fixed_income/hull_white/prices/nelson_siegel/caplets/"
    "hull_white_01__nelson_siegel_01__caplets_01__01/dataset.yaml",
    "catalog/model/fixed_income/hull_white/prices/nelson_siegel/floorlets/"
    "hull_white_01__nelson_siegel_01__floorlets_01__01/dataset.yaml",
    "catalog/model/fixed_income/hull_white/prices/nelson_siegel/zero_coupon_bond_calls/"
    "hull_white_01__nelson_siegel_01__zero_coupon_bond_calls_01__01/dataset.yaml",
    "catalog/model/fixed_income/hull_white/prices/nelson_siegel/zero_coupon_bond_puts/"
    "hull_white_01__nelson_siegel_01__zero_coupon_bond_puts_01__01/dataset.yaml",
    "catalog/model/fixed_income/hull_white/prices/svensson/caplets/"
    "hull_white_01__svensson_01__caplets_01__01/dataset.yaml",
    "catalog/model/fixed_income/hull_white/prices/svensson/floorlets/"
    "hull_white_01__svensson_01__floorlets_01__01/dataset.yaml",
    "catalog/model/fixed_income/hull_white/prices/svensson/zero_coupon_bond_calls/"
    "hull_white_01__svensson_01__zero_coupon_bond_calls_01__01/dataset.yaml",
    "catalog/model/fixed_income/hull_white/prices/svensson/zero_coupon_bond_puts/"
    "hull_white_01__svensson_01__zero_coupon_bond_puts_01__01/dataset.yaml",
    "catalog/model/fixed_income/ornstein_uhlenbeck/prices/caplets/"
    "ornstein_uhlenbeck_01__caplets_01__01/dataset.yaml",
    "catalog/model/fixed_income/ornstein_uhlenbeck/prices/floorlets/"
    "ornstein_uhlenbeck_01__floorlets_01__01/dataset.yaml",
    "catalog/model/fixed_income/ornstein_uhlenbeck/prices/zero_coupon_bond_calls/"
    "ornstein_uhlenbeck_01__zero_coupon_bond_calls_01__01/dataset.yaml",
    "catalog/model/fixed_income/ornstein_uhlenbeck/prices/zero_coupon_bond_puts/"
    "ornstein_uhlenbeck_01__zero_coupon_bond_puts_01__01/dataset.yaml",
    "catalog/model/fixed_income/vasicek/prices/caplets/"
    "vasicek_01__caplets_01__01/dataset.yaml",
    "catalog/model/fixed_income/vasicek/prices/floorlets/"
    "vasicek_01__floorlets_01__01/dataset.yaml",
    "catalog/model/fixed_income/vasicek/prices/zero_coupon_bond_calls/"
    "vasicek_01__zero_coupon_bond_calls_01__01/dataset.yaml",
    "catalog/model/fixed_income/vasicek/prices/zero_coupon_bond_puts/"
    "vasicek_01__zero_coupon_bond_puts_01__01/dataset.yaml",
    "catalog/model/fixed_income/cir/prices/caplets/"
    "cir_01__caplets_01__01/dataset.yaml",
    "catalog/model/fixed_income/cir/prices/floorlets/"
    "cir_01__floorlets_01__01/dataset.yaml",
    "catalog/model/fixed_income/cir/prices/zero_coupon_bond_calls/"
    "cir_01__zero_coupon_bond_calls_01__01/dataset.yaml",
    "catalog/model/fixed_income/cir/prices/zero_coupon_bond_puts/"
    "cir_01__zero_coupon_bond_puts_01__01/dataset.yaml",
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

// Product and price catalogs declare one shared business-day convention.
void validate_time_convention_metadata(const std::filesystem::path& path) {
    const std::string catalog = read_text(path);
    require(
        occurrence_count(catalog, "\ntime_convention:\n") == 1U
            && catalog.find(
                "\ntime_convention:\n"
                "  unit: \"business_day\"\n"
                "  days_per_year: 252\n"
            ) != std::string::npos,
        "product/price YAML must declare the 252-business-day convention"
    );
}

// Every price catalog must disclose whether an independent reference passed.
void validate_price_validation_metadata(const std::filesystem::path& path) {
    const std::string catalog = read_text(path);
    require(
        occurrence_count(catalog, "\nvalidation:\n") == 1U,
        "price YAML must contain exactly one root validation block"
    );
    const std::size_t validation_start = catalog.find("\nvalidation:\n") + 1U;
    std::size_t validation_end = catalog.size();
    for (std::size_t position = validation_start + 12U;
         position + 1U < catalog.size();
         ++position) {
        if (catalog[position] == '\n'
            && catalog[position + 1U] != ' '
            && catalog[position + 1U] != '\n') {
            validation_end = position + 1U;
            break;
        }
    }
    const std::string validation = catalog.substr(
        validation_start, validation_end - validation_start
    );
    require(
        validation.find("\n  method:") == std::string::npos
            && validation.find("\n    method:") == std::string::npos
            && validation.find("relationship:") == std::string::npos,
        "price YAML must fuse the reference and method and omit relationship"
    );

    const bool cached_reference =
        validation.find("  status: \"available\"") != std::string::npos
        && validation.find("  dataset: \"validation/datasets/price/")
            != std::string::npos;
    if (cached_reference) {
        const std::string dataset_marker =
            "  dataset: \"validation/datasets/price/";
        const std::size_t dataset_start =
            validation.find(dataset_marker) + std::string("  dataset: \"").size();
        const std::size_t dataset_end = validation.find('"', dataset_start);
        require(
            dataset_end != std::string::npos
                && std::filesystem::is_regular_file(
                    validation.substr(dataset_start, dataset_end - dataset_start)
                ),
            "cached validation dataset does not exist"
        );
        require(
            validation.find("  verified: true") != std::string::npos
                && validation.find("  core:") == std::string::npos
                && validation.find("  stress:") == std::string::npos
                && validation.find("  reference:") == std::string::npos
                && validation.find("  notebook:") == std::string::npos,
            "cached validation metadata must be compact and verified"
        );
        return;
    }

    const bool split_regimes = validation.find("  core:\n") != std::string::npos;
    if (split_regimes) {
        require(
            validation.find("  stress:\n") != std::string::npos
                && validation.find("    row_count: 900") != std::string::npos
                && validation.find("    row_count: 100") != std::string::npos,
            "split validation metadata must declare core and stress row counts"
        );
        if (!(occurrence_count(validation, "\n    reference: \"") >= 2U
              && (
                  occurrence_count(validation, "\n    verified: true") == 2U
                  || validation.find("\n  verified: true")
                      != std::string::npos
                  || validation.find("\n  verified: false")
                      != std::string::npos
              ))) {
            throw std::runtime_error(
                "split validation metadata must document both selected "
                "backends: " + path.string()
            );
        }
        return;
    }
    const bool premia = validation.find("  reference: \"Premia (")
        != std::string::npos;
    const bool quantlib = validation.find("  reference: \"QuantLib (")
        != std::string::npos;
    const bool none = validation.find("  reference: \"none\"")
        != std::string::npos;
    require(
        static_cast<unsigned int>(premia)
            + static_cast<unsigned int>(quantlib)
            + static_cast<unsigned int>(none) == 1U,
        "price YAML validation reference is missing or unsupported"
    );
    require(
        none
            ? validation.find("  verified: false") != std::string::npos
            : validation.find("  verified: true") != std::string::npos,
        "price YAML validation status contradicts its reference"
    );
}

}  // namespace

// Verify construction counts and the stable public catalog contract.
int main() {
    using namespace ai_factory::workbench;

    for (const std::uint32_t days_per_year : {252U, 360U, 365U}) {
        const time::DayFractionTimeConfiguration time_configuration{
            1.0f / static_cast<float>(days_per_year)
        };
        require(
            std::fabs(
                time::year_fraction(days_per_year, time_configuration) - 1.0f
            ) <= 2.0f * std::numeric_limits<float>::epsilon(),
            "contract days and model years disagree at a day-count boundary"
        );
    }

    require(
        ai_factory::workbench::price_row_count(
            100U, 100U, ai_factory::workbench::PriceConstruction::Aligned
        ) == 100U,
        "aligned construction count is not 100"
    );
    require(
        ai_factory::workbench::price_row_count(
            100U, 100U, ai_factory::workbench::PriceConstruction::CartesianProduct
        ) == 10'000U,
        "Cartesian construction count is not 10000"
    );
    require(
        ai_factory::workbench::price_row_count(
            100U, 100U, 100U, ai_factory::workbench::PriceConstruction::Aligned
        ) == 100U,
        "three-input aligned construction count is not 100"
    );
    require(
        ai_factory::workbench::price_row_count(
            100U,
            100U,
            100U,
            ai_factory::workbench::PriceConstruction::CartesianProduct
        ) == 1'000'000U,
        "three-input Cartesian construction count is not 1000000"
    );
    const ModelProductIndices product_indices =
        decode_model_product_result_index(
            7U, 3U, PriceConstruction::CartesianProduct
        );
    require(
        product_indices.model_index == 2U
            && product_indices.product_index == 1U,
        "Cartesian model/product IDs do not follow model-major order"
    );
    const ModelProductIndices product_indices_32 =
        decode_model_product_result_index_32(
            7U, 3U, PriceConstruction::CartesianProduct
        );
    require(
        product_indices_32.model_index == product_indices.model_index
            && product_indices_32.product_index
                == product_indices.product_index,
        "validated uint32 model/product decoding changed canonical IDs"
    );
    const ModelCurveProductIndices curve_indices =
        decode_model_curve_product_result_index(
            17U, 2U, 4U, PriceConstruction::CartesianProduct
        );
    require(
        curve_indices.model_index == 2U
            && curve_indices.curve_index == 0U
            && curve_indices.product_index == 1U,
        "Cartesian model/curve/product IDs do not follow canonical order"
    );
    const ModelCurveProductIndices curve_indices_32 =
        decode_model_curve_product_result_index_32(
            17U, 2U, 4U, PriceConstruction::CartesianProduct
        );
    require(
        curve_indices_32.model_index == curve_indices.model_index
            && curve_indices_32.curve_index == curve_indices.curve_index
            && curve_indices_32.product_index == curve_indices.product_index,
        "validated uint32 model/curve/product decoding changed canonical IDs"
    );
    const ModelCurveProductIndices aligned_indices =
        decode_model_curve_product_result_index(
            7U, 99U, 99U, PriceConstruction::Aligned
        );
    require(
        aligned_indices.model_index == 7U
            && aligned_indices.curve_index == 7U
            && aligned_indices.product_index == 7U,
        "aligned construction does not preserve row IDs"
    );
    bool rejected_overflow = false;
    try {
        static_cast<void>(price_row_count(
            std::numeric_limits<std::size_t>::max(),
            2U,
            PriceConstruction::CartesianProduct
        ));
    } catch (const std::overflow_error&) {
        rejected_overflow = true;
    }
    require(rejected_overflow, "Cartesian cardinality accepted size_t overflow");
    bool rejected_device_index_overflow = false;
    try {
        validate_model_product_construction(
            1U,
            static_cast<std::size_t>(
                std::numeric_limits<std::uint32_t>::max()
            ) + 1U,
            PriceConstruction::CartesianProduct,
            static_cast<std::size_t>(
                std::numeric_limits<std::uint32_t>::max()
            ) + 1U
        );
    } catch (const std::overflow_error& error) {
        rejected_device_index_overflow =
            std::string(error.what()).find("split") != std::string::npos;
    }
    require(
        rejected_device_index_overflow,
        "CUDA construction accepted an unrepresentable device row index"
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
    for (const auto& path : catalog_paths) {
        validate_catalog_locations(path);
        const std::string generic_path = path.generic_string();
        if (generic_path.find("/parameters/") != std::string::npos
            || generic_path.rfind("catalog/product/", 0U) == 0U) {
            const std::string parameter_catalog = read_text(path);
            require(
                parameter_catalog.find("core_share: 0.9")
                        != std::string::npos
                    && parameter_catalog.find("stress_share: 0.1")
                        != std::string::npos,
                "model/product YAML does not document its 90/10 split"
            );
        }
        if (generic_path.rfind("catalog/product/", 0U) == 0U
            || generic_path.find("/prices/") != std::string::npos) {
            validate_time_convention_metadata(path);
        }
    }
    const std::string cir_catalog = read_text(cir_catalog_path);
    require(
        cir_catalog.find(
            "definition: \"2 * mean_reversion * long_term_mean / volatility^2\""
        ) != std::string::npos
            && cir_catalog.find("guaranteed_minimum: \"1 / 6\"")
                != std::string::npos
            && cir_catalog.find("guaranteed_maximum: 10.0")
                != std::string::npos
            && cir_catalog.find("guaranteed_minimum: \"1 / 10\"")
                != std::string::npos
            && cir_catalog.find("guaranteed_maximum: 16.0")
                != std::string::npos,
        "CIR YAML does not document its core/stress Feller-ratio bounds"
    );
    require(
        !std::filesystem::exists("catalog/model/sample_generation"),
        "model-sample recipes must remain in their adjacent generators"
    );
    for (const auto& entry : std::filesystem::recursive_directory_iterator(
             "catalog/model")) {
        if (entry.is_regular_file()
            && entry.path().filename() == "dataset.yaml"
            && entry.path().generic_string().find("/prices/")
                != std::string::npos) {
            validate_catalog_locations(entry.path());
            validate_price_validation_metadata(entry.path());
            validate_time_convention_metadata(entry.path());
        }
        if (entry.is_regular_file()
            && entry.path().filename() == "dataset.yaml"
            && entry.path().generic_string().find("/samples/")
                != std::string::npos) {
            validate_catalog_locations(entry.path());
        }
    }
    require(
        catalog.find("batch_count:") == std::string::npos
            && catalog.find("kernel_launch_count:") == std::string::npos
            && catalog.find("maximum_prices_per_batch:") == std::string::npos
            && catalog.find("workspace_bytes:") == std::string::npos,
        "American-put YAML exposes internal batching metadata"
    );
}
