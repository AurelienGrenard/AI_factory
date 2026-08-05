// Generate options on zero-coupon bonds with broad strike coverage.
#include "tools/datasets/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <filesystem>
#include <string>

// Generate the zero-coupon bond option dataset and catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/product/fixed_income/zero_coupon_bond_options/"
        "zero_coupon_bond_options_01.json";
    const std::filesystem::path catalog_path =
        "catalog/product/fixed_income/zero_coupon_bond_options/"
        "zero_coupon_bond_options_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/product/"
        "zero_coupon_bond_options/zero_coupon_bond_options_01.json";

    constexpr std::size_t expiry_count = 20U;
    constexpr std::size_t strike_count = 25U;
    const std::vector<float> option_expiries =
        linear_grid(0.25f, 5.0f, expiry_count);
    const std::vector<float> bond_tenors = {0.5f, 1.0f};

    // Concentrate bond strikes near 0.97 while preserving broad tail cases.
    std::vector<float> strikes;
    strikes.reserve(strike_count);
    for (const float coordinate : linear_grid(-1.0f, 1.0f, strike_count)) {
        const float cube = coordinate * coordinate * coordinate;
        strikes.push_back(
            coordinate < 0.0f ? 0.97f + 0.17f * cube
                              : 0.97f + 0.13f * cube
        );
    }

    GeneratedRows rows;
    rows.rows.reserve(
        option_expiries.size() * bond_tenors.size() * strikes.size()
    );
    for (const float option_expiry : option_expiries) {
        for (const float bond_tenor : bond_tenors) {
            for (const float strike : strikes) {
                rows.rows.push_back({
                    {"notional", 1.0f},
                    {"strike", strike},
                    {"option_expiry", option_expiry},
                    {"bond_maturity", option_expiry + bond_tenor},
                });
            }
        }
    }
    rows.construction = {
        {"method", "Cartesian grid"},
        {"rule", "Every option expiry, bond tenor, and strike combination."},
        {"grid", {
            {"option_expiry", {
                {"minimum", "1 / 4"}, {"maximum", 5.0f},
                {"count", expiry_count}, {"spacing", "linear"},
            }},
            {"bond_tenor", {
                {"values", {"1 / 2", "1"}},
                {"count", bond_tenors.size()},
            }},
            {"strike", {
                {"minimum", "0.80"}, {"central_value", "0.97"},
                {"maximum", "1.10"}, {"count", strike_count},
                {"spacing", "cubic concentration around 0.97"},
            }},
            {"notional", {{"value", 1.0f}}},
        }},
    };

    write_product_dataset(
        "zero_coupon_bond_options_01",
        "Zero-Coupon Bond Options",
        dataset_path,
        catalog_path,
        url,
        {
            {"notional", "Normalized contract notional."},
            {"strike", "Strike expressed as a zero-coupon bond price."},
            {"option_expiry", "Option expiry S in years."},
            {"bond_maturity", "Underlying bond maturity T > S in years."},
        },
        {
            {"expression", "N * max(side * (P(S,T) - K), 0), side = +1 call / -1 put"},
            {"payment_time", "S"},
            {"normalization", "N = 1"},
        },
        rows
    );
    validate_product_dataset_file(dataset_path);
}
