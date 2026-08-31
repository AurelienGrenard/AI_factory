// Generate options on zero-coupon bonds with broad strike coverage.
#include "tools/datasets/parameter_dataset.hpp"
#include "common/dataset_validation.hpp"

#include <filesystem>
#include <string>
#include <utility>

// Generate the zero-coupon bond option dataset and catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/product/zero_coupon_bond_option/"
        "zero_coupon_bond_options_01.json";
    const std::filesystem::path catalog_path =
        "catalog/product/zero_coupon_bond_option/"
        "zero_coupon_bond_options_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/product/"
        "zero_coupon_bond_options/zero_coupon_bond_options_01.json";

    std::vector<float> core_strikes;
    core_strikes.reserve(25U);
    for (const float coordinate : linear_grid(-1.0f, 1.0f, 25U)) {
        const float cube = coordinate * coordinate * coordinate;
        core_strikes.push_back(
            coordinate < 0.0f ? 0.97f + 0.17f * cube
                              : 0.97f + 0.13f * cube
        );
    }
    const auto regime = [](
        const std::vector<std::uint32_t>& option_expiries,
        const std::vector<std::uint32_t>& bond_tenors,
                           const std::vector<float>& strikes,
                           const std::string& description) {
        GeneratedRows generated;
        for (const std::uint32_t option_expiry_days : option_expiries) {
            for (const std::uint32_t bond_tenor : bond_tenors) {
                for (const float strike : strikes) {
                    generated.rows.push_back({
                    {"notional", 1.0f},
                    {"strike", strike},
                    {"option_expiry", option_expiry_days},
                    {"bond_maturity", option_expiry_days + bond_tenor},
                    });
                }
            }
        }
        generated.construction = {
            {"method", "Cartesian grid"},
            {"rule", "Every option expiry, bond tenor, and strike combination."},
            {"description", description},
            {"option_expiries", option_expiries},
            {"bond_tenors", bond_tenors},
            {"strikes", strikes},
            {"notional", 1.0},
        };
        return generated;
    };
    GeneratedRows core = regime(
        linear_business_day_grid(63U, 1260U, 18U),
        {126U, 252U},
        core_strikes,
        "Representative expiries and strikes concentrated around 0.97."
    );
    GeneratedRows stress = regime(
        linear_business_day_grid(5U, 3780U, 10U),
        {21U, 2520U},
        {0.20f, 0.60f, 0.97f, 1.25f, 1.75f},
        "Very short/long expiries and tenors with unusually wide bond strikes."
    );
    const GeneratedRows rows = core_stress_rows(
        std::move(core), std::move(stress)
    );

    write_product_dataset(
        "zero_coupon_bond_options_01",
        "Zero-Coupon Bond Options",
        dataset_path,
        catalog_path,
        url,
        {
            {"notional", "Normalized contract notional."},
            {"strike", "Strike expressed as a zero-coupon bond price."},
            {"option_expiry", "Option expiry S in business days."},
            {"bond_maturity", "Underlying bond maturity T > S in business days."},
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
