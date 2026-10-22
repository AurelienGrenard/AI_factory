// Generate options on zero-coupon bonds with broad strike coverage.
#include "tools/datasets/parameter_dataset.hpp"
#include "datasets/validation.hpp"

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
        "zero_coupon_bond_options_01/generation.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/product/"
        "zero_coupon_bond_options/zero_coupon_bond_options_01.json";

    std::vector<float> core_strikes;
    core_strikes.reserve(25U);
    for (const float coordinate : linear_grid(-1.0f, 1.0f, 25U)) {
        const float cube = coordinate * coordinate * coordinate;
        core_strikes.push_back(
            coordinate < 0.0f ? 0.975f + 0.105f * cube
                              : 0.975f + 0.025f * cube
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
        "Expiry/tenor grid with strike ratios concentrated below the initial forward bond price."
    );
    GeneratedRows stress = regime(
        linear_business_day_grid(21U, 1890U, 10U),
        {63U, 1260U},
        {0.78f, 0.87f, 0.95f, 0.99f, 1.03f},
        "One-month to 7.5-year expiries with moderate dimensionless strike-ratio tails."
    );
    const GeneratedRows rows = core_stress_rows(
        std::move(core), std::move(stress)
    );

    write_fixed_income_product_dataset(
        "zero_coupon_bond_options_01",
        "Zero-Coupon Bond Options",
        dataset_path,
        catalog_path,
        url,
        {
            {"notional", "Normalized contract notional."},
            {"strike", "Dimensionless strike ratio to P(0,U)/P(0,T)."},
            {"option_expiry", "Option expiry S in business days."},
            {"bond_maturity", "Underlying bond maturity T > S in business days."},
        },
        {
            {"expression", "N * max(side * (P(S,T) - K), 0), side = +1 call / -1 put"},
            {"payment_time", "S"},
            {"normalization", "N = 1"},
        },
        rows,
        { {"strike", { {"mode", "ratio_to_initial_forward_bond_price"}, {"unit", "dimensionless"}, {"effective_level", "strike = strike_ratio * P(0,U)/P(0,T)"} } },
          {"reference_policy", "Resolved at the unbumped model and initial curve; absolute contract levels stay fixed for model and curve bumps."} }
    );
    validate_product_dataset_file(dataset_path);
}
