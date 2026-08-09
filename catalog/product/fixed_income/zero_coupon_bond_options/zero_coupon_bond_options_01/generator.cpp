// Generate options on zero-coupon bonds with broad strike coverage.
#include "tools/datasets/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <filesystem>
#include <cmath>
#include <string>
#include <utility>

namespace {

std::vector<float> actual_360_grid(
    int first_day,
    int last_day,
    std::size_t point_count
) {
    std::vector<float> times;
    times.reserve(point_count);
    for (std::size_t index = 0U; index < point_count; ++index) {
        const double weight = static_cast<double>(index)
            / static_cast<double>(point_count - 1U);
        const int day = static_cast<int>(std::lround(
            first_day + weight * (last_day - first_day)
        ));
        times.push_back(static_cast<float>(day) / 360.0f);
    }
    return times;
}

}  // namespace

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

    std::vector<float> core_strikes;
    core_strikes.reserve(25U);
    for (const float coordinate : linear_grid(-1.0f, 1.0f, 25U)) {
        const float cube = coordinate * coordinate * coordinate;
        core_strikes.push_back(
            coordinate < 0.0f ? 0.97f + 0.17f * cube
                              : 0.97f + 0.13f * cube
        );
    }
    const auto regime = [](const std::vector<float>& option_expiries,
                           const std::vector<float>& bond_tenors,
                           const std::vector<float>& strikes,
                           const std::string& description) {
        GeneratedRows generated;
        for (const float option_expiry : option_expiries) {
            for (const float bond_tenor : bond_tenors) {
                for (const float strike : strikes) {
                    generated.rows.push_back({
                    {"notional", 1.0f},
                    {"strike", strike},
                    {"option_expiry", option_expiry},
                    {"bond_maturity", option_expiry + bond_tenor},
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
        actual_360_grid(90, 1800, 18U),
        {0.5f, 1.0f},
        core_strikes,
        "Representative expiries and strikes concentrated around 0.97."
    );
    GeneratedRows stress = regime(
        actual_360_grid(7, 5400, 10U),
        {30.0f / 360.0f, 10.0f},
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
