// Generate 900 core and 100 stress unit-bond up-and-out call contracts.
#include "tools/datasets/parameter_dataset.hpp"
#include "common/dataset_validation.hpp"
#include "product/zero_coupon_bond_up_and_out/dataset.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

int main() {
    using namespace ai_factory::workbench::datasets;
    const std::filesystem::path dataset_path =
        "datasets/product/zero_coupon_bond_up_and_out/zero_coupon_bond_up_and_outs_01.json";
    const std::filesystem::path catalog_path =
        "catalog/product/zero_coupon_bond_up_and_out/zero_coupon_bond_up_and_outs_01/generation.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/product/zero_coupon_bond_up_and_out/zero_coupon_bond_up_and_outs_01.json";
    const auto regime = [](
        const std::vector<std::uint32_t>& expiries,
        const std::vector<std::uint32_t>& tenors,
        const std::vector<float>& strikes,
        const std::vector<float>& barriers,
        const std::string& description) {
        GeneratedRows result;
        for (const auto expiry : expiries)
            for (const auto tenor : tenors)
                for (const auto strike : strikes)
                    for (const auto barrier : barriers)
                        result.rows.push_back({
                            {"notional", 1.0f}, {"strike", strike},
                            {"barrier", barrier}, {"option_expiry", expiry},
                            {"bond_maturity", expiry + tenor}});
        result.construction = {
            {"method", "Cartesian grid"},
            {"rule", "Every expiry, bond tenor, strike, and barrier combination."},
            {"description", description}, {"option_expiries", expiries},
            {"bond_tenors", tenors}, {"strikes", strikes},
            {"barriers", barriers}, {"notional", 1.0f}};
        return result;
    };
    auto core = regime(
        {1U, 2U, 3U, 5U, 10U}, {21U, 63U, 126U},
        {0.82f, 0.86f, 0.90f, 0.94f, 0.97f, 0.99f},
        {1.015f, 1.02f, 1.03f, 1.04f, 1.06f,
         1.08f, 1.10f, 1.13f, 1.17f, 1.22f},
        "Short expiry grid with all core upper barriers above the initial bond price.");
    auto stress = regime(
        {1U, 5U, 21U, 42U, 63U}, {21U, 252U},
        {0.75f, 1.01f}, {1.008f, 1.012f, 1.025f, 1.06f, 1.30f},
        "Longer expiries and near or remote upper-barrier ratios.");
    const auto rows = core_stress_rows(std::move(core), std::move(stress));
    write_fixed_income_product_dataset(
        "zero_coupon_bond_up_and_outs_01", "Zero-Coupon Bond Up-and-Out Calls",
        dataset_path, catalog_path, url,
        {{"notional", "Units of a zero-coupon bond paying 1 currency unit at U."},
         {"strike", "Dimensionless ratio to P(0,U)/P(0,T)."},
         {"barrier", "Dimensionless ratio to P(0,U); exceeds 1 in core rows."},
         {"option_expiry", "Payoff date T in business days."},
         {"bond_maturity", "Bond face-value date U > T in business days."}},
        {{"expression", "N * exp(-integral_0^T r_s ds) * max(P(T,U)-K,0) * 1{P(t,U)<B for all monitored t}"},
         {"payment_time", "T"}, {"normalization", "N = 1"},
         {"barrier_breach", "P(t,U) >= B, including t=0"},
         {"monitoring", "numerical grid; continuous-barrier price is approximated"}},
        rows,
        { {"strike", { {"mode", "ratio_to_initial_forward_bond_price"}, {"unit", "dimensionless"}, {"effective_level", "strike = strike_ratio * P(0,U)/P(0,T)"} } },
          {"barrier", { {"mode", "ratio_to_initial_bond_price"}, {"unit", "dimensionless"}, {"effective_level", "barrier = barrier_ratio * P(0,U)"} } },
          {"reference_policy", "Resolved at the unbumped model and initial curve; absolute K and B stay fixed for model and curve bumps."} }
);
    ai_factory::workbench::product::load_zero_coupon_bond_up_and_outs(dataset_path);
    validate_product_dataset_file(dataset_path);
}
